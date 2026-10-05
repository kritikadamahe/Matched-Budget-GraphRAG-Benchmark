"""Blueprint Phase 8: knowledge-graph construction from extraction results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random

import pytest

from extraction import MockExtractor
from extraction.pipeline import run_extraction
from extraction.schemas import Entity, ExtractionResult, Relationship
from graph.build import main as graph_cli
from graph.graph_builder import build_graph, check_provenance, load_graph, normalize_entity, save_graph
from src.corpus import build_chunk_manifest
from strategies.fastgraphrag_strategy import FastGraphRAGStrategy
from strategies.random_strategy import RandomStrategy


def res(chunk_id, entities=(), relationships=(), status="ok"):
    return ExtractionResult(
        chunk_id=chunk_id, status=status, extractor="mock", model="mock-v1", prompt_version="t",
        entities=[Entity(name=n, type=t, description=d) for n, t, d in entities],
        relationships=[Relationship(source=s, target=o, description=d) for s, o, d in relationships],
    )


GUMP = res("c1",
           [("Forrest Gump", "WORK", "A 1994 film."), ("Robert Zemeckis", "PERSON", "Director.")],
           [("Forrest Gump", "Robert Zemeckis", "directed by")])
ZEMECKIS = res("c2",
               [("Robert Zemeckis", "PERSON", "Born in Chicago."), ("Chicago", "LOCATION", "A city.")],
               [("Robert Zemeckis", "Chicago", "born in")])


# ------------------------------------------------------------ normalisation
@pytest.mark.parametrize("raw,key", [
    ("The Beatles", "beatles"), ("Beatles", "beatles"), ("beatles.", "beatles"),
    ("U.S.", "us"), ("St. Louis", "st louis"), ("  Robert   Zemeckis ", "robert zemeckis"),
    ("O'Neil", "oneil"), ("AC/DC", "ac dc"), ("Theodore Roosevelt", "theodore roosevelt"),
    ("Ｆｕｌｌｗｉｄｔｈ", "fullwidth"), ("...", ""),
])
def test_normalize_entity(raw, key):
    assert normalize_entity(raw) == key


# ------------------------------------------------------------------ merging
def test_entities_merge_across_chunks_with_provenance():
    G, stats = build_graph([GUMP, ZEMECKIS])
    assert set(G.nodes) == {"forrest gump", "robert zemeckis", "chicago"}
    z = G.nodes["robert zemeckis"]
    assert z["chunk_ids"] == ["c1", "c2"] and z["mentions"] == 2
    assert z["descriptions"] == ["Director.", "Born in Chicago."]      # all kept, no LLM summary
    assert G.number_of_edges() == 2 and stats["n_chunks_used"] == 2
    edge = next(iter(G.get_edge_data("forrest gump", "robert zemeckis").values()))
    assert edge == {"description": "directed by", "chunk_id": "c1"}


def test_majority_type_and_all_types_kept():
    results = [res("a", [("Paris", "LOCATION", "")]), res("b", [("Paris", "LOCATION", "")]),
               res("c", [("Paris", "WORK", "a film")])]
    node = build_graph(results)[0].nodes["paris"]
    assert node["type"] == "LOCATION" and node["type_counts"] == {"LOCATION": 2, "WORK": 1}


def test_type_tie_is_broken_alphabetically():
    node = build_graph([res("a", [("X", "WORK", "")]), res("b", [("X", "EVENT", "")])])[0].nodes["x"]
    assert node["type"] == "EVENT"


def test_relationship_collapsing_into_self_loop_is_dropped_and_counted():
    # Distinct names in the chunk, identical after normalisation.
    r = res("c", [("The Beatles", "ORGANIZATION", ""), ("Beatles", "ORGANIZATION", "")],
            [("The Beatles", "Beatles", "same band")])
    G, stats = build_graph([r])
    assert list(G.nodes) == ["beatles"] and G.number_of_edges() == 0 and stats["dropped_self_loops"] == 1


def test_failed_chunks_contribute_nothing_but_are_counted():
    G, stats = build_graph([GUMP, res("c9", status="truncated")])
    assert "c9" not in {c for _, d in G.nodes(data=True) for c in d["chunk_ids"]}
    assert stats["n_chunks_skipped_not_ok"] == 1 and stats["skipped_chunk_ids"] == ["c9"]


def test_graph_is_independent_of_result_order():
    results = [GUMP, ZEMECKIS, res("c3", [("Chicago", "LOCATION", "Windy city.")])]
    a = build_graph(results)[0]
    shuffled = results[:]
    random.Random(1).shuffle(shuffled)
    b = build_graph(shuffled)[0]
    assert list(a.nodes(data=True)) == list(b.nodes(data=True))
    assert list(a.edges(data=True)) == list(b.edges(data=True))


def test_empty_input_gives_empty_graph():
    G, stats = build_graph([])
    assert G.number_of_nodes() == 0 and stats["n_components"] == 0


# --------------------------------------------------------------- provenance
def test_provenance_check_passes_for_selected_chunks_and_catches_orphans():
    G, _ = build_graph([GUMP, ZEMECKIS])
    check_provenance(G, ["c1", "c2"])
    with pytest.raises(ValueError, match="unselected"):
        check_provenance(G, ["c1"])
    G.nodes["chicago"]["chunk_ids"] = []
    with pytest.raises(ValueError, match="no provenance"):
        check_provenance(G, ["c1", "c2"])


def test_save_and_load_round_trip(tmp_path):
    G, _ = build_graph([GUMP, ZEMECKIS])
    save_graph(G, tmp_path / "g.json")
    H = load_graph(tmp_path / "g.json")
    assert list(G.nodes(data=True)) == list(H.nodes(data=True))
    assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))


# -------------------------------------------------------- with the pipeline
def _chunks():
    return build_chunk_manifest("hotpotqa", "mock", 5, 42, 8, 2)[0]


def test_all_strategies_give_the_same_graph_at_100_percent(tmp_path):
    # Blueprint §R sanity check carried through extraction and graph building.
    chunks = _chunks()
    graphs = []
    for strategy in (RandomStrategy(seed=1), FastGraphRAGStrategy(use_spacy=False)):
        run = run_extraction(chunks, strategy.rank(chunks), 1.0, MockExtractor())
        graphs.append(build_graph(run.results)[0])
    assert list(graphs[0].nodes(data=True)) == list(graphs[1].nodes(data=True))
    assert list(graphs[0].edges(data=True)) == list(graphs[1].edges(data=True))


def test_smaller_budget_gives_a_subgraph():
    chunks = _chunks()
    ranking = RandomStrategy(seed=3).rank(chunks)
    small = build_graph(run_extraction(chunks, ranking, 0.25, MockExtractor()).results)[0]
    large = build_graph(run_extraction(chunks, ranking, 0.75, MockExtractor()).results)[0]
    assert set(small.nodes) <= set(large.nodes)


def test_cli_builds_graph_from_an_extraction_run(tmp_path, capsys):
    from extraction.run import main as extraction_cli
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("\n".join([
        "experiment_id: graph_cli", "seed: 42", "dataset: hotpotqa", "data_source: mock",
        "num_questions: 5", "strategy: random", "budget: 0.5", "embedding_backend: tfidf",
        "ketrag_mode: tfidf", "fast_use_spacy: false", "extraction_backend: mock",
        f"cache_dir: {tmp_path / 'cache'}", f"output_dir: {tmp_path / 'results'}",
    ]))
    assert graph_cli(["--config", str(cfg)]) == 2            # no extraction yet -> clear stop
    assert extraction_cli(["--config", str(cfg)]) == 0
    assert graph_cli(["--config", str(cfg)]) == 0
    (out,) = list((tmp_path / "results" / "graphs").iterdir())
    stats = json.loads((out / "graph_stats.json").read_text())
    assert stats["provenance_check"] == "passed" and stats["n_nodes"] > 0
    assert load_graph(out / "graph.json").number_of_nodes() == stats["n_nodes"]
