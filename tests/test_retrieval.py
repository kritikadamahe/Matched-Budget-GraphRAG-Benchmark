"""Blueprint Phase 9: personalised-PageRank retrieval over the knowledge graph."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from extraction import MockExtractor
from extraction.pipeline import run_extraction
from extraction.schemas import Entity, ExtractionResult, Relationship
from graph.graph_builder import build_graph
from retrieval import GraphRetriever, entity_card, retriever_from_config
from src.config import ExperimentConfig
from src.corpus import build_chunk_manifest
from src.embeddings import TfidfEmbedder
from strategies.random_strategy import RandomStrategy


def res(chunk_id, entities, relationships=()):
    return ExtractionResult(
        chunk_id=chunk_id, status="ok", extractor="mock", model="m", prompt_version="t",
        entities=[Entity(name=n, type=t, description=d) for n, t, d in entities],
        relationships=[Relationship(source=s, target=o, description=d) for s, o, d in relationships],
    )


CHUNKS = {
    "c1": "Forrest Gump: a 1994 film directed by Robert Zemeckis.",
    "c2": "Robert Zemeckis: an American director born in Chicago.",
    "c3": "Tuscaloosa: a city in Alabama, home of the University of Alabama.",
}
RESULTS = [
    res("c1", [("Forrest Gump", "WORK", "1994 film"), ("Robert Zemeckis", "PERSON", "film director")],
        [("Forrest Gump", "Robert Zemeckis", "directed by")]),
    res("c2", [("Robert Zemeckis", "PERSON", "born in Chicago"), ("Chicago", "LOCATION", "city in Illinois")],
        [("Robert Zemeckis", "Chicago", "born in")]),
    res("c3", [("Tuscaloosa", "LOCATION", "city in Alabama"), ("University of Alabama", "ORGANIZATION", "university")],
        [("University of Alabama", "Tuscaloosa", "located in")]),
]


def retriever(**kw):
    G, _ = build_graph(RESULTS)
    return GraphRetriever(G, CHUNKS, TfidfEmbedder(), **kw)


def test_entity_card_uses_name_and_descriptions():
    assert entity_card({"name": "Robert Zemeckis", "descriptions": ["film director", "born in Chicago"]}) == \
        "Robert Zemeckis: film director born in Chicago"
    assert entity_card({"name": "X", "descriptions": []}) == "X"


def test_multi_hop_question_reaches_the_second_hop():
    # "director of Forrest Gump" -> seed Forrest Gump -> PPR walks to Zemeckis -> Chicago
    r = retriever(k_seeds=1, top_m=3).retrieve("Where was the director of Forrest Gump born?")
    assert r.seeds == ["forrest gump"]
    assert set(r.top_entities) == {"forrest gump", "robert zemeckis", "chicago"}
    assert set(r.chunk_ids) == {"c1", "c2"} and "c3" not in r.chunk_ids
    assert "Robert Zemeckis -- born in -- Chicago" in r.context


def test_context_layout_facts_then_passages():
    ctx = retriever().retrieve("Forrest Gump director").context
    assert ctx.startswith("Facts:\n- ") and "\n\nPassages:\n[1] " in ctx


def test_word_budget_is_respected_and_oversized_passages_are_skipped():
    r = retriever(max_context_words=25, max_facts=0).retrieve("Forrest Gump Robert Zemeckis Chicago")
    assert r.n_words <= 25 + len("Passages:".split()) + len(r.chunk_ids)
    assert len(r.chunk_ids) >= 1


def test_empty_graph_gives_empty_context():
    G, _ = build_graph([])
    r = GraphRetriever(G, {}, TfidfEmbedder()).retrieve("anything")
    assert r.empty_graph and r.context == "" and r.chunk_ids == []


def test_deterministic():
    a = retriever().retrieve("Who directed Forrest Gump?")
    b = retriever().retrieve("Who directed Forrest Gump?")
    assert (a.context, a.seeds, a.chunk_ids) == (b.context, b.seeds, b.chunk_ids)


def test_edges_are_used_undirected():
    # The relationship points University -> Tuscaloosa; a seed on Tuscaloosa must still reach it.
    r = retriever(k_seeds=1, top_m=2).retrieve("Tuscaloosa city Alabama")
    assert r.seeds == ["tuscaloosa"] and "university of alabama" in r.top_entities


def test_only_extracted_chunks_can_appear():
    # Retrieval can only return chunks that were extracted (graph provenance) - never others.
    G, _ = build_graph(RESULTS[:1])
    r = GraphRetriever(G, CHUNKS, TfidfEmbedder()).retrieve("Chicago Tuscaloosa Alabama")
    assert set(r.chunk_ids) <= {"c1"}


def test_from_config_with_the_mock_pipeline(tmp_path):
    chunks, questions, _ = build_chunk_manifest("hotpotqa", "mock", 5, 42, 250, 40)
    run = run_extraction(chunks, RandomStrategy(seed=1).rank(chunks), 1.0, MockExtractor())
    G, _ = build_graph(run.results)
    cfg = ExperimentConfig(experiment_id="t", seed=0, dataset="hotpotqa", num_questions=5,
                           strategy="random", budget=1.0, embedding_backend="tfidf")
    r = retriever_from_config(cfg, G, chunks).retrieve(questions[0]["question"])
    assert r.context and len(r.seeds) == 5 and len(r.top_entities) <= 20 and r.n_words <= 1500 + 50


def test_inspect_cli(tmp_path, capsys):
    from extraction.run import main as extraction_cli
    from graph.build import main as graph_cli
    from retrieval.inspect import main as inspect_cli
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("\n".join([
        "experiment_id: ret_cli", "seed: 42", "dataset: hotpotqa", "data_source: mock",
        "num_questions: 5", "strategy: random", "budget: 1.0", "embedding_backend: tfidf",
        "ketrag_mode: tfidf", "fast_use_spacy: false", "extraction_backend: mock",
        f"cache_dir: {tmp_path / 'cache'}", f"output_dir: {tmp_path / 'results'}",
    ]))
    assert inspect_cli(["--config", str(cfg)]) == 2                       # no graph yet
    assert extraction_cli(["--config", str(cfg)]) == 0 and graph_cli(["--config", str(cfg)]) == 0
    capsys.readouterr()
    assert inspect_cli(["--config", str(cfg), "--n", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("Gold answer:") == 2 and "Seeds:" in out and "gold chunks retrieved" in out
