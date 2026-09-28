"""Tests for the LazyGraphRAG (L3) and FastGraphRAG (F1) strategies and the
native LazyGraphRAG (L4) reference point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy

import pytest

from src.budget import select_top_budget
from src.chunking import Chunk
from src.config import ExperimentConfig
from src.corpus import build_chunk_manifest
from src.embeddings import TfidfEmbedder
from native.lazygraphrag_native import NativeLazyGraphRAG
from strategies import build_strategy
from strategies.fastgraphrag_strategy import FastGraphRAGStrategy
from strategies.ketrag_strategy import KETRAGStrategy
from strategies.lazygraphrag_strategy import LazyGraphRAGStrategy
from strategies.random_strategy import RandomStrategy


def _chunks():
    chunks, _, _ = build_chunk_manifest(
        dataset="hotpotqa", data_source="mock", num_questions=5, seed=42,
        chunk_size_words=8, chunk_overlap_words=2,
    )
    return chunks


NEW = {
    "lazygraphrag": lambda: LazyGraphRAGStrategy(seed=0),
    "fastgraphrag": lambda: FastGraphRAGStrategy(use_spacy=False),
    "fastgraphrag_spacy": lambda: FastGraphRAGStrategy(use_spacy=True),   # falls back if spaCy is missing
}


@pytest.mark.parametrize("name", NEW)
def test_full_permutation_and_deterministic(name):
    chunks = _chunks()
    r1, r2 = NEW[name]().rank(chunks), NEW[name]().rank(chunks)
    assert sorted(r1) == sorted(c.chunk_id for c in chunks) and len(r1) == len(set(r1))
    assert r1 == r2


@pytest.mark.parametrize("name", NEW)
def test_never_reads_gold_labels(name):
    # Leakage guard (blueprint §S): changing the analysis-only gold labels
    # must not change the ranking at all.
    chunks = _chunks()
    scrambled = copy.deepcopy(chunks)
    for c in scrambled:
        c.is_gold_for_question_ids = ["q1", "q2", "q3"] if c.is_gold_for_question_ids == [] else []
    assert NEW[name]().rank(chunks) == NEW[name]().rank(scrambled)


def test_all_four_strategies_differ_but_converge_at_100_percent():
    chunks = _chunks()
    rankings = {
        "random": RandomStrategy(seed=1).rank(chunks),
        "ketrag": KETRAGStrategy(knn_k=3, embedder=TfidfEmbedder()).rank(chunks),
        "lazygraphrag": LazyGraphRAGStrategy(seed=0).rank(chunks),
        "fastgraphrag": FastGraphRAGStrategy(use_spacy=False).rank(chunks),
    }
    orders = list(rankings.values())
    assert all(a != b for i, a in enumerate(orders) for b in orders[i + 1:]), "rankings must be distinct"
    all_ids = {c.chunk_id for c in chunks}
    assert all(set(select_top_budget(r, 1.0)) == all_ids for r in orders)   # blueprint §R


def test_lazy_covers_every_topic_first():
    chunks = _chunks()
    s = LazyGraphRAGStrategy(n_clusters=4, seed=0)
    ranking = s.rank(chunks)
    assert s.last_run_info["n_clusters"] == 4
    # first round of the round-robin = one chunk from each of the 4 topics
    from sklearn.cluster import KMeans
    vectors = s.embedder.fit_transform([c.text for c in chunks])
    labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit(vectors).labels_
    idx = {c.chunk_id: i for i, c in enumerate(chunks)}
    assert len({labels[idx[cid]] for cid in ranking[:4]}) == 4


def test_fast_prefers_entity_dense_chunks():
    chunks = [
        Chunk("a", "the weather was calm and nothing happened", "Weather", 0, 7),
        Chunk("b", "Tom Hanks met Robert Zemeckis at Paramount Pictures", "Meeting", 0, 8),
        Chunk("c", "Tom Hanks starred in Forrest Gump", "Film", 0, 6),
    ]
    ranking = FastGraphRAGStrategy(use_spacy=False).rank(chunks)
    assert ranking[0] == "b" and ranking[-1] == "a"


def test_fast_uses_spacy_when_available():
    s = FastGraphRAGStrategy(use_spacy=True)
    if s.nlp is None:
        pytest.skip("spaCy or en_core_web_sm not installed")
    s.rank(_chunks())
    assert s.last_run_info["ner_backend"] == "spacy:en_core_web_sm"
    assert s.last_run_info["n_entities"] > 0


@pytest.mark.parametrize("strategy", ["random", "ketrag", "lazygraphrag", "fastgraphrag"])
def test_build_strategy_from_config(strategy):
    cfg = ExperimentConfig(experiment_id="t", seed=0, dataset="hotpotqa", num_questions=5,
                           strategy=strategy, budget=0.1, fast_use_spacy=False, ketrag_mode="tfidf")
    assert build_strategy(cfg).name == strategy


# ---------------------------------------------------------------- native L4
def _keyword_relevance(question: str, text: str) -> bool:
    """Free stand-in for the LLM yes/no check: >= 2 shared content words."""
    stop = {"the", "of", "a", "in", "what", "was", "is", "by", "who", "which", "did", "for"}
    words = lambda s: {w.strip("?.,:").lower() for w in s.split()} - stop
    return len(words(question) & words(text)) >= 2


def test_native_index_uses_no_llm():
    info = NativeLazyGraphRAG(use_spacy=False).index(_chunks())
    assert info["index_llm_calls"] == 0 and info["index_cost_usd"] == 0.0
    assert info["n_communities"] > 0


def test_native_respects_relevance_budget():
    calls = []

    def counting_fn(q, t):
        calls.append(t)
        return _keyword_relevance(q, t)

    system = NativeLazyGraphRAG(relevance_budget=4, per_community=2, max_relevant=10, use_spacy=False)
    system.index(_chunks())
    _, stats = system.retrieve("Who directed the film Forrest Gump?", counting_fn)
    assert stats["relevance_tests"] == len(calls) <= 4


def test_native_finds_relevant_chunks():
    system = NativeLazyGraphRAG(relevance_budget=10, use_spacy=False)
    system.index(_chunks())
    found, stats = system.retrieve("Which studio produced the film Forrest Gump?", _keyword_relevance)
    assert stats["n_relevant"] == len(found) > 0
    assert any(c.source_title == "Forrest Gump" for c in found)


def test_native_requires_index_first():
    with pytest.raises(RuntimeError):
        NativeLazyGraphRAG(use_spacy=False).retrieve("q", _keyword_relevance)
