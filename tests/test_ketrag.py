"""KET-RAG faithful mode: K/2 keyword-overlap + K/2 embedding neighbours, PageRank."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.chunking import Chunk
from src.corpus import build_chunk_manifest
from src.embeddings import TfidfEmbedder
from strategies.ketrag_strategy import KETRAGStrategy, top_neighbors


def _chunks():
    chunks, _, _ = build_chunk_manifest("hotpotqa", "mock", 5, 42, 8, 2)
    return chunks


class FixedEmbedder:
    """Hand-made vectors so the semantic neighbours are known in advance."""
    name = "fixed"

    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=float)

    def fit_transform(self, texts):
        return self.vectors


def test_top_neighbors_excludes_self_breaks_ties_by_id_and_skips_zero():
    sim = np.array([[9, 5, 5, 0], [5, 9, 1, 0], [5, 1, 9, 0], [0, 0, 0, 9]])
    order = np.array([0, 1, 2, 3])
    assert top_neighbors(sim, 0, 2, order) == [1, 2]      # tie 5/5 -> lower chunk_id rank first
    assert top_neighbors(sim, 0, 1, np.array([0, 2, 1, 3])) == [2]
    assert top_neighbors(sim, 3, 2, order) == []          # zero similarity is not a neighbour


def test_faithful_graph_uses_keyword_and_semantic_neighbours():
    # c1-c2 share keywords only; c1-c3 are close in embedding space only.
    chunks = [
        Chunk("c1", "zemeckis directed gump", "A", 0, 3),
        Chunk("c2", "zemeckis directed hanks", "B", 0, 3),
        Chunk("c3", "unrelated words entirely", "C", 0, 3),
        Chunk("c4", "tuscaloosa alabama campus", "D", 0, 3),
    ]
    vectors = [[1, 0, 0], [0, 1, 0], [0.99, 0.14, 0], [0, 0, 1]]
    s = KETRAGStrategy(knn_k=2, embedder=FixedEmbedder(vectors))
    graph = s._faithful_graph(chunks)
    assert graph.has_edge("c1", "c2")      # lexical neighbour (2 shared keywords)
    assert graph.has_edge("c1", "c3")      # semantic neighbour (cosine 0.99)
    assert s.rank(chunks)[0] == "c1"       # linked both ways -> most central


def test_each_chunk_adds_at_most_k_edges():
    chunks = _chunks()
    s = KETRAGStrategy(knn_k=4, embedder=TfidfEmbedder())
    graph = s._faithful_graph(chunks)
    assert graph.number_of_edges() <= 4 * len(chunks)
    assert set(graph.nodes) == {c.chunk_id for c in chunks}


def test_faithful_is_deterministic_and_differs_from_tfidf_mode():
    chunks = _chunks()
    faithful = lambda: KETRAGStrategy(knn_k=2, embedder=TfidfEmbedder()).rank(chunks)
    assert faithful() == faithful()
    tfidf = KETRAGStrategy(knn_k=10, mode="tfidf").rank(chunks)
    assert sorted(tfidf) == sorted(faithful()) and tfidf != faithful()


def test_tfidf_mode_matches_phase1_behaviour():
    # mode="tfidf" must reproduce the original Phase 1 ranking exactly.
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    chunks = _chunks()
    ids = [c.chunk_id for c in chunks]
    sim = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform([c.text for c in chunks]))
    np.fill_diagonal(sim, 0.0)
    g = nx.Graph()
    g.add_nodes_from(ids)
    for i, cid in enumerate(ids):
        for j in np.argsort(sim[i])[::-1][:3]:
            if sim[i, j] > 0:
                g.add_edge(cid, ids[j], weight=float(sim[i, j]))
    pr = nx.pagerank(g, weight="weight")
    expected = sorted(ids, key=lambda cid: (-pr.get(cid, 0.0), cid))
    assert KETRAGStrategy(knn_k=3, mode="tfidf").rank(chunks) == expected


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        KETRAGStrategy(mode="nope")


def test_missing_embedding_model_gives_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="ketrag_mode: tfidf"):
        KETRAGStrategy(knn_k=2).rank(_chunks())


def test_faithful_with_real_sentence_transformer():
    pytest.importorskip("sentence_transformers")
    chunks = _chunks()
    s = KETRAGStrategy(knn_k=2)            # default embedder: all-MiniLM-L6-v2
    ranking = s.rank(chunks)
    assert sorted(ranking) == sorted(c.chunk_id for c in chunks)
    assert s.last_run_info["embedder"] == "all-MiniLM-L6-v2"
    assert ranking == KETRAGStrategy(knn_k=2).rank(chunks)
