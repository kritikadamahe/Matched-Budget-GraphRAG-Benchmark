"""
KET-RAG strategy (blueprint §H) - Category A/B: faithful to KET-RAG's own
budget mechanism (its beta parameter IS our budget) and, in the default
"faithful" mode, to its core-chunk graph construction as well.

WHAT KET-RAG DOES (Huang, Zhang, Xiao, KDD 2025, §4.1 and Algorithm 3):
"KET-RAG first identifies a set of core text chunks ... based on their PageRank
centralities in an intermediate KNN graph." Each chunk is linked to
  - the top-K/2 chunks by LEXICAL similarity  (number of co-occurring keywords), and
  - the top-K/2 chunks by SEMANTIC similarity (cosine similarity of embeddings);
then the top ceil(beta * N) chunks by PageRank become the core chunks that get
full LLM extraction. KET-RAG's default is K = 2.

MODES:
- "faithful" (default): the construction above. Keywords = content words
  (lowercased, English stop words removed, >= 3 letters); embeddings come from a
  local sentence-transformers model (all-MiniLM-L6-v2, free, cached - see
  src/embeddings.py). KET-RAG itself used OpenAI text-embedding-3-small - using a
  local model is our documented choice.
- "tfidf": the Phase 1 version - one k-NN graph on TF-IDF cosine similarity,
  weighted PageRank. Kept as an ablation ("does the semantic half matter?").

Either way: rank by PageRank score, descending; ties broken by chunk_id.

WHY A GRAPH BEFORE EXTRACTION: this is what makes KET-RAG fundamentally
different from Random. Random needs nothing but chunk_ids. KET-RAG needs
to look at ALL chunks' content first (cheaply, no LLM) to decide which
ones are structurally important - that's the whole idea of the method.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking import Chunk
from strategies.base_strategy import SelectionStrategy

MODES = ("faithful", "tfidf")


def top_neighbors(similarity: np.ndarray, i: int, k: int, order: np.ndarray) -> list[int]:
    """Indices of the k most similar chunks to chunk i (never i itself), ties
    broken by chunk_id via the precomputed `order` rank, so results are deterministic.
    Only strictly positive similarities count as neighbours."""
    if k <= 0:
        return []
    row = similarity[i].astype(float)
    candidates = [j for j in np.lexsort((order, -row)) if j != i and row[j] > 0]
    return candidates[:k]


class KETRAGStrategy(SelectionStrategy):
    name = "ketrag"

    def __init__(self, knn_k: int = 2, seed: int = 0, mode: str = "faithful",
                 embedder=None, damping: float = 0.85):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.knn_k = knn_k        # K: total neighbours per chunk (KET-RAG default 2)
        self.seed = seed          # PageRank is deterministic; kept for a uniform interface
        self.mode = mode
        self.embedder = embedder  # faithful mode: anything with fit_transform(texts) -> vectors
        self.damping = damping    # PageRank damping (networkx default 0.85)
        self.last_run_info: dict = {}

    # --------------------------------------------------------------- public
    def rank(self, chunks: list[Chunk]) -> list[str]:
        if len(chunks) == 1:
            return [chunks[0].chunk_id]
        if self.mode == "tfidf":
            graph = self._tfidf_graph(chunks)
            scores = nx.pagerank(graph, alpha=self.damping, weight="weight")
        else:
            graph = self._faithful_graph(chunks)
            scores = nx.pagerank(graph, alpha=self.damping)

        self.last_run_info = {"mode": self.mode, "K": self.knn_k,
                              "n_edges": graph.number_of_edges(),
                              "embedder": getattr(self.embedder, "name", None)}
        chunk_ids = [c.chunk_id for c in chunks]
        return sorted(chunk_ids, key=lambda cid: (-scores.get(cid, 0.0), cid))

    # -------------------------------------------------------------- faithful
    def _faithful_graph(self, chunks: list[Chunk]) -> nx.Graph:
        chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        order = np.argsort(np.argsort(chunk_ids))   # rank of each chunk_id, for tie-breaks

        # Lexical similarity = number of co-occurring keywords.
        vectorizer = CountVectorizer(stop_words="english", binary=True,
                                     token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
        try:
            keywords = vectorizer.fit_transform(texts)
            lexical = (keywords @ keywords.T).toarray()
        except ValueError:                           # no keywords at all in the corpus
            lexical = np.zeros((len(chunks), len(chunks)))

        # Semantic similarity = cosine of embeddings (vectors are L2-normalised).
        vectors = self._get_embedder().fit_transform(texts)
        semantic = np.asarray(vectors) @ np.asarray(vectors).T

        k_lexical = (self.knn_k + 1) // 2           # K/2 each; an odd K gives lexical the extra one
        k_semantic = self.knn_k // 2
        graph = nx.Graph()
        graph.add_nodes_from(chunk_ids)
        for i, cid in enumerate(chunk_ids):
            for j in top_neighbors(lexical, i, k_lexical, order):
                graph.add_edge(cid, chunk_ids[j])
            for j in top_neighbors(semantic, i, k_semantic, order):
                graph.add_edge(cid, chunk_ids[j])
        return graph

    def _get_embedder(self):
        if self.embedder is None:
            try:
                import sentence_transformers  # noqa: F401  (fail early, with a clear message)
            except ImportError as e:
                raise ImportError(
                    "KET-RAG faithful mode needs a semantic embedding model: "
                    "pip install sentence-transformers (CPU-only torch is enough), "
                    "or set ketrag_mode: tfidf."
                ) from e
            from src.embeddings import CachedEmbedder
            self.embedder = CachedEmbedder()          # shared default model, cached
        return self.embedder

    # ----------------------------------------------------------------- tfidf
    def _tfidf_graph(self, chunks: list[Chunk]) -> nx.Graph:
        """Phase 1 construction: TF-IDF cosine k-NN graph with similarity weights."""
        chunk_ids = [c.chunk_id for c in chunks]
        tfidf_matrix = TfidfVectorizer(stop_words="english").fit_transform([c.text for c in chunks])
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0.0)  # a chunk is not its own neighbor

        graph = nx.Graph()
        graph.add_nodes_from(chunk_ids)
        k = min(self.knn_k, len(chunks) - 1)
        for i, chunk_id in enumerate(chunk_ids):
            for j in np.argsort(sim_matrix[i])[::-1][:k]:
                weight = float(sim_matrix[i, j])
                if weight > 0:
                    graph.add_edge(chunk_id, chunk_ids[j], weight=weight)
        return graph
