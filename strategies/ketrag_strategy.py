"""
KET-RAG strategy (blueprint §H) - Category A/B: faithful to KET-RAG's own
budget mechanism (its beta parameter IS our budget), our specific choice
of HOW to build the similarity graph is an implementation detail.

WHAT IT DOES:
1. Represent every chunk's text as a TF-IDF vector (keyword-overlap based -
   one of the two options the blueprint allowed; no embeddings/API needed).
2. Build a k-nearest-neighbor graph: connect each chunk to its k most
   textually-similar other chunks (cosine similarity on TF-IDF vectors).
3. Run PageRank on that graph. Chunks that are textually "central" -
   similar to many other chunks - get high scores. This mirrors the real
   KET-RAG idea: central chunks are worth the expensive extraction because
   they connect to more of the rest of the corpus.
4. Rank chunks by PageRank score, descending. Tie-break by chunk_id for
   determinism.

WHY A GRAPH BEFORE EXTRACTION: this is what makes KET-RAG fundamentally
different from Random. Random needs nothing but chunk_ids. KET-RAG needs
to look at ALL chunks' content first (cheaply, no LLM) to decide which
ones are structurally important - that's the whole idea of the method.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking import Chunk
from strategies.base_strategy import SelectionStrategy


class KETRAGStrategy(SelectionStrategy):
    name = "ketrag"

    def __init__(self, knn_k: int = 10, seed: int = 0):
        self.knn_k = knn_k
        self.seed = seed  # used only for tie-breaking, PageRank itself is deterministic

    def rank(self, chunks: list[Chunk]) -> list[str]:
        if len(chunks) == 1:
            return [chunks[0].chunk_id]

        chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]

        # Step 1: TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Step 2: pairwise cosine similarity, then keep only each chunk's
        # top-k neighbors (k-NN graph) - full pairwise would be O(N^2) edges,
        # which doesn't scale and isn't what a k-NN similarity graph is.
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0.0)  # a chunk is not its own neighbor

        graph = nx.Graph()
        graph.add_nodes_from(chunk_ids)

        k = min(self.knn_k, len(chunks) - 1)
        for i, chunk_id in enumerate(chunk_ids):
            neighbor_indices = np.argsort(sim_matrix[i])[::-1][:k]
            for j in neighbor_indices:
                weight = float(sim_matrix[i, j])
                if weight > 0:
                    graph.add_edge(chunk_id, chunk_ids[j], weight=weight)

        # Step 3: PageRank centrality
        scores = nx.pagerank(graph, weight="weight")

        # Step 4: rank descending by score, tie-break by chunk_id for determinism
        ranked = sorted(chunk_ids, key=lambda cid: (-scores.get(cid, 0.0), cid))
        return ranked
