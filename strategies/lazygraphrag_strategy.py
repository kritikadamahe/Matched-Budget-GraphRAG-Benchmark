"""
LazyGraphRAG strategy, option L3 (blueprint §B, §H) - Category B: ADAPTATION.

WHAT THE REAL METHOD DOES (and why we can't copy it directly):
LazyGraphRAG (Edge et al., Microsoft Research, 2024) runs NO LLM at indexing
time and spends its LLM budget per question, at query time. It has no
"pick B% of chunks before extraction" step. That native behaviour is
implemented separately in native/lazygraphrag_native.py (option L4).

WHAT THIS ADAPTATION DOES:
We keep only LazyGraphRAG's "relevance by embedding similarity" idea and turn
it into a question-agnostic ranking, so - like every other strategy - it never
sees the questions (blueprint §S):
1. Embed every chunk with the shared embedder (the cached semantic model when
   built from a config - see src/embeddings.py; TF-IDF if none is given).
2. Cluster the vectors into k topics with k-means (k = sqrt(N/2) by default).
3. Score each chunk by cosine similarity to its own topic centre - how
   "typical" / representative it is of that topic.
4. Interleave topics round-robin (largest topic first), taking each topic's
   most typical remaining chunk per round. At low budgets every topic gets
   covered instead of one big topic swallowing the whole budget.

WHAT TO SAY IN THE REPORT: this removes LazyGraphRAG's headline properties
(near-zero index cost, query-adaptive spend); it is "a matched-budget
adaptation of LazyGraphRAG's relevance signal", not a reproduction.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans

from src.chunking import Chunk
from src.embeddings import TfidfEmbedder
from strategies.base_strategy import SelectionStrategy


class LazyGraphRAGStrategy(SelectionStrategy):
    name = "lazygraphrag"

    def __init__(self, n_clusters: int | None = None, seed: int = 0, embedder=None):
        self.n_clusters = n_clusters      # None -> sqrt(N/2)
        self.seed = seed                  # k-means initialisation
        self.embedder = embedder or TfidfEmbedder()
        self.last_run_info: dict = {}     # for logging selection overhead (blueprint §M)

    def rank(self, chunks: list[Chunk]) -> list[str]:
        n = len(chunks)
        if n <= 1:
            return [c.chunk_id for c in chunks]

        vectors = self.embedder.fit_transform([c.text for c in chunks])
        k = self.n_clusters or round(math.sqrt(n / 2))
        k = max(1, min(k, n))

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(vectors)
        centres = kmeans.cluster_centers_
        centres = centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
        labels = kmeans.labels_
        typicality = np.einsum("ij,ij->i", vectors, centres[labels])

        # One queue per topic, most typical chunk first (tie-break: chunk_id).
        topics: dict[int, list[tuple[float, str]]] = {}
        for i, c in enumerate(chunks):
            topics.setdefault(int(labels[i]), []).append((-float(typicality[i]), c.chunk_id))
        queues = [sorted(members) for _, members in
                  sorted(topics.items(), key=lambda kv: (-len(kv[1]), kv[0]))]

        # Round-robin across topics.
        ranked: list[str] = []
        depth = 0
        while len(ranked) < n:
            for queue in queues:
                if depth < len(queue):
                    ranked.append(queue[depth][1])
            depth += 1

        self.last_run_info = {"n_clusters": k, "embedder": self.embedder.name}
        return ranked
