"""
Native LazyGraphRAG, option L4 (blueprint §B, "Path 2") - a REFERENCE POINT,
not one of the budgeted strategies.

Unlike strategies/lazygraphrag_strategy.py (the L3 adaptation), this follows
LazyGraphRAG's actual design (Edge et al., Microsoft Research, 2024), simplified:

INDEX TIME - no LLM at all, so indexing cost is $0:
  1. extract "concepts" per chunk with cheap NLP: spaCy noun phrases, or the
     capitalised-phrase rule if spaCy is not installed;
  2. build a concept co-occurrence graph and split it into communities (Louvain);
  3. embed every chunk with the shared embedder (cached semantic model when built
     with build_native(cfg); TF-IDF if no embedder is given).

QUERY TIME - LLM spend capped per question by `relevance_budget`:
  4. rank communities by the best question-chunk similarity inside them;
  5. walk communities in that order; in each, test its most similar untested
     chunks (`per_community` per community) with a yes/no relevance check;
  6. stop when `relevance_budget` tests are used or `max_relevant` relevant
     chunks are found; the relevant chunks are the context for answering.

The relevance check is INJECTED as `relevance_fn(question, chunk_text) -> bool`.
In the real pipeline it will be one cheap GPT-4o-mini yes/no call (added with
the LLM client in roadmap Phase 7); tests pass a free fake. Every call counts
toward QUERY-time cost, never index cost (blueprint §M).

On the results plot this is ONE point per dataset at ~0% index budget, with
variable per-question cost - the honest comparison next to the 48 matched-budget
conditions.
"""

from __future__ import annotations

import itertools
from typing import Callable

import networkx as nx
import numpy as np

from src.chunking import Chunk
from src.embeddings import TfidfEmbedder
from src.text_utils import capitalized_phrases, load_spacy, normalize_name

RelevanceFn = Callable[[str, str], bool]


class NativeLazyGraphRAG:
    name = "lazygraphrag_native"

    def __init__(self, relevance_budget: int = 20, per_community: int = 3, max_relevant: int = 5,
                 use_spacy: bool = True, spacy_model: str = "en_core_web_sm", seed: int = 0, embedder=None):
        self.relevance_budget = relevance_budget
        self.per_community = per_community
        self.max_relevant = max_relevant
        self.seed = seed
        self.embedder = embedder or TfidfEmbedder()
        self.nlp = load_spacy(spacy_model, disable=["ner", "lemmatizer"]) if use_spacy else None
        self.concept_backend = f"spacy-noun-chunks:{spacy_model}" if self.nlp else "regex-capitalised"
        self._indexed = False

    # ------------------------------------------------------------ index time
    def extract_concepts(self, text: str) -> set[str]:
        if self.nlp is not None:
            phrases = [nc.text for nc in self.nlp(text).noun_chunks if nc.root.pos_ != "PRON"]
        else:
            phrases = capitalized_phrases(text)
        return {normalize_name(p) for p in phrases if len(p.strip()) > 2}

    def index(self, chunks: list[Chunk]) -> dict:
        self.chunks = chunks
        concepts = [self.extract_concepts(c.text) for c in chunks]

        graph = nx.Graph()
        for cs in concepts:
            graph.add_nodes_from(cs)
            graph.add_edges_from(itertools.combinations(sorted(cs), 2))
        communities = nx.community.louvain_communities(graph, seed=self.seed) if graph.number_of_nodes() else []
        concept_to_comm = {c: i for i, comm in enumerate(communities) for c in comm}

        # A chunk belongs to every community its concepts fall in. Chunks with no
        # concepts get a singleton community so they remain reachable.
        self.community_chunks: dict[int, list[int]] = {}
        next_id = len(communities)
        for i, cs in enumerate(concepts):
            comms = {concept_to_comm[c] for c in cs}
            if not comms:
                comms = {next_id}
                next_id += 1
            for comm in comms:
                self.community_chunks.setdefault(comm, []).append(i)

        self.vectors = self.embedder.fit_transform([c.text for c in chunks])
        self._indexed = True
        return {"index_llm_calls": 0, "index_cost_usd": 0.0, "concept_backend": self.concept_backend,
                "n_concepts": graph.number_of_nodes(), "n_communities": len(self.community_chunks)}

    # ------------------------------------------------------------ query time
    @staticmethod
    def build_context(relevant_chunks: list[Chunk], max_words: int = 1500) -> str:
        """Context for the shared answer generator: the relevant passages, numbered,
        within the same word limit as graph retrieval (so answering is comparable)."""
        passages, used = [], 0
        for c in relevant_chunks:
            n = len(c.text.split())
            if used + n + 1 <= max_words:
                passages.append(f"[{len(passages) + 1}] {c.text}")
                used += n + 1
        return "Passages:\n" + "\n".join(passages) if passages else ""

    def retrieve(self, question: str, relevance_fn: RelevanceFn) -> tuple[list[Chunk], dict]:
        """Return (relevant chunks, stats) for one question."""
        if not self._indexed:
            raise RuntimeError("call index(chunks) first")
        sims = self.vectors @ self.embedder.transform([question])[0]
        ranked_comms = sorted(self.community_chunks,
                              key=lambda k: (-max(sims[i] for i in self.community_chunks[k]), k))

        tested: set[int] = set()
        relevant: list[int] = []

        def done() -> bool:
            return len(tested) >= self.relevance_budget or len(relevant) >= self.max_relevant

        for comm in ranked_comms:
            if done():
                break
            candidates = sorted((i for i in self.community_chunks[comm] if i not in tested),
                                key=lambda i: (-sims[i], self.chunks[i].chunk_id))
            for i in candidates[: self.per_community]:
                if done():
                    break
                tested.add(i)
                if relevance_fn(question, self.chunks[i].text):
                    relevant.append(i)

        return [self.chunks[i] for i in relevant], {"relevance_tests": len(tested),
                                                    "n_relevant": len(relevant)}
