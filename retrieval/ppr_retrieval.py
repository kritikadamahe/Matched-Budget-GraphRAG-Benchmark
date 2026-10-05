"""
Personalised-PageRank retrieval (blueprint Phase 9, §J).

One algorithm, identical for every strategy and budget - only the graph it runs
on differs (because different chunks were extracted). Steps for one question:

  1. Embed the question with the ONE shared embedder.
  2. Seeds = the k entities whose "entity card" (name + its descriptions, e.g.
     "Robert Zemeckis: American film director.") is most similar to the question.
  3. Personalised PageRank from the seeds (restart weights = their similarity),
     on the graph with edges treated as UNDIRECTED: the LLM's choice of source vs
     target ("A directed B" / "B was directed by A") is arbitrary wording.
  4. Keep the top-m entities by PageRank score.
  5. Context = "Facts:" (relationships between top entities, best first) followed
     by "Passages:" (the original text of the chunks those entities came from,
     ordered by the PageRank mass of their entities), within a fixed word limit.
     A passage that does not fit is skipped and smaller ones are still tried.

An empty graph (possible at tiny budgets) gives an empty context - the answer
step then says "not found". That is a result, not a bug (blueprint §S).
Gold labels are never used here; the inspection CLI only shows them afterwards.

Defaults (agreed for Phase 9; all configurable, fixed across conditions):
k_seeds=5, damping=0.85, top_m=20, max_context_words=1500 (~2,000 tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

MAX_DESCRIPTIONS_PER_CARD = 3


@dataclass
class RetrievalResult:
    question: str
    context: str
    seeds: list[str] = field(default_factory=list)            # node keys, best first
    top_entities: list[str] = field(default_factory=list)     # node keys, best first
    facts: list[str] = field(default_factory=list)            # fact lines included in the context
    chunk_ids: list[str] = field(default_factory=list)        # passages included, in order
    n_words: int = 0
    empty_graph: bool = False


def entity_card(data: dict) -> str:
    """Text that represents one entity for seed matching: name + a few descriptions."""
    descriptions = " ".join(data.get("descriptions", [])[:MAX_DESCRIPTIONS_PER_CARD])
    return f"{data['name']}: {descriptions}" if descriptions else data["name"]


def _n_words(text: str) -> int:
    return len(text.split())


class GraphRetriever:
    def __init__(self, graph: nx.MultiDiGraph, chunk_text: dict[str, str], embedder,
                 k_seeds: int = 5, damping: float = 0.85, top_m: int = 20,
                 max_context_words: int = 1500, max_facts: int = 30):
        self.graph = graph
        self.chunk_text = chunk_text
        self.embedder = embedder
        self.k_seeds = k_seeds
        self.damping = damping
        self.top_m = top_m
        self.max_context_words = max_context_words
        self.max_facts = max_facts

        self.nodes = sorted(graph.nodes)
        # Undirected, parallel edges collapsed; weight = how many relationships link the pair.
        self.undirected = nx.Graph()
        self.undirected.add_nodes_from(self.nodes)
        for u, v in graph.edges():
            w = self.undirected.get_edge_data(u, v, {"weight": 0})["weight"]
            self.undirected.add_edge(u, v, weight=w + 1)
        self._node_vectors = None

    def _vectors(self) -> np.ndarray:
        if self._node_vectors is None:
            cards = [entity_card(self.graph.nodes[n]) for n in self.nodes]
            self._node_vectors = np.asarray(self.embedder.fit_transform(cards))   # fit: for TF-IDF
        return self._node_vectors

    # ----------------------------------------------------------------- query
    def retrieve(self, question: str) -> RetrievalResult:
        if not self.nodes:
            return RetrievalResult(question=question, context="", empty_graph=True)

        sims = self._vectors() @ np.asarray(self.embedder.transform([question]))[0]
        order = sorted(range(len(self.nodes)), key=lambda i: (-sims[i], self.nodes[i]))
        seeds = [self.nodes[i] for i in order[: self.k_seeds]]
        weights = {self.nodes[i]: max(float(sims[i]), 0.0) for i in order[: self.k_seeds]}
        if sum(weights.values()) == 0:                      # nothing similar at all: uniform restart
            weights = {s: 1.0 for s in seeds}

        scores = nx.pagerank(self.undirected, alpha=self.damping, personalization=weights, weight="weight")
        top = sorted(self.nodes, key=lambda n: (-scores[n], n))[: self.top_m]
        top_set = set(top)

        facts = self._facts(top_set, scores)
        chunk_order = self._chunk_order(top, scores)
        context, used_facts, used_chunks, n_words = self._assemble(facts, chunk_order)
        return RetrievalResult(question=question, context=context, seeds=seeds, top_entities=top,
                               facts=used_facts, chunk_ids=used_chunks, n_words=n_words)

    def _facts(self, top_set: set[str], scores: dict) -> list[str]:
        ranked = []
        for u, v, d in self.graph.edges(data=True):
            if u in top_set and v in top_set:
                line = f"{self.graph.nodes[u]['name']} -- {d['description']} -- {self.graph.nodes[v]['name']}"
                ranked.append((-(scores[u] + scores[v]), line))
        lines = []
        for _, line in sorted(set(ranked)):
            if line not in lines:
                lines.append(line)
        return lines[: self.max_facts]

    def _chunk_order(self, top: list[str], scores: dict) -> list[str]:
        mass: dict[str, float] = {}
        for n in top:
            for chunk_id in self.graph.nodes[n]["chunk_ids"]:
                mass[chunk_id] = mass.get(chunk_id, 0.0) + scores[n]
        return sorted((c for c in mass if c in self.chunk_text), key=lambda c: (-mass[c], c))

    def _assemble(self, facts: list[str], chunk_order: list[str]):
        budget = self.max_context_words
        used_facts, used_chunks, parts = [], [], []
        for line in facts:
            if _n_words(line) + 2 > budget:
                break
            used_facts.append(line)
            budget -= _n_words(line) + 1
        passages = []
        for chunk_id in chunk_order:
            text = self.chunk_text[chunk_id]
            if _n_words(text) + 1 <= budget:               # skip what does not fit, keep trying
                used_chunks.append(chunk_id)
                passages.append(f"[{len(passages) + 1}] {text}")
                budget -= _n_words(text) + 1
        if used_facts:
            parts.append("Facts:\n" + "\n".join(f"- {line}" for line in used_facts))
        if passages:
            parts.append("Passages:\n" + "\n".join(passages))
        context = "\n\n".join(parts)
        return context, used_facts, used_chunks, _n_words(context)


def retriever_from_config(cfg, graph: nx.MultiDiGraph, chunks, embedder=None) -> GraphRetriever:
    """The retriever for one condition, with the run's shared embedder and fixed settings."""
    from src.embeddings import embedder_from_config
    return GraphRetriever(
        graph, {c.chunk_id: c.text for c in chunks}, embedder or embedder_from_config(cfg),
        k_seeds=cfg.retrieval_k_seeds, damping=cfg.retrieval_damping, top_m=cfg.retrieval_top_m,
        max_context_words=cfg.retrieval_max_context_words, max_facts=cfg.retrieval_max_facts,
    )
