"""
Base strategy interface.

WHY THIS EXISTS:
This is the contract that makes the whole benchmark fair (blueprint §N).
Every strategy - Random, KET-RAG now, LazyGraphRAG/FastGraphRAG later -
must expose the exact same method signature, so the experiment runner
(Phase 13) can swap strategies without changing anything else in the
pipeline. A strategy's ONLY job is to RANK all chunks; it must NEVER see
the budget, the question set, or gold answers - that would be leakage
(blueprint §S) and would break the "strategy is the only variable" claim.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from src.chunking import Chunk


class SelectionStrategy(ABC):
    name: str  # e.g. "random", "ketrag" - set by each subclass

    @abstractmethod
    def rank(self, chunks: list[Chunk]) -> list[str]:
        """
        Given ALL chunks in the corpus, return ALL their chunk_ids ordered
        most-important-first. Must:
          - return every chunk_id exactly once (a full permutation, not a subset)
          - be deterministic given the strategy's own seed/parameters
          - NEVER look at budget, questions, or gold answers

        The budget cutoff (src/budget.py) is applied AFTER this, by the
        caller - not by the strategy itself.
        """
        raise NotImplementedError
