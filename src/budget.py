"""
Budget cutoff (blueprint §G).

A strategy's job is only to RANK all N chunks, most-important-first.
This module's job is to turn a budget fraction (e.g. 0.10) into an exact
count and slice the ranking - kept separate from every strategy so the
budget math is identical no matter which strategy produced the ranking.

Formula (from the blueprint): selected_count = round(B * N), minimum 1.
"""

from __future__ import annotations


def selected_count(n_total_chunks: int, budget: float) -> int:
    if n_total_chunks <= 0:
        raise ValueError("n_total_chunks must be positive")
    if not (0 < budget <= 1.0):
        raise ValueError("budget must be in (0, 1.0]")
    return max(1, round(budget * n_total_chunks))


def select_top_budget(ranked_chunk_ids: list[str], budget: float) -> list[str]:
    """
    ranked_chunk_ids: ALL chunk_ids, ordered most-important-first by a strategy.
    Returns the top `selected_count` of them - the ones that go to expensive
    LLM extraction. Order of the returned list is preserved (still ranked).
    """
    n = len(ranked_chunk_ids)
    k = selected_count(n, budget)
    return ranked_chunk_ids[:k]
