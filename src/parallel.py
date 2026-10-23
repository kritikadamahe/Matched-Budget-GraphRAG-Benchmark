"""Optional parallelism for LLM API calls (extraction, answering, judging).

Set LLM_WORKERS (an integer) in .env to run that many calls at once. Unset, 0 or 1 = the original
sequential behaviour, so nothing changes unless you ask for it. Keep LLM_MIN_INTERVAL_S unset on a
paid key (it serialises every call through one clock); use it only for rate-limited free tiers.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

SKIPPED = object()          # marker for an item that was never started because another item failed


def llm_workers() -> int:
    try:
        return max(1, int(os.environ.get("LLM_WORKERS", "1") or 1))
    except ValueError:
        return 1


def map_ordered(fn: Callable[[T], R], items: Sequence[T], workers: int | None = None) -> list:
    """fn applied to every item, results in INPUT order. With one worker this is a plain loop.

    If fn raises, no new items are started; items already running finish, and the exception of the
    earliest failing item (input order) is re-raised. Never returns SKIPPED entries."""
    workers = llm_workers() if workers is None else workers
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    results = map_ordered_collect(fn, items, workers)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return results


def map_ordered_collect(fn: Callable[[T], R], items: Sequence[T], workers: int) -> list:
    """Like map_ordered but never raises: each slot holds the result, the exception fn raised, or
    SKIPPED (not started because an earlier call failed). Input order."""
    stop = threading.Event()

    def run(item):
        if stop.is_set():
            return SKIPPED
        try:
            return fn(item)
        except BaseException as e:           # noqa: BLE001 - handed back to the caller in order
            stop.set()
            return e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, items))
