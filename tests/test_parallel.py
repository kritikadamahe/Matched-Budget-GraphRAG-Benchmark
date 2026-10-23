"""LLM_WORKERS parallelism: same results/order as sequential, safe failure handling."""
import threading
import time

import pytest

from extraction.base_extractor import ExtractionAbort
from src.parallel import llm_workers, map_ordered


def test_workers_env(monkeypatch):
    monkeypatch.delenv("LLM_WORKERS", raising=False)
    assert llm_workers() == 1
    monkeypatch.setenv("LLM_WORKERS", "8")
    assert llm_workers() == 8
    monkeypatch.setenv("LLM_WORKERS", "junk")
    assert llm_workers() == 1


def test_order_is_preserved():
    def slow_first(x):
        time.sleep(0.05 if x == 0 else 0)
        return x * 2
    assert map_ordered(slow_first, list(range(20)), workers=8) == [x * 2 for x in range(20)]


def test_actually_concurrent():
    seen, lock, peak = 0, threading.Lock(), [0]

    def fn(_):
        nonlocal seen
        with lock:
            seen += 1
            peak[0] = max(peak[0], seen)
        time.sleep(0.05)
        with lock:
            seen -= 1
    map_ordered(fn, list(range(16)), workers=4)
    assert 1 < peak[0] <= 4


def test_earliest_failure_is_raised_and_work_stops():
    started = []

    def fn(x):
        started.append(x)
        if x == 2:
            raise ExtractionAbort("no quota")
        time.sleep(0.01)
        return x
    with pytest.raises(ExtractionAbort):
        map_ordered(fn, list(range(200)), workers=4)
    assert len(started) < 200
