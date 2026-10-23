"""Optional global pacing for LLM API calls (for providers with low rate limits, e.g. free tiers).

Set LLM_MIN_INTERVAL_S (seconds) in .env to space every call at least that far apart. Unset or 0 =
no pacing, so paid runs are unaffected. One shared clock covers extraction, answering and judging.
"""
from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_last_call = 0.0


def wait_for_slot() -> None:
    global _last_call
    try:
        interval = float(os.environ.get("LLM_MIN_INTERVAL_S", "0") or 0)
    except ValueError:
        interval = 0.0
    if interval <= 0:
        return
    with _lock:
        delay = _last_call + interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _last_call = time.monotonic()
