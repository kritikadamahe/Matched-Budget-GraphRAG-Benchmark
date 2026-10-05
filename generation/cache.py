"""
Cache for query-time LLM replies (blueprint Phase 10), same rules as the
extraction cache (extraction/cache.py):
- key = SHA-256 of (settings: task, backend, model, prompt version, temperature,
  max output tokens, seed) + (the exact inputs: question and context/passage);
- only successful results with KNOWN usage are stored;
- atomic writes; a corrupt entry is a miss;
- cache/answers/<backend>/<model>/ and cache/relevance/<backend>/<model>/, git-ignored.
At 100% budget every strategy produces the same context, so those answers are paid once.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from extraction.cache import _safe, canonical_json


class QueryCache:
    def __init__(self, root: str | Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    @staticmethod
    def make_key(settings: dict, inputs: dict) -> str:
        return hashlib.sha256(canonical_json({"settings": settings, "inputs": inputs}).encode("utf-8")).hexdigest()

    def _path(self, settings: dict, inputs: dict) -> Path:
        return (self.root / _safe(settings["backend"]) / _safe(settings["model"])
                / f"{self.make_key(settings, inputs)}.json")

    def get(self, settings: dict, inputs: dict) -> dict | None:
        if not self.enabled:
            return None
        try:
            data = json.loads(self._path(settings, inputs).read_text(encoding="utf-8"))
            if data["key"] != self.make_key(settings, inputs):
                return None
            return data["result"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, settings: dict, inputs: dict, result: dict, usage_known: bool) -> bool:
        if not self.enabled or result.get("status") != "ok" or not usage_known:
            return False
        path = self._path(settings, inputs)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"key": self.make_key(settings, inputs), "settings": settings,
                                   "result": result}, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
