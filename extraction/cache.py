"""
Extraction cache (Phase 4).

WHY THIS EXISTS:
Extraction is the one step that costs real money. The same chunk is selected by
several strategies and at several budgets, and a run may crash half-way. The
cache means each (chunk text, settings) pair is sent to the LLM at most once:
re-running, resuming after a crash and comparing strategies all reuse earlier
answers for free.

THE KEY is a SHA-256 hash of:
    the extractor's cache_settings()  (extractor, model, prompt version, schema
                                       version, temperature, max output tokens)
  + the SHA-256 of the chunk text.
So changing the model, the prompt, the temperature... or the chunk text gives a
different key and old entries are simply not found. It is content-addressed on
purpose: chunk_id is NOT in the key, so the same text selected by two strategies
is one API call. The chunk_id is re-attached on every cache hit.

RULES
- Only successful ("ok") results are cached. A failed chunk (including a truncated
  one) is retried on the next run.
- A result whose token usage is UNKNOWN (the API sent no usage information) is not
  cached either. Cost is a benchmark axis; caching it would make that chunk's cost
  unknown in every future run. Not caching it means a re-run extracts it again and
  records a real cost.
- Files are written atomically (temp file, then rename), so a crash can never
  leave a half-written entry. A corrupt or unreadable entry counts as a miss.
- Layout: cache/extractions/<extractor>/<model>/<key>.json. Mock and OpenAI
  entries live in different folders and can never be mixed up.
- The chunk text itself is not stored, only its hash.
- cache/ is git-ignored: cached files hold real API response content and must
  never be committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from extraction.schemas import ExtractionInput, ExtractionResult


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(part)) or "_"


class ExtractionCache:
    def __init__(self, root: str | Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    # ------------------------------------------------------------------ keys
    @staticmethod
    def make_key(settings: dict, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return hashlib.sha256(
            canonical_json({"settings": settings, "text_sha256": text_hash}).encode("utf-8")
        ).hexdigest()

    def path_for(self, settings: dict, text: str) -> Path:
        key = self.make_key(settings, text)
        return self.root / _safe(settings["extractor"]) / _safe(settings["model"]) / f"{key}.json"

    # ------------------------------------------------------------- get / put
    def get(self, settings: dict, item: ExtractionInput) -> ExtractionResult | None:
        """Cached result for this text+settings (re-stamped with item.chunk_id), or None."""
        if not self.enabled:
            return None
        path = self.path_for(settings, item.text)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["key"] != self.make_key(settings, item.text):
                return None
            result = ExtractionResult.model_validate(data["result"])
        except (OSError, ValueError, KeyError, TypeError):
            return None          # missing, unreadable or corrupt -> treat as a miss
        if result.status != "ok" or not result.usage.known:
            return None          # failed, or cost-unknown entries are never served
        return result.model_copy(update={"chunk_id": item.chunk_id, "cache_hit": True})

    def put(self, settings: dict, item: ExtractionInput, result: ExtractionResult) -> bool:
        """Store a SUCCESSFUL result with KNOWN usage. Returns False (and writes nothing) otherwise."""
        if not self.enabled or result.status != "ok" or not result.usage.known:
            return False
        path = self.path_for(settings, item.text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": self.make_key(settings, item.text),
            "settings": settings,
            "text_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
            "result": result.model_copy(update={"cache_hit": False}).model_dump(mode="json"),
        }
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
