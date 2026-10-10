"""
LLM-as-a-Judge (Phase 8, Evaluation).

Grades one generated answer against the reference answer. It follows the same rules as
the other query-time LLM modules and reuses their machinery (generation/llm_client.py
for the call, retries and token accounting; generation/cache.py for the cache):
  - temporary errors retried with backoff; run-level errors raise ExtractionAbort;
  - invalid JSON / schema violation / refusal retried, then given up on;
  - a truncated reply is NOT retried;
  - unknown token usage stays unknown (None), never $0;
  - never falls back to the mock; a missing OPENAI_API_KEY stops with an error.

A FAILED call gives correct=None. It is never turned into "incorrect": the record keeps
its status and is reported separately (unlike RelevanceChecker, where a failed check
conservatively counts as "not relevant").

The judge sees ONLY question, reference answer (+ aliases) and candidate answer
(evaluation/prompts.py). Its cost is query-time/evaluation cost, reported separately
from indexing and answer-generation cost.

Cache: cache/judge_calls/<backend>/<model>/<key>.json, key = SHA-256 of (task, backend,
model, prompt version, temperature, max output tokens, seed) + (question, reference,
aliases, candidate). Only successful results with known usage are stored.

Backends: "mock" is free, deterministic and offline (verdict = strict exact match) - for
tests only; every result is stamped backend="mock" and is never reportable.

BIAS WARNING: if judge_model equals the generation model, the judge may favour its own
style of answer (self-preference). The score summary records whether that is the case.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from evaluation.metrics import best_scores
from evaluation.prompts import (JUDGE_PROMPT_VERSION, JUDGE_RESPONSE_FORMAT, build_judge_messages)
from evaluation.schemas import JudgeOutput, JudgeResult
from extraction.pricing import cost_usd, estimate_tokens
from extraction.schemas import Usage
from generation.cache import QueryCache
from generation.llm_client import GENERATION_SEED, ChatJSONClient

ASSUMED_JUDGE_OUTPUT_TOKENS = 40       # one short sentence + a boolean (estimate only)
MESSAGE_FRAMING_TOKENS = 60


def _sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class Judge:
    task = "judge"
    prompt_version = JUDGE_PROMPT_VERSION

    def __init__(self, backend: str = "mock", client: ChatJSONClient | None = None,
                 cache: QueryCache | None = None, model: str = "gpt-4o-mini"):
        if backend not in ("mock", "openai"):
            raise ValueError(f"unknown judge backend: {backend!r}")
        if backend == "openai" and client is None:
            raise ValueError("the openai judge backend needs a ChatJSONClient")
        self.backend = backend
        self.client = client
        self.cache = cache
        self.model = client.model if client is not None else ("mock-v1" if backend == "mock" else model)
        self.results: list[JudgeResult] = []        # every result produced, for summaries

    def settings(self) -> dict:
        s = {"task": self.task, "backend": self.backend, "model": self.model,
             "prompt_version": self.prompt_version, "seed": GENERATION_SEED}
        if self.client is not None:
            s.update(temperature=self.client.temperature, max_output_tokens=self.client.max_output_tokens)
        return s

    def estimate_cost_usd(self, question: str, reference: str, aliases: list[str], candidate: str) -> float:
        if self.client is None:
            return 0.0
        messages = build_judge_messages(question, reference, aliases, candidate)
        tokens = sum(estimate_tokens(m["content"]) for m in messages) + MESSAGE_FRAMING_TOKENS
        return cost_usd(tokens, ASSUMED_JUDGE_OUTPUT_TOKENS,
                        self.client.price_input_per_1m, self.client.price_output_per_1m)

    def judge(self, question: str, reference: str, aliases: list[str], candidate: str) -> JudgeResult:
        messages = build_judge_messages(question, reference, aliases, candidate)
        base = dict(backend=self.backend, model=self.model, prompt_version=self.prompt_version,
                    prompt_sha256=_sha(messages))
        inputs = {"question": question, "reference": reference, "aliases": list(aliases), "candidate": candidate}

        cached = self.cache.get(self.settings(), inputs) if self.cache is not None else None
        if cached is not None:
            result = JudgeResult.model_validate({**cached, "cache_hit": True})
            self.results.append(result)
            return result

        started = time.perf_counter()
        if self.backend == "mock":
            correct = best_scores(candidate, [reference, *aliases])[0] == 1.0
            reply = json.dumps({"reasoning": "mock: exact match after normalisation", "correct": correct})
            result = JudgeResult(
                **base, status="ok", correct=correct, reasoning="mock: exact match after normalisation",
                usage=Usage(input_tokens=sum(estimate_tokens(m["content"]) for m in messages),
                            output_tokens=estimate_tokens(reply), cost_usd=0.0),
                runtime_seconds=time.perf_counter() - started)
        else:
            call = self.client.call(messages, JUDGE_RESPONSE_FORMAT, JudgeOutput, label="answer judging")
            ok = call.status == "ok"
            result = JudgeResult(**base, status=call.status, error=call.error, usage=call.usage,
                                 attempts=call.attempts, runtime_seconds=call.runtime_seconds,
                                 correct=bool(call.parsed.correct) if ok else None,
                                 reasoning=call.parsed.reasoning.strip() if ok else "")
        if self.cache is not None:
            self.cache.put(self.settings(), inputs, result.model_dump(mode="json"), result.usage.known)
        self.results.append(result)
        return result


def build_judge(cfg, dry_run: bool = False) -> Judge:
    """The judge named by the config's judge_* fields. With dry_run=True the real judge is
    built with a client that refuses to send anything (cost estimates need no key)."""
    from extraction.openai_extractor import DisabledClient
    from src.corpus import PROJECT_ROOT

    client = None
    if cfg.judge_backend == "openai":
        client = ChatJSONClient(
            model=cfg.judge_model, temperature=cfg.judge_temperature,
            max_output_tokens=cfg.judge_max_output_tokens, max_retries=cfg.judge_max_retries,
            timeout_s=cfg.judge_request_timeout_s, price_input_per_1m=cfg.judge_price_input_per_1m,
            price_output_per_1m=cfg.judge_price_output_per_1m,
            client=DisabledClient() if dry_run else None,
        )
    root = Path(cfg.cache_dir)
    root = root if root.is_absolute() else PROJECT_ROOT / root
    cache = QueryCache(root / "judge_calls", enabled=cfg.judge_cache_enabled)
    return Judge(cfg.judge_backend, client=client, cache=cache, model=cfg.judge_model)
