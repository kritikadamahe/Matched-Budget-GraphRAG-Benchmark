"""
Answer generation (blueprint Phase 10, §K) and the native LazyGraphRAG relevance
check (option L4). Both are QUERY-TIME LLM use: their cost is reported separately
from indexing (extraction) cost (blueprint §M).

Backends
- "mock": free, deterministic, offline - for tests and dry wiring runs only. Its
  answers are heuristics and must never appear in a reported result.
- "openai": GPT-4o-mini with strict JSON output (generation/llm_client.py). Never
  falls back to the mock.

Answering one question (Answerer.answer):
  empty context  -> no call, answer "not found", status "skipped_empty_context", $0
  cache hit      -> stored answer, re-used for free (cost still reported as original)
  otherwise      -> one call (with retries); answer tidied by clean_answer()
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from extraction.pricing import cost_usd, estimate_tokens
from extraction.schemas import Usage
from generation.cache import QueryCache
from generation.llm_client import GENERATION_SEED, ChatJSONClient
from generation.prompts import (ANSWER_PROMPT_VERSION, ANSWER_RESPONSE_FORMAT, NOT_FOUND,
                                RELEVANCE_PROMPT_VERSION, RELEVANCE_RESPONSE_FORMAT,
                                build_answer_messages, build_relevance_messages)
from generation.schemas import AnswerOutput, AnswerResult, RelevanceOutput, RelevanceResult, clean_answer
from src.text_utils import capitalized_phrases

ASSUMED_ANSWER_OUTPUT_TOKENS = 120      # short reasoning + answer (estimate only)
ASSUMED_RELEVANCE_OUTPUT_TOKENS = 10
MESSAGE_FRAMING_TOKENS = 60
RELEVANCE_MAX_OUTPUT_TOKENS = 64


def _sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _words(text: str) -> set[str]:
    return {w.strip(".,;:!?\"'()").lower() for w in text.split()} - {"", "the", "a", "an", "of", "in", "is",
                                                                       "was", "what", "which", "who", "did"}


# ----------------------------------------------------------------------------- mocks
def mock_answer(question: str, context: str) -> AnswerOutput:
    """Picks the first capitalised phrase in the context that is not in the question."""
    q = _words(question)
    for phrase in capitalized_phrases(context.replace("Facts:", " ").replace("Passages:", " ")):
        if not _words(phrase) & q:
            return AnswerOutput(reasoning="mock: first new capitalised phrase in the context", answer=phrase)
    return AnswerOutput(reasoning="mock: nothing suitable", answer=NOT_FOUND)


def mock_relevant(question: str, passage: str) -> bool:
    return len(_words(question) & _words(passage)) >= 2


# ---------------------------------------------------------------------------- shared
class _QueryLLM:
    task = ""
    prompt_version = ""
    assumed_output_tokens = 0

    def __init__(self, backend: str = "mock", client: ChatJSONClient | None = None,
                 cache: QueryCache | None = None, model: str = "gpt-4o-mini"):
        if backend not in ("mock", "openai"):
            raise ValueError(f"unknown generation backend: {backend!r}")
        if backend == "openai" and client is None:
            raise ValueError("the openai backend needs a ChatJSONClient")
        self.backend = backend
        self.client = client
        self.cache = cache
        self.model = client.model if client is not None else ("mock-v1" if backend == "mock" else model)
        self.results: list = []          # every result produced, for cost/status summaries

    def settings(self) -> dict:
        s = {"task": self.task, "backend": self.backend, "model": self.model,
             "prompt_version": self.prompt_version, "seed": GENERATION_SEED}
        if self.client is not None:
            s.update(temperature=self.client.temperature, max_output_tokens=self.client.max_output_tokens)
        return s

    def _estimate(self, messages: list[dict]) -> float:
        if self.client is None:
            return 0.0
        tokens = sum(estimate_tokens(m["content"]) for m in messages) + MESSAGE_FRAMING_TOKENS
        return cost_usd(tokens, self.assumed_output_tokens,
                        self.client.price_input_per_1m, self.client.price_output_per_1m)

    def _mock_usage(self, messages: list[dict], reply: str) -> Usage:
        # Mock: plausible token counts, $0 (it is free), so pipelines can be wired and checked.
        return Usage(input_tokens=sum(estimate_tokens(m["content"]) for m in messages),
                     output_tokens=estimate_tokens(reply), cost_usd=0.0)


# --------------------------------------------------------------------------- answerer
class Answerer(_QueryLLM):
    task = "answer"
    prompt_version = ANSWER_PROMPT_VERSION
    assumed_output_tokens = ASSUMED_ANSWER_OUTPUT_TOKENS

    def estimate_cost_usd(self, question: str, context: str) -> float:
        return 0.0 if not context.strip() else self._estimate(build_answer_messages(question, context))

    def answer(self, question: str, context: str) -> AnswerResult:
        messages = build_answer_messages(question, context)
        base = dict(question=question, backend=self.backend, model=self.model,
                    prompt_version=self.prompt_version, prompt_sha256=_sha(messages),
                    context_words=len(context.split()))
        if not context.strip():
            result = AnswerResult(**base, status="skipped_empty_context", answer=NOT_FOUND, attempts=0,
                                  reasoning="no context was retrieved, so no call was made")
            self.results.append(result)
            return result

        inputs = {"question": question, "context": context}
        cached = self.cache.get(self.settings(), inputs) if self.cache is not None else None
        if cached is not None:
            result = AnswerResult.model_validate({**cached, "cache_hit": True})
            self.results.append(result)
            return result

        started = time.perf_counter()
        if self.backend == "mock":
            out = mock_answer(question, context)
            result = AnswerResult(**base, status="ok", answer=clean_answer(out.answer), reasoning=out.reasoning,
                                  usage=self._mock_usage(messages, out.model_dump_json()),
                                  runtime_seconds=time.perf_counter() - started)
        else:
            call = self.client.call(messages, ANSWER_RESPONSE_FORMAT, AnswerOutput, label="answer generation")
            ok = call.status == "ok"
            result = AnswerResult(**base, status=call.status, error=call.error, usage=call.usage,
                                  attempts=call.attempts, runtime_seconds=call.runtime_seconds,
                                  answer=clean_answer(call.parsed.answer) if ok else "",
                                  reasoning=call.parsed.reasoning.strip() if ok else "")
        if self.cache is not None:
            self.cache.put(self.settings(), inputs, result.model_dump(mode="json"), result.usage.known)
        self.results.append(result)
        return result


# ------------------------------------------------------------------ relevance checker
class RelevanceChecker(_QueryLLM):
    """Yes/no relevance test for the native LazyGraphRAG (L4). Callable as
    relevance_fn(question, passage) -> bool, as NativeLazyGraphRAG.retrieve expects.
    A failed check counts as "not relevant" and stays visible in .results."""
    task = "relevance"
    prompt_version = RELEVANCE_PROMPT_VERSION
    assumed_output_tokens = ASSUMED_RELEVANCE_OUTPUT_TOKENS

    def estimate_cost_usd(self, question: str, passage: str) -> float:
        return self._estimate(build_relevance_messages(question, passage))

    def check(self, question: str, passage: str) -> RelevanceResult:
        messages = build_relevance_messages(question, passage)
        base = dict(question=question, backend=self.backend, model=self.model,
                    prompt_version=self.prompt_version, prompt_sha256=_sha(messages))
        inputs = {"question": question, "passage": passage}
        cached = self.cache.get(self.settings(), inputs) if self.cache is not None else None
        if cached is not None:
            result = RelevanceResult.model_validate({**cached, "cache_hit": True})
            self.results.append(result)
            return result

        started = time.perf_counter()
        if self.backend == "mock":
            relevant = mock_relevant(question, passage)
            result = RelevanceResult(**base, status="ok", relevant=relevant,
                                     usage=self._mock_usage(messages, json.dumps({"relevant": relevant})),
                                     runtime_seconds=time.perf_counter() - started)
        else:
            call = self.client.call(messages, RELEVANCE_RESPONSE_FORMAT, RelevanceOutput, label="relevance check")
            result = RelevanceResult(**base, status=call.status, error=call.error, usage=call.usage,
                                     attempts=call.attempts, runtime_seconds=call.runtime_seconds,
                                     relevant=bool(call.parsed.relevant) if call.status == "ok" else False)
        if self.cache is not None:
            self.cache.put(self.settings(), inputs, result.model_dump(mode="json"), result.usage.known)
        self.results.append(result)
        return result

    def __call__(self, question: str, passage: str) -> bool:
        return self.check(question, passage).relevant


# ----------------------------------------------------------------------------- summary
def summarize_query_usage(results: list) -> dict:
    """Totals for a list of Answer/RelevanceResults. Unknown is never free: if any
    usage is unknown, the totals are None and the known part is a lower bound."""
    fresh = [r for r in results if not r.cache_hit]
    unknown = [r for r in results if not r.usage.known]
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1

    def tot(rs, field):
        return sum(getattr(r.usage, field) for r in rs if r.usage.known)

    return {
        "n": len(results),
        "status_counts": statuses,
        "n_cache_hits": len(results) - len(fresh),
        "n_calls": sum(1 for r in fresh if r.status != "skipped_empty_context"),
        "usage_complete": not unknown,
        "input_tokens": None if unknown else tot(results, "input_tokens"),
        "output_tokens": None if unknown else tot(results, "output_tokens"),
        "cost_if_uncached_usd": None if unknown else round(tot(results, "cost_usd"), 6),
        "cost_spent_usd": None if any(not r.usage.known for r in fresh) else round(tot(fresh, "cost_usd"), 6),
        "known_cost_lower_bound_usd": round(tot(results, "cost_usd"), 6) if unknown else None,
    }


# ----------------------------------------------------------------------------- builders
def _client_from_config(cfg, max_output_tokens: int, dry_run: bool):
    from extraction.openai_extractor import DisabledClient
    return ChatJSONClient(
        model=cfg.generation_model, temperature=cfg.generation_temperature,
        max_output_tokens=max_output_tokens, max_retries=cfg.generation_max_retries,
        timeout_s=cfg.generation_request_timeout_s,
        price_input_per_1m=cfg.generation_price_input_per_1m,
        price_output_per_1m=cfg.generation_price_output_per_1m,
        client=DisabledClient() if dry_run else None,
    )


def _cache_root(cfg) -> Path:
    from src.corpus import PROJECT_ROOT
    root = Path(cfg.cache_dir)
    return root if root.is_absolute() else PROJECT_ROOT / root


def build_answerer(cfg, dry_run: bool = False) -> Answerer:
    client = (_client_from_config(cfg, cfg.generation_max_output_tokens, dry_run)
              if cfg.generation_backend == "openai" else None)
    cache = QueryCache(_cache_root(cfg) / "answers", enabled=cfg.generation_cache_enabled)
    return Answerer(cfg.generation_backend, client=client, cache=cache)


def build_relevance_checker(cfg, dry_run: bool = False) -> RelevanceChecker:
    client = (_client_from_config(cfg, RELEVANCE_MAX_OUTPUT_TOKENS, dry_run)
              if cfg.generation_backend == "openai" else None)
    cache = QueryCache(_cache_root(cfg) / "relevance", enabled=cfg.generation_cache_enabled)
    return RelevanceChecker(cfg.generation_backend, client=client, cache=cache)
