"""
One OpenAI chat call with strict JSON output - used by answer generation and the
native LazyGraphRAG relevance check (blueprint Phase 10).

It follows exactly the same rules as the extraction client
(extraction/openai_extractor.py), reusing its error classifier, dry-run client
and missing-key error:
  - temporary errors (rate limit, timeout, 5xx) -> retry with backoff
  - run-level errors (bad key, no quota, unknown model, bad request, anything
    unrecognised) -> raise immediately, the whole run stops
  - invalid JSON / schema violation / refusal -> retry, then give up for this item
  - truncated reply (finish_reason "length") -> NOT retried
  - tokens of every attempt are counted; a response without usage information
    makes the usage UNKNOWN (None), never 0 / $0
  - never falls back to a mock; a missing OPENAI_API_KEY stops with a clear error
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from extraction.base_extractor import ExtractionAbort
from extraction.openai_extractor import (MAX_BACKOFF_SECONDS, MissingAPIKeyError, OpenAIExtractor,
                                         classify_api_error)
from extraction.pricing import cost_usd
from extraction.schemas import Usage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixed request seed (not the experiment seed), like extraction: Random runs with
# different seeds must be able to share cached answers for identical prompts.
GENERATION_SEED = 1234


@dataclass
class CallOutcome:
    status: str                    # ok | api_error | malformed | refused | truncated
    parsed: BaseModel | None
    error: str | None
    usage: Usage
    attempts: int
    runtime_seconds: float


class ChatJSONClient:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, max_output_tokens: int = 512,
                 max_retries: int = 3, timeout_s: float = 60.0, price_input_per_1m: float = 0.15,
                 price_output_per_1m: float = 0.60, client=None, api_key: str | None = None,
                 sleep=time.sleep, backoff_base_s: float = 1.0, dotenv_path: str | Path | None = None):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.price_input_per_1m = price_input_per_1m
        self.price_output_per_1m = price_output_per_1m
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep
        self.client = client if client is not None else self._make_real_client(api_key, dotenv_path)

    def _make_real_client(self, api_key, dotenv_path):
        load_dotenv(dotenv_path or PROJECT_ROOT / ".env")        # never overrides real env vars
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key.strip():
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set, so the OpenAI answer generator cannot run. It will NOT "
                "fall back to the mock. Put the key in .env, or set generation_backend: mock."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ExtractionAbort("The openai package is not installed: pip install openai") from e
        return OpenAI(api_key=key, timeout=self.timeout_s, max_retries=0)

    def call(self, messages: list[dict], response_format: dict, output_model: type[BaseModel],
             label: str = "") -> CallOutcome:
        started = time.perf_counter()
        in_tok = out_tok = 0
        usage_known = True
        last_status, last_error, attempts = "api_error", "no attempt made", 0

        for attempt in range(1, self.max_retries + 2):           # 1 try + max_retries retries
            attempts = attempt
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                    max_completion_tokens=self.max_output_tokens, seed=GENERATION_SEED,
                    response_format=response_format,
                )
            except Exception as exc:                             # noqa: BLE001 - classified below
                if classify_api_error(exc) == "fatal":
                    raise ExtractionAbort(f"{type(exc).__name__} during {label or 'a query call'}: {exc}") from exc
                last_status, last_error = "api_error", f"{type(exc).__name__}: {exc}"
                self._backoff(attempt)
                continue

            reported = OpenAIExtractor._usage(response)          # same "unknown is not zero" rule
            if reported is None:
                usage_known = False
            else:
                in_tok += reported[0]
                out_tok += reported[1]

            status, parsed, error = self._parse(response, output_model)
            if status == "ok":
                return self._outcome("ok", parsed, None, started, attempts, in_tok, out_tok, usage_known)
            if status == "truncated":
                return self._outcome("truncated", None,
                                     f"reply truncated at max_output_tokens={self.max_output_tokens}; not retried",
                                     started, attempts, in_tok, out_tok, usage_known)
            last_status, last_error = status, error
            self._backoff(attempt)

        return self._outcome(last_status, None, f"gave up after {attempts} attempt(s): {last_error}",
                             started, attempts, in_tok, out_tok, usage_known)

    def _backoff(self, attempt: int) -> None:
        if attempt <= self.max_retries:
            self._sleep(min(self.backoff_base_s * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS))

    @staticmethod
    def _parse(response, output_model):
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError, TypeError):
            return "malformed", None, "response has no choices/message"
        if getattr(message, "refusal", None):
            return "refused", None, f"model refused: {message.refusal}"
        if getattr(choice, "finish_reason", None) == "length":
            return "truncated", None, "output was truncated"
        content = getattr(message, "content", None)
        if not content:
            return "malformed", None, "response content is empty"
        try:
            return "ok", output_model.model_validate(json.loads(content)), ""
        except json.JSONDecodeError as e:
            return "malformed", None, f"invalid JSON: {e}"
        except ValidationError as e:
            return "malformed", None, f"does not match the schema: {e.error_count()} error(s)"

    def _outcome(self, status, parsed, error, started, attempts, in_tok, out_tok, usage_known) -> CallOutcome:
        usage = (Usage(input_tokens=in_tok, output_tokens=out_tok,
                       cost_usd=cost_usd(in_tok, out_tok, self.price_input_per_1m, self.price_output_per_1m))
                 if usage_known else Usage.unknown())
        return CallOutcome(status, parsed, error, usage, attempts, time.perf_counter() - started)
