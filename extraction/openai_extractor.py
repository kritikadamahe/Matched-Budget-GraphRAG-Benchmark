"""
OpenAI extractor (Phase 4) - the real GPT-4o-mini entity/relationship extractor.

CATEGORY B - ADAPTATION: GraphRAG-style extraction with our own prompt and
simplified schema (see extraction/prompts.py).

SAFETY RULES
- It NEVER silently falls back to the mock. If OPENAI_API_KEY is missing, it
  raises MissingAPIKeyError with instructions. (The key is read from the
  environment, or from a .env file in the project root.)
- The API key is never stored in results, caches or logs.
- Tests inject a fake client, so they need no key and make no network call.

HOW ONE CHUNK IS HANDLED
  send request (structured JSON output, temperature 0)
    -> response parsed and validated against the schema
    -> relationships naming unknown entities are dropped + counted (no retry)

WHAT HAPPENS WHEN THINGS GO WRONG
  Temporary API problems (rate limit, timeout, connection error, 5xx)
      -> retry with exponential backoff (1s, 2s, 4s ...), up to max_retries
      -> still failing: that chunk gets status "api_error", the run continues
  Run-level problems (bad key, no quota, unknown model, bad request, or any
  error type we do not recognise)
      -> raise ExtractionAbort immediately: it would hit every chunk the same
         way, so continuing only wastes time
  Bad answers (invalid JSON, schema violation, refusal)
      -> retry up to max_retries; still bad: status "malformed"/"refused", the
         run continues
  Truncated output (the reply hit max_output_tokens, finish_reason "length")
      -> status "truncated" at once, NO retry. The same input is cut off again
         at the same limit, so retrying would only pay for the same failure
         several times. The fix is a larger extraction_max_output_tokens. Transient
         API errors are still retried as above; truncation just ends that chunk.
  Tokens from EVERY attempt, including failed ones, are counted in the cost.
  If an API response carries no usage information, the chunk's usage is recorded
  as UNKNOWN (None), never as 0 tokens / $0, and no token counts are invented.

REPRODUCIBILITY: temperature is 0 and a fixed seed is sent, but OpenAI only
promises best-effort determinism. The extraction CACHE is what guarantees an
identical re-run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from extraction.base_extractor import ExtractionAbort, Extractor
from extraction.pricing import ASSUMED_OUTPUT_TOKENS_PER_CHUNK, cost_usd, estimate_tokens
from extraction.prompts import PROMPT_VERSION, RESPONSE_FORMAT, SYSTEM_PROMPT, build_messages
from extraction.schemas import (SCHEMA_VERSION, ExtractionInput, ExtractionResult, RawExtraction,
                                Usage, clean_extraction)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Sent with every request. NOT the experiment seed: extraction must not depend on
# it, otherwise Random runs with different seeds could not share cached extractions.
EXTRACTION_SEED = 1234

# Rough size of the schema + message framing that is sent along with the system prompt.
PROMPT_FRAMING_TOKENS = 150

# Cap on the size of ONE extraction reply. Sized from the schema, not guessed: a dense
# chunk with 15 entities + 15 relationships already needs ~1,900 tokens, and 40 + 40
# needs ~5,100 (conservative 3 characters per token), so the old 1,500 would truncate
# ordinary dense chunks. 8,192 is half of gpt-4o-mini's 16,384 output ceiling. Headroom
# costs nothing: you pay only for tokens actually generated. Keep in sync with the
# extraction_max_output_tokens default in src/config.py (a test checks they match).
DEFAULT_MAX_OUTPUT_TOKENS = 8192

MAX_BACKOFF_SECONDS = 30.0

_FATAL_CODES = {"insufficient_quota", "invalid_api_key", "model_not_found", "billing_not_active"}
_FATAL_NAMES = {"AuthenticationError", "PermissionDeniedError", "NotFoundError",
                "BadRequestError", "UnprocessableEntityError"}
_FATAL_STATUS = {400, 401, 403, 404, 422}
_RETRY_NAMES = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
                "TimeoutError", "ConnectionError"}


class MissingAPIKeyError(ExtractionAbort):
    """OPENAI_API_KEY is not set."""


def classify_api_error(exc: BaseException) -> str:
    """Return "retry" (temporary) or "fatal" (stop the whole run).

    Uses the error's class name, HTTP status and error code instead of importing
    the openai package, so it also works for the fake errors used in tests.
    Anything unrecognised is "fatal": surfacing an unexpected error is safer than
    silently retrying it for every chunk.
    """
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if code is None and isinstance(body, dict):
        err = body.get("error")
        code = err.get("code") if isinstance(err, dict) else body.get("code")
    if code in _FATAL_CODES:
        return "fatal"                      # e.g. a 429 that really means "out of quota"
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name in _FATAL_NAMES or status in _FATAL_STATUS:
        return "fatal"
    if name in _RETRY_NAMES or status == 429 or (isinstance(status, int) and status >= 500):
        return "retry"
    return "fatal"


class _DisabledCompletions:
    def create(self, **kwargs):
        raise RuntimeError("API calls are disabled (dry run): this client never sends requests")


class DisabledClient:
    """Stand-in client for --dry-run: any attempt to call the API raises."""
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _DisabledCompletions()})()


class OpenAIExtractor(Extractor):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_retries: int = 3,
        timeout_s: float = 60.0,
        price_input_per_1m: float = 0.15,
        price_output_per_1m: float = 0.60,
        api_key: str | None = None,
        client=None,
        sleep=time.sleep,
        backoff_base_s: float = 1.0,
        dotenv_path: str | Path | None = None,
    ):
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

    # ---------------------------------------------------------------- client
    def _make_real_client(self, api_key: str | None, dotenv_path: str | Path | None):
        load_dotenv(dotenv_path or PROJECT_ROOT / ".env")   # never overrides real env vars
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key.strip():
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set, so the OpenAI extractor cannot run. It will NOT fall "
                "back to the mock extractor. Copy .env.example to .env and put your key in it, or "
                "export OPENAI_API_KEY. To test the pipeline for free, set extraction_backend: mock."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ExtractionAbort("The openai package is not installed: pip install openai") from e
        # max_retries=0: we do our own retries so they are counted and testable.
        return OpenAI(api_key=key, timeout=self.timeout_s, max_retries=0)

    # ------------------------------------------------------- Extractor API
    def cache_settings(self) -> dict:
        return {"extractor": self.name, "model": self.model, "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION, "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens, "seed": EXTRACTION_SEED}

    def estimate_cost_usd(self, item: ExtractionInput) -> float:
        input_tokens = estimate_tokens(item.text) + estimate_tokens(SYSTEM_PROMPT) + PROMPT_FRAMING_TOKENS
        return cost_usd(input_tokens, ASSUMED_OUTPUT_TOKENS_PER_CHUNK,
                        self.price_input_per_1m, self.price_output_per_1m)

    def extract(self, item: ExtractionInput) -> ExtractionResult:
        started = time.perf_counter()
        input_tokens = output_tokens = 0
        usage_known = True            # becomes False if ANY response arrives without usage info
        last_status, last_error = "api_error", "no attempt made"
        attempts = 0

        for attempt in range(1, self.max_retries + 2):          # 1 try + max_retries retries
            attempts = attempt
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=build_messages(item.text),
                    temperature=self.temperature,
                    max_completion_tokens=self.max_output_tokens,
                    seed=EXTRACTION_SEED,
                    response_format=RESPONSE_FORMAT,
                )
            except Exception as exc:                            # noqa: BLE001 - classified below
                if classify_api_error(exc) == "fatal":
                    raise ExtractionAbort(
                        f"{type(exc).__name__} while extracting chunk {item.chunk_id}: {exc}"
                    ) from exc
                last_status, last_error = "api_error", f"{type(exc).__name__}: {exc}"
                self._backoff(attempt)
                continue

            reported = self._usage(response)
            if reported is None:
                usage_known = False   # cannot add up a missing number; the total is now unknown
            else:
                input_tokens += reported[0]
                output_tokens += reported[1]

            status, raw, error = self._parse(response)
            if status == "ok":
                entities, relationships, issues, notes = clean_extraction(raw.entities, raw.relationships)
                return self._result(
                    item, "ok", started, attempts, input_tokens, output_tokens, usage_known,
                    entities=entities, relationships=relationships,
                    validation_issues=issues, validation_notes=notes,
                )
            if status == "truncated":
                # Deterministic failure: do NOT retry (it would only repeat the same cost).
                return self._result(
                    item, "truncated", started, attempts, input_tokens, output_tokens, usage_known,
                    error=(f"output truncated at max_output_tokens={self.max_output_tokens} "
                           f"(finish_reason=length); not retried because the same input would be cut "
                           f"off again - raise extraction_max_output_tokens"),
                )
            last_status, last_error = status, error
            self._backoff(attempt)

        return self._result(item, last_status, started, attempts, input_tokens, output_tokens, usage_known,
                            error=f"gave up after {attempts} attempt(s): {last_error}")

    # --------------------------------------------------------------- helpers
    def _backoff(self, attempt: int) -> None:
        """Wait before the next attempt; does nothing after the final attempt."""
        if attempt <= self.max_retries:
            self._sleep(min(self.backoff_base_s * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS))

    @staticmethod
    def _usage(response) -> tuple[int, int] | None:
        """(input_tokens, output_tokens) as REPORTED by the API, or None if the response
        has no usage information (missing block, or either number missing). A reported
        0 is a real value and stays 0; only absence means unknown."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is None or completion is None:
            return None
        try:
            return int(prompt), int(completion)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse(response) -> tuple[str, RawExtraction | None, str]:
        """Turn one API response into (status, parsed data or None, error text)."""
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError, TypeError):
            return "malformed", None, "response has no choices/message"
        refusal = getattr(message, "refusal", None)
        if refusal:
            return "refused", None, f"model refused: {refusal}"
        if getattr(choice, "finish_reason", None) == "length":
            return "truncated", None, "output was truncated (hit max_output_tokens)"
        content = getattr(message, "content", None)
        if not content:
            return "malformed", None, "response content is empty"
        try:
            return "ok", RawExtraction.model_validate(json.loads(content)), ""
        except json.JSONDecodeError as e:
            return "malformed", None, f"invalid JSON: {e}"
        except ValidationError as e:
            return "malformed", None, f"does not match the schema: {e.error_count()} error(s)"

    def _result(self, item, status, started, attempts, input_tokens, output_tokens, usage_known, *,
                entities=None, relationships=None, validation_issues=0,
                validation_notes=None, error=None) -> ExtractionResult:
        if usage_known:
            usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens,
                          cost_usd=cost_usd(input_tokens, output_tokens,
                                            self.price_input_per_1m, self.price_output_per_1m))
        else:
            usage = Usage.unknown()   # never 0 / $0: we simply do not know what was spent
        return ExtractionResult(
            chunk_id=item.chunk_id, status=status,
            entities=entities or [], relationships=relationships or [],
            validation_issues=validation_issues, validation_notes=validation_notes or [],
            error=error, extractor=self.name, model=self.model, prompt_version=PROMPT_VERSION,
            usage=usage,
            runtime_seconds=time.perf_counter() - started,
            attempts=attempts,
        )
