"""Phase 4: the real OpenAI extractor, tested with a FAKE client - zero real API calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from types import SimpleNamespace

import pytest

from extraction import MockExtractor, build_extractor
from extraction.base_extractor import ExtractionAbort
from extraction.openai_extractor import (DEFAULT_MAX_OUTPUT_TOKENS, EXTRACTION_SEED, DisabledClient,
                                         MissingAPIKeyError, OpenAIExtractor, classify_api_error)
from extraction.pricing import cost_usd
from extraction.prompts import PROMPT_VERSION, RESPONSE_FORMAT, build_messages
from extraction.schemas import Entity, ExtractionInput, RawExtraction, Relationship
from src.config import ExperimentConfig
from helpers_extraction import (GOOD_PAYLOAD, FakeOpenAIClient, good_handler, make_cfg, make_error,
                                make_response)

pytestmark = pytest.mark.usefixtures("no_network", "no_api_key")

ITEM = ExtractionInput(chunk_id="c1", text="Tom Hanks starred in Forrest Gump.")


def make(client, **kw):
    sleeps = []
    ex = OpenAIExtractor(client=client, sleep=sleeps.append, **kw)
    return ex, sleeps


# ------------------------------------------------------------------ API key
def test_missing_key_raises_clear_error_and_never_falls_back_to_mock():
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY") as exc:
        OpenAIExtractor()
    assert "NOT fall back" in str(exc.value)


def test_build_extractor_with_openai_backend_and_no_key_raises_instead_of_returning_a_mock():
    cfg = make_cfg(extraction_backend="openai")
    with pytest.raises(MissingAPIKeyError):
        build_extractor(cfg)


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(MissingAPIKeyError):
        OpenAIExtractor()


def test_mock_backend_is_the_default_and_builds_a_mock():
    assert isinstance(build_extractor(make_cfg()), MockExtractor)


def test_dry_run_extractor_needs_no_key_and_can_never_call_the_api():
    ex = build_extractor(make_cfg(extraction_backend="openai"), dry_run=True)
    assert isinstance(ex, OpenAIExtractor) and isinstance(ex.client, DisabledClient)
    assert ex.estimate_cost_usd(ITEM) > 0
    with pytest.raises(RuntimeError, match="disabled"):
        ex.extract(ITEM)


def test_api_key_never_appears_in_results_or_settings():
    secret = "sk-test-SECRET-123456"
    client = FakeOpenAIClient(handler=good_handler())
    ex = OpenAIExtractor(api_key=secret, client=client, sleep=lambda s: None)
    result = ex.extract(ITEM)
    assert secret not in result.model_dump_json()
    assert secret not in json.dumps(ex.cache_settings())
    assert secret not in json.dumps(client.calls, default=str)


# ------------------------------------------------------------- happy path
def test_good_response_is_parsed_with_usage_cost_model_and_attempts():
    client = FakeOpenAIClient([make_response(prompt_tokens=1000, completion_tokens=500)])
    ex, sleeps = make(client)
    r = ex.extract(ITEM)
    assert r.status == "ok" and r.error is None and r.attempts == 1
    assert [e.name for e in r.entities] == ["Alpha", "Beta"]
    assert [(x.source, x.target) for x in r.relationships] == [("Alpha", "Beta")]
    assert r.usage.input_tokens == 1000 and r.usage.output_tokens == 500
    assert r.usage.cost_usd == pytest.approx(1000 * 0.15 / 1e6 + 500 * 0.60 / 1e6)
    assert r.extractor == "openai" and r.model == "gpt-4o-mini" and r.prompt_version == PROMPT_VERSION
    assert r.runtime_seconds >= 0 and sleeps == []


def test_request_uses_structured_output_temperature_zero_and_only_the_chunk_text():
    client = FakeOpenAIClient([make_response()])
    make(client, max_output_tokens=777)[0].extract(ITEM)
    (call,) = client.calls
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["max_completion_tokens"] == 777
    assert call["seed"] == EXTRACTION_SEED
    assert call["response_format"] == RESPONSE_FORMAT and call["response_format"]["json_schema"]["strict"]
    assert call["messages"] == build_messages(ITEM.text)


def test_prices_come_from_settings_not_hardcoded():
    client = FakeOpenAIClient([make_response(prompt_tokens=1_000_000, completion_tokens=1_000_000)])
    ex, _ = make(client, price_input_per_1m=2.0, price_output_per_1m=8.0)
    assert ex.extract(ITEM).usage.cost_usd == pytest.approx(10.0)
    assert cost_usd(1_000_000, 0, 0.15, 0.6) == pytest.approx(0.15)


def test_cache_settings_include_model_prompt_schema_and_temperature():
    s = OpenAIExtractor(client=FakeOpenAIClient(), temperature=0.3, model="m-x").cache_settings()
    assert s["model"] == "m-x" and s["temperature"] == 0.3
    assert s["prompt_version"] == PROMPT_VERSION and "schema_version" in s


# ------------------------------------------------ validation (no retry needed)
def test_relationship_to_unknown_entity_is_dropped_counted_and_not_retried():
    payload = {"entities": GOOD_PAYLOAD["entities"],
               "relationships": GOOD_PAYLOAD["relationships"]
               + [{"source": "Alpha", "target": "Ghost", "description": "haunts"}]}
    client = FakeOpenAIClient([make_response(payload)])
    r = make(client)[0].extract(ITEM)
    assert r.status == "ok" and r.validation_issues == 1 and len(r.relationships) == 1
    assert "Ghost" in r.validation_notes[0]
    assert len(client.calls) == 1 and r.attempts == 1       # NOT retried


# ------------------------------------------------------- malformed answers
@pytest.mark.parametrize("bad,status", [
    (make_response(content="this is not json"), "malformed"),
    (make_response(content=""), "malformed"),
    (make_response(content=json.dumps({"entities": []})), "malformed"),            # missing field
    (make_response({"entities": [{"name": "X", "type": "ANIMAL", "description": ""}],
                    "relationships": []}), "malformed"),                           # bad entity type
    (make_response({**GOOD_PAYLOAD, "extra": 1}), "malformed"),                    # extra field
    (make_response(content=None, refusal="I can't help with that"), "refused"),
])
def test_bad_answers_are_retried_then_marked_failed_with_all_tokens_counted(bad, status):
    # (truncated replies are deliberately NOT in this list: they are never retried - see below)
    client = FakeOpenAIClient(handler=lambda kwargs: bad)
    ex, sleeps = make(client, max_retries=2)
    r = ex.extract(ITEM)
    assert r.status == status and r.entities == [] and r.error
    assert r.attempts == 3 and len(client.calls) == 3
    assert r.usage.input_tokens == 300 and r.usage.output_tokens == 150     # every attempt is paid for
    assert r.usage.cost_usd > 0
    assert sleeps == [1.0, 2.0]                                             # backoff, none after the last try


def test_recovers_when_a_later_attempt_is_good():
    client = FakeOpenAIClient([make_response(content="oops"), make_response()])
    r = make(client)[0].extract(ITEM)
    assert r.status == "ok" and r.attempts == 2
    assert r.usage.input_tokens == 200                                      # both attempts counted


def test_zero_retries_means_one_attempt():
    client = FakeOpenAIClient(handler=lambda k: make_response(content="oops"))
    r = make(client, max_retries=0)[0].extract(ITEM)
    assert r.status == "malformed" and r.attempts == 1 and len(client.calls) == 1


# -------------------------------------------------- temporary API problems
@pytest.mark.parametrize("err", [
    make_error("RateLimitError", 429),
    make_error("APITimeoutError"),
    make_error("APIConnectionError"),
    make_error("InternalServerError", 500),
    make_error("SomeGatewayError", 503),
])
def test_temporary_api_errors_back_off_and_retry_then_succeed(err):
    client = FakeOpenAIClient([err, err, make_response()])
    ex, sleeps = make(client, max_retries=3)
    r = ex.extract(ITEM)
    assert r.status == "ok" and r.attempts == 3
    assert sleeps == [1.0, 2.0]


def test_backoff_doubles_and_gives_up_as_api_error():
    err = make_error("RateLimitError", 429)
    client = FakeOpenAIClient(handler=lambda k: err)
    ex, sleeps = make(client, max_retries=3)
    r = ex.extract(ITEM)
    assert r.status == "api_error" and r.attempts == 4 and len(client.calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]
    assert r.usage.cost_usd == 0.0 and "RateLimitError" in r.error


# ------------------------------------------------------- run-level problems
@pytest.mark.parametrize("err", [
    make_error("AuthenticationError", 401),
    make_error("PermissionDeniedError", 403),
    make_error("NotFoundError", 404),
    make_error("BadRequestError", 400),
    make_error("RateLimitError", 429, code="insufficient_quota"),      # a 429 that is really "no money"
    make_error("WeirdError", None, code="invalid_api_key"),
    ValueError("some unexpected bug"),                                  # unknown errors are not retried
])
def test_fatal_errors_abort_the_whole_run_immediately_without_retrying(err):
    client = FakeOpenAIClient(handler=lambda k: err)
    ex, sleeps = make(client, max_retries=5)
    with pytest.raises(ExtractionAbort):
        ex.extract(ITEM)
    assert len(client.calls) == 1 and sleeps == []


def test_classification_rules():
    assert classify_api_error(make_error("RateLimitError", 429)) == "retry"
    assert classify_api_error(make_error("RateLimitError", 429, code="insufficient_quota")) == "fatal"
    assert classify_api_error(make_error("AuthenticationError", 401)) == "fatal"
    assert classify_api_error(TimeoutError()) == "retry"
    assert classify_api_error(KeyError("x")) == "fatal"


def test_real_openai_sdk_exception_classes_are_classified_correctly():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def sdk_error(cls, status, body=None):
        return cls("x", response=httpx.Response(status, request=request), body=body)

    assert classify_api_error(sdk_error(openai.RateLimitError, 429, {"error": {"code": "rate_limit_exceeded"}})) == "retry"
    assert classify_api_error(sdk_error(openai.RateLimitError, 429, {"error": {"code": "insufficient_quota"}})) == "fatal"
    assert classify_api_error(sdk_error(openai.AuthenticationError, 401)) == "fatal"
    assert classify_api_error(sdk_error(openai.NotFoundError, 404)) == "fatal"
    assert classify_api_error(sdk_error(openai.InternalServerError, 500)) == "retry"
    assert classify_api_error(openai.APITimeoutError(request=request)) == "retry"
    assert classify_api_error(openai.APIConnectionError(request=request)) == "retry"


# -------------------------------------------------------------- estimates
def test_estimate_is_positive_scales_with_text_and_prices():
    ex, _ = make(FakeOpenAIClient())
    short = ex.estimate_cost_usd(ExtractionInput(chunk_id="a", text="one two"))
    long = ex.estimate_cost_usd(ExtractionInput(chunk_id="b", text="word " * 400))
    assert 0 < short < long
    pricey, _ = make(FakeOpenAIClient(), price_input_per_1m=1.5, price_output_per_1m=6.0)
    assert pricey.estimate_cost_usd(ITEM) == pytest.approx(10 * ex.estimate_cost_usd(ITEM))


# ============================================================================
# Truncation: deterministic failure -> status "truncated", NEVER retried
# ============================================================================
def truncated_response(**kw):
    """A reply that was cut off at the output limit (partial JSON, finish_reason=length)."""
    return make_response(content='{"entities": [{"name": "Tom Han', finish_reason="length", **kw)


def test_truncated_reply_is_not_retried_and_is_clearly_marked():
    client = FakeOpenAIClient(handler=lambda kwargs: truncated_response(prompt_tokens=700, completion_tokens=999))
    ex, sleeps = make(client, max_retries=3, max_output_tokens=999)
    r = ex.extract(ITEM)
    assert r.status == "truncated"                       # its own status, not lumped with "malformed"
    assert len(client.calls) == 1 and r.attempts == 1    # ONE call, although 3 retries were allowed
    assert sleeps == []                                  # no backoff either
    assert "truncated" in r.error and "999" in r.error and "not retried" in r.error
    assert "extraction_max_output_tokens" in r.error     # tells the user the fix
    assert r.entities == [] and r.relationships == []
    assert r.usage.known and r.usage.input_tokens == 700 and r.usage.output_tokens == 999
    assert r.usage.cost_usd > 0                          # the single attempt is still paid for and counted


def test_truncation_is_decided_by_finish_reason_even_if_the_partial_text_happens_to_be_valid():
    # A reply that stopped because of the limit must never be accepted as a complete extraction.
    client = FakeOpenAIClient(handler=lambda kwargs: make_response(finish_reason="length"))   # valid JSON inside
    r = make(client, max_retries=3)[0].extract(ITEM)
    assert r.status == "truncated" and r.entities == [] and len(client.calls) == 1


def test_a_transient_error_is_still_retried_before_a_truncation_ends_the_chunk():
    client = FakeOpenAIClient([make_error("RateLimitError", 429), truncated_response()])
    ex, sleeps = make(client, max_retries=3)
    r = ex.extract(ITEM)
    assert r.status == "truncated" and r.attempts == 2 and len(client.calls) == 2
    assert sleeps == [1.0]                               # the transient failure WAS retried, once


def test_a_malformed_attempt_followed_by_truncation_stops_at_the_truncation():
    client = FakeOpenAIClient([make_response(content="oops"), truncated_response()])
    r = make(client, max_retries=3)[0].extract(ITEM)
    assert r.status == "truncated" and r.attempts == 2 and len(client.calls) == 2
    assert r.usage.input_tokens == 200                   # both attempts counted


def test_transient_errors_are_still_retried_when_nothing_is_truncated():
    # Guard: the truncation rule must not have weakened transient-error retries.
    err = make_error("APITimeoutError")
    client = FakeOpenAIClient([err, err, make_response()])
    r = make(client, max_retries=3)[0].extract(ITEM)
    assert r.status == "ok" and r.attempts == 3


def _conservative_tokens(n_each: int) -> int:
    """Token estimate for a reply with n entities + n relationships, at a CONSERVATIVE
    3 characters per token (real English is about 4+), using realistic word lengths."""
    words = ["film", "was", "released", "in", "the", "united", "states", "directed", "by", "famous", "author"]
    def text(k, seed): return " ".join(words[(seed + i) % len(words)] for i in range(k))
    ents = [Entity(name=text(3, i).title() + f" {i}", type="ORGANIZATION", description=text(20, i))
            for i in range(n_each)]
    rels = [Relationship(source=ents[i].name, target=ents[(i + 1) % n_each].name, description=text(20, i))
            for i in range(n_each)]
    return len(RawExtraction(entities=ents, relationships=rels).model_dump_json()) // 3


def test_default_output_cap_is_sized_from_the_schema_and_is_consistent_everywhere():
    # The old cap (1500) would truncate even a fairly ordinary dense chunk (15 entities + 15 relationships)...
    assert _conservative_tokens(15) > 1500
    # ...so the default must cover the intended ceiling (40 + 40) with a 1.5x safety margin.
    assert DEFAULT_MAX_OUTPUT_TOKENS >= 1.5 * _conservative_tokens(40)
    # It stays under gpt-4o-mini's 16,384-token output ceiling.
    assert DEFAULT_MAX_OUTPUT_TOKENS <= 16384
    # Extractor default, config default and the cache key all agree.
    cfg = ExperimentConfig(experiment_id="t", seed=1, dataset="hotpotqa", num_questions=5,
                           strategy="random", budget=0.1)
    assert cfg.extraction_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    ex = OpenAIExtractor(client=FakeOpenAIClient())
    assert ex.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert ex.cache_settings()["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_the_configured_output_cap_is_what_is_sent_to_the_api():
    client = FakeOpenAIClient([make_response()])
    make(client)[0].extract(ITEM)
    assert client.calls[0]["max_completion_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


# ============================================================================
# Missing usage information: UNKNOWN, never silently 0 tokens / $0
# ============================================================================
def without_usage(response):
    response.usage = None
    return response


def test_missing_usage_block_is_unknown_not_zero():
    client = FakeOpenAIClient([without_usage(make_response())])
    r = make(client)[0].extract(ITEM)
    assert r.status == "ok" and [e.name for e in r.entities] == ["Alpha", "Beta"]   # extraction itself is fine
    assert r.usage.known is False
    assert r.usage.input_tokens is None and r.usage.output_tokens is None and r.usage.cost_usd is None


def test_response_object_with_no_usage_attribute_at_all_is_unknown():
    response = make_response()
    del response.usage
    r = make(FakeOpenAIClient([response]))[0].extract(ITEM)
    assert r.usage.known is False and r.usage.cost_usd is None


@pytest.mark.parametrize("partial", [
    SimpleNamespace(prompt_tokens=100, completion_tokens=None),
    SimpleNamespace(prompt_tokens=None, completion_tokens=50),
    SimpleNamespace(prompt_tokens=100),                       # completion_tokens missing entirely
    SimpleNamespace(prompt_tokens="many", completion_tokens=50),   # not a number
])
def test_partial_or_garbled_usage_is_unknown_too(partial):
    response = make_response()
    response.usage = partial
    r = make(FakeOpenAIClient([response]))[0].extract(ITEM)
    assert r.usage.known is False
    assert r.usage.input_tokens is None and r.usage.output_tokens is None and r.usage.cost_usd is None


def test_a_reported_zero_is_a_real_value_not_unknown():
    r = make(FakeOpenAIClient([make_response(prompt_tokens=10, completion_tokens=0)]))[0].extract(ITEM)
    assert r.usage.known is True
    assert r.usage.input_tokens == 10 and r.usage.output_tokens == 0
    assert r.usage.cost_usd == pytest.approx(10 * 0.15 / 1e6)


def test_no_token_counts_are_invented_when_usage_is_missing():
    # The extractor has an estimator for dry runs; it must NOT leak into recorded usage.
    ex, _ = make(FakeOpenAIClient([without_usage(make_response())]))
    assert ex.estimate_cost_usd(ITEM) > 0                       # an estimate exists...
    r = ex.extract(ITEM)
    assert r.usage.cost_usd is None and r.usage.input_tokens is None   # ...but is never recorded as actual


def test_usage_reported_on_attempt_1_but_missing_on_attempt_2_makes_the_total_unknown():
    # A partial sum would understate the cost, so the whole total is unknown.
    client = FakeOpenAIClient([make_response(content="oops"), without_usage(make_response())])
    r = make(client)[0].extract(ITEM)
    assert r.status == "ok" and r.attempts == 2
    assert r.usage.known is False and r.usage.cost_usd is None and r.usage.input_tokens is None


def test_usage_missing_on_attempt_1_but_reported_on_attempt_2_makes_the_total_unknown():
    client = FakeOpenAIClient([without_usage(make_response(content="oops")), make_response()])
    r = make(client)[0].extract(ITEM)
    assert r.status == "ok" and r.attempts == 2
    assert r.usage.known is False and r.usage.cost_usd is None and r.usage.input_tokens is None


def test_a_failed_chunk_whose_responses_lacked_usage_is_unknown_too():
    client = FakeOpenAIClient(handler=lambda kw: without_usage(make_response(content="not json")))
    r = make(client, max_retries=1)[0].extract(ITEM)
    assert r.status == "malformed" and r.usage.known is False and r.usage.cost_usd is None


def test_a_truncated_reply_without_usage_is_unknown_too():
    client = FakeOpenAIClient([without_usage(truncated_response())])
    r = make(client)[0].extract(ITEM)
    assert r.status == "truncated" and r.usage.known is False


def test_when_no_response_ever_arrived_nothing_was_reported_so_usage_is_a_known_zero():
    # Documented distinction: a timeout / 429 returns NO response (nothing billable was reported),
    # which is different from a response that arrived WITHOUT its usage block.
    err = make_error("RateLimitError", 429)
    r = make(FakeOpenAIClient(handler=lambda kw: err), max_retries=1)[0].extract(ITEM)
    assert r.status == "api_error" and r.usage.known is True and r.usage.cost_usd == 0.0
