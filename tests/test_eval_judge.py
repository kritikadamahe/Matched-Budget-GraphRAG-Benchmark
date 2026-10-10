"""Phase 8 Evaluation: the LLM judge. The OpenAI path is tested with a fake client:
no key, no network, no cost."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect

import pytest

from evaluation.judge import Judge, build_judge
from evaluation.prompts import JUDGE_PROMPT_VERSION, build_judge_messages
from extraction.base_extractor import ExtractionAbort
from extraction.openai_extractor import MissingAPIKeyError
from generation.cache import QueryCache
from generation.llm_client import ChatJSONClient
from src.config import ExperimentConfig
from tests.helpers_evaluation import Q, judge_reply, openai_judge, unknown_usage
from tests.helpers_extraction import make_error, make_response

REF, CAND = "Chicago", "Chicago, Illinois"


# --------------------------------------------------------------------- the prompt
def test_judge_prompt_contains_only_question_reference_aliases_and_candidate():
    system, user = build_judge_messages(Q, "USA", ["US", "United States"], "America")
    assert user["content"] == (f"Question: {Q}\nReference answer: USA\n"
                               "Other acceptable answers: US; United States\nCandidate answer: America")
    assert "context" not in system["content"].lower() and "passage" not in system["content"].lower()
    # no way to pass retrieval context, chunks, supporting facts or graph data into the judge
    assert list(inspect.signature(build_judge_messages).parameters) == ["question", "reference", "aliases", "candidate"]
    assert list(inspect.signature(Judge.judge).parameters)[1:] == ["question", "reference", "aliases", "candidate"]


def test_prompt_omits_alias_line_when_there_are_none_or_only_the_reference():
    assert "Other acceptable" not in build_judge_messages(Q, "USA", [], "x")[1]["content"]
    assert "Other acceptable" not in build_judge_messages(Q, "USA", ["USA", " "], "x")[1]["content"]


# --------------------------------------------------------------------- mock backend
def test_mock_judge_is_free_deterministic_and_stamped_mock():
    judge = Judge("mock")
    yes = judge.judge(Q, "The Beatles", [], "beatles.")
    no = judge.judge(Q, "The Beatles", [], "The Stones")
    assert (yes.correct, no.correct) == (True, False)
    assert yes.backend == "mock" and yes.model == "mock-v1" and yes.status == "ok"
    assert yes.usage.known and yes.usage.cost_usd == 0.0
    assert judge.judge(Q, "USA", ["US"], "U.S.").correct is True          # aliases count
    assert judge.estimate_cost_usd(Q, "a", [], "b") == 0.0
    assert yes == Judge("mock").judge(Q, "The Beatles", [], "beatles.").model_copy(update={"runtime_seconds": yes.runtime_seconds})


def test_unknown_backend_and_missing_client_are_rejected():
    with pytest.raises(ValueError):
        Judge("anthropic")
    with pytest.raises(ValueError):
        Judge("openai")


# --------------------------------------------------------------------- openai path
def test_openai_judge_ok_parses_verdict_and_counts_usage(no_network):
    judge, fake = openai_judge(script=[judge_reply(correct=True, reasoning="Same city.")])
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "ok" and r.correct is True and r.reasoning == "Same city."
    assert r.backend == "openai" and r.prompt_version == JUDGE_PROMPT_VERSION
    assert r.usage.input_tokens == 100 and r.usage.output_tokens == 50 and r.usage.cost_usd > 0
    sent = fake.calls[0]
    assert sent["messages"] == build_judge_messages(Q, REF, [], CAND)
    assert sent["response_format"]["json_schema"]["strict"] is True and sent["temperature"] == 0.0


def test_incorrect_verdict_is_a_real_false_not_none(no_network):
    judge, _ = openai_judge(script=[judge_reply(correct=False, reasoning="Different city.")])
    assert judge.judge(Q, REF, [], "Boston").correct is False


# --------------------------------------------------------------------- caching
def test_verdicts_are_cached_and_reused_without_an_api_call(tmp_path, no_network):
    cache = QueryCache(tmp_path / "judge_calls")
    judge, fake = openai_judge(script=[judge_reply(correct=True)], cache=cache)
    first = judge.judge(Q, REF, ["Chi"], CAND)
    again = judge.judge(Q, REF, ["Chi"], CAND)
    assert len(fake.calls) == 1
    assert first.cache_hit is False and again.cache_hit is True
    assert again.correct is True and again.usage == first.usage        # original cost stays visible

    judge2, fake2 = openai_judge(script=[], cache=cache)               # a fresh process, same cache dir
    assert judge2.judge(Q, REF, ["Chi"], CAND).cache_hit is True and fake2.calls == []


def test_cache_key_covers_candidate_reference_aliases_model_and_prompt_version(tmp_path, no_network):
    cache = QueryCache(tmp_path)
    judge, fake = openai_judge(handler=lambda kw: judge_reply(), cache=cache)
    judge.judge(Q, REF, [], CAND)
    judge.judge(Q, REF, [], "Boston")                  # different candidate
    judge.judge(Q, "Paris", [], CAND)                  # different reference
    judge.judge(Q, REF, ["Chi"], CAND)                 # different aliases
    assert len(fake.calls) == 4
    other_model, fake2 = openai_judge(handler=lambda kw: judge_reply(), cache=cache, model="gpt-4o")
    other_model.judge(Q, REF, [], CAND)                # different judge model
    assert len(fake2.calls) == 1
    assert judge.settings()["prompt_version"] == JUDGE_PROMPT_VERSION


def test_cache_can_be_disabled(tmp_path, no_network):
    judge, fake = openai_judge(handler=lambda kw: judge_reply(), cache=QueryCache(tmp_path, enabled=False))
    judge.judge(Q, REF, [], CAND)
    judge.judge(Q, REF, [], CAND)
    assert len(fake.calls) == 2


def test_failed_calls_are_not_cached(tmp_path, no_network):
    judge, fake = openai_judge(script=[make_response(content="nope")] * 2 + [judge_reply()],
                               cache=QueryCache(tmp_path), max_retries=1)
    assert judge.judge(Q, REF, [], CAND).status == "malformed"
    assert judge.judge(Q, REF, [], CAND).status == "ok"              # re-asked, not served from cache
    assert len(fake.calls) == 3


# --------------------------------------------------------------------- malformed output
def test_malformed_output_is_retried_then_gives_no_verdict(no_network):
    judge, fake = openai_judge(script=[make_response(content="not json")] * 3, max_retries=2)
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "malformed" and r.correct is None and "gave up after 3" in r.error
    assert len(fake.calls) == 3 and r.attempts == 3
    assert r.usage.input_tokens == 300                               # every attempt is counted


def test_schema_violation_counts_as_malformed(no_network):
    bad = make_response(payload={"reasoning": "x", "correct": "maybe"})
    extra = make_response(payload={"reasoning": "x", "correct": True, "confidence": 0.9})
    judge, _ = openai_judge(script=[bad, extra], max_retries=1)
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "malformed" and r.correct is None


def test_malformed_then_valid_recovers(no_network):
    judge, fake = openai_judge(script=[make_response(content="{"), judge_reply(correct=False)])
    r = judge.judge(Q, REF, [], "Boston")
    assert r.status == "ok" and r.correct is False and r.attempts == 2 and r.usage.input_tokens == 200


def test_refusal_is_retried_then_reported(no_network):
    judge, _ = openai_judge(script=[make_response(content="", refusal="I can't help")] * 2, max_retries=1)
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "refused" and r.correct is None


# --------------------------------------------------------------------- API errors / retries
def test_temporary_errors_are_retried_then_succeed(no_network):
    judge, fake = openai_judge(script=[make_error("RateLimitError", status=429),
                                       make_error("APITimeoutError"), judge_reply()])
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "ok" and r.attempts == 3 and len(fake.calls) == 3


def test_temporary_errors_that_never_clear_give_api_error_and_no_verdict(no_network):
    judge, _ = openai_judge(script=[make_error("InternalServerError", status=500)] * 3, max_retries=2)
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "api_error" and r.correct is None and "InternalServerError" in r.error


@pytest.mark.parametrize("err", [make_error("AuthenticationError", status=401),
                                 make_error("RateLimitError", status=429, code="insufficient_quota"),
                                 make_error("NotFoundError", status=404)])
def test_run_level_errors_stop_everything(err, no_network):
    judge, fake = openai_judge(script=[err])
    with pytest.raises(ExtractionAbort):
        judge.judge(Q, REF, [], CAND)
    assert len(fake.calls) == 1 and judge.results == []


# --------------------------------------------------------------------- truncation
def test_truncated_reply_is_not_retried(no_network):
    judge, fake = openai_judge(script=[make_response(content='{"reasoning": "cut', finish_reason="length")])
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "truncated" and r.correct is None and len(fake.calls) == 1
    assert "not retried" in r.error and r.usage.known and r.usage.input_tokens == 100   # still paid for


# --------------------------------------------------------------------- unknown usage / cost
def test_unknown_usage_stays_unknown_and_is_not_cached(tmp_path, no_network):
    cache = QueryCache(tmp_path)
    judge, fake = openai_judge(script=[unknown_usage(judge_reply()), judge_reply()], cache=cache)
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "ok" and r.correct is True
    assert r.usage.known is False and r.usage.cost_usd is None and r.usage.input_tokens is None   # never 0
    again = judge.judge(Q, REF, [], CAND)                       # not cached: asked again, real cost recorded
    assert len(fake.calls) == 2 and again.usage.known is True and again.cache_hit is False


def test_one_attempt_without_usage_makes_the_whole_call_unknown(no_network):
    judge, _ = openai_judge(script=[unknown_usage(make_response(content="bad")), judge_reply()])
    r = judge.judge(Q, REF, [], CAND)
    assert r.status == "ok" and r.usage.known is False


# --------------------------------------------------------------------- builder / config
def _cfg(tmp_path, **kw):
    base = dict(experiment_id="t", seed=0, dataset="hotpotqa", data_source="mock", num_questions=5,
                strategy="random", budget=0.1, embedding_backend="tfidf", ketrag_mode="tfidf",
                fast_use_spacy=False, cache_dir=str(tmp_path / "cache"))
    return ExperimentConfig(**{**base, **kw})


def test_judge_config_defaults_are_a_free_mock():
    cfg = ExperimentConfig(experiment_id="t", seed=0, dataset="hotpotqa", num_questions=5,
                           strategy="random", budget=0.1)
    assert cfg.judge_backend == "mock" and cfg.judge_model == "gpt-4o-mini"
    assert cfg.judge_temperature == 0.0 and cfg.judge_max_output_tokens == 256 and cfg.judge_cache_enabled
    with pytest.raises(Exception):
        ExperimentConfig(experiment_id="t", seed=0, dataset="hotpotqa", num_questions=5,
                         strategy="random", budget=0.1, judge_backend="anthropic")


def test_builder_follows_the_config(tmp_path, no_network):
    assert build_judge(_cfg(tmp_path)).backend == "mock"
    dry = build_judge(_cfg(tmp_path, judge_backend="openai", judge_model="gpt-4o"), dry_run=True)
    assert dry.backend == "openai" and dry.model == "gpt-4o" and dry.estimate_cost_usd(Q, REF, [], CAND) > 0
    with pytest.raises(Exception):                              # a dry-run judge can never send anything
        dry.judge(Q, REF, [], CAND)


def test_builder_caches_under_cache_judge_calls(tmp_path, no_network):
    judge = build_judge(_cfg(tmp_path))
    judge.judge(Q, REF, [], CAND)
    assert any((tmp_path / "cache" / "judge_calls" / "mock" / "mock-v1").glob("*.json"))


def test_missing_key_never_falls_back_to_mock(no_network, no_api_key, monkeypatch, tmp_path):
    import generation.llm_client as module
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: False)
    with pytest.raises(MissingAPIKeyError, match="NOT fall back"):
        build_judge(_cfg(tmp_path, judge_backend="openai"))
