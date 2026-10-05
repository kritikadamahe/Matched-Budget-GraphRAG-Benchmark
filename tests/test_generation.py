"""Blueprint Phase 10: answer generation + native L4 relevance check.
The OpenAI path is tested with a fake client: no key, no network, no cost."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pytest

from extraction.base_extractor import ExtractionAbort
from extraction.openai_extractor import MissingAPIKeyError
from generation import NOT_FOUND, Answerer, RelevanceChecker, clean_answer, summarize_query_usage
from generation.answerer import build_answerer, build_relevance_checker
from generation.cache import QueryCache
from generation.llm_client import ChatJSONClient
from generation.prompts import build_answer_messages
from native import build_native
from src.config import ExperimentConfig
from src.corpus import build_chunk_manifest
from tests.helpers_extraction import FakeOpenAIClient, make_error, make_response

CTX = "Facts:\n- Forrest Gump -- directed by -- Robert Zemeckis\n\nPassages:\n[1] Robert Zemeckis was born in Chicago."
Q = "Where was the director of Forrest Gump born?"


def openai_answerer(script=None, handler=None, cache=None, **client_kw):
    fake = FakeOpenAIClient(script=script, handler=handler)
    client = ChatJSONClient(client=fake, sleep=lambda s: None, **client_kw)
    return Answerer("openai", client=client, cache=cache), fake


def reply(answer="Chicago", reasoning="Zemeckis directed it and was born in Chicago.", **kw):
    return make_response(payload={"reasoning": reasoning, "answer": answer}, **kw)


# ---------------------------------------------------------------- clean_answer
@pytest.mark.parametrize("raw,clean", [
    ("Chicago.", "Chicago"), ('"Chicago"', "Chicago"), ("  New   York  City ", "New York City"),
    ("Not found.", NOT_FOUND), ("unknown", NOT_FOUND), ("Washington D.C.", "Washington D.C."),
    ("yes", "yes"), ("1831", "1831"),
])
def test_clean_answer(raw, clean):
    assert clean_answer(raw) == clean


# ------------------------------------------------------------------- prompt
def test_prompt_follows_the_blueprint_template():
    system, user = build_answer_messages(Q, CTX)
    assert "ONLY the provided context" in system["content"] and '"not found"' in system["content"]
    assert user["content"] == f"Context:\n{CTX}\n\nQuestion: {Q}\nAnswer:"


# ------------------------------------------------------------- openai path
def test_openai_answer_ok_parses_to_a_clean_short_answer(no_network):
    answerer, fake = openai_answerer(script=[reply(answer="Chicago.")])
    r = answerer.answer(Q, CTX)
    assert r.status == "ok" and r.answer == "Chicago" and "Chicago" in r.reasoning   # Phase 10 checkpoint
    assert r.usage.input_tokens == 100 and r.usage.output_tokens == 50 and r.usage.cost_usd > 0
    sent = fake.calls[0]
    assert sent["model"] == "gpt-4o-mini" and sent["temperature"] == 0.0
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["messages"] == build_answer_messages(Q, CTX)


def test_empty_context_skips_the_call_and_costs_nothing(no_network):
    answerer, fake = openai_answerer(script=[])
    r = answerer.answer(Q, "   ")
    assert r.status == "skipped_empty_context" and r.answer == NOT_FOUND
    assert fake.calls == [] and r.usage.cost_usd == 0.0 and r.attempts == 0


def test_temporary_errors_are_retried_then_succeed(no_network):
    answerer, fake = openai_answerer(script=[make_error("RateLimitError", status=429), reply()])
    r = answerer.answer(Q, CTX)
    assert r.status == "ok" and r.attempts == 2 and len(fake.calls) == 2


def test_fatal_error_stops_the_run(no_network):
    answerer, _ = openai_answerer(script=[make_error("AuthenticationError", status=401)])
    with pytest.raises(ExtractionAbort):
        answerer.answer(Q, CTX)


def test_malformed_reply_is_retried_then_marked(no_network):
    answerer, fake = openai_answerer(script=[make_response(content="not json")] * 4, max_retries=3)
    r = answerer.answer(Q, CTX)
    assert r.status == "malformed" and r.answer == "" and len(fake.calls) == 4
    assert r.usage.input_tokens == 400          # every attempt is paid for and counted


def test_truncated_reply_is_not_retried(no_network):
    answerer, fake = openai_answerer(script=[reply(finish_reason="length")])
    r = answerer.answer(Q, CTX)
    assert r.status == "truncated" and len(fake.calls) == 1


def test_missing_usage_is_unknown_not_free(no_network):
    resp = reply()
    resp.usage = None
    answerer, _ = openai_answerer(script=[resp])
    r = answerer.answer(Q, CTX)
    assert r.status == "ok" and not r.usage.known
    summary = summarize_query_usage([r])
    assert summary["usage_complete"] is False and summary["cost_if_uncached_usd"] is None


def test_missing_key_never_falls_back_to_mock(no_network, no_api_key, monkeypatch):
    import generation.llm_client as module
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: False)
    with pytest.raises(MissingAPIKeyError, match="NOT fall back"):
        ChatJSONClient()


# --------------------------------------------------------------------- cache
def test_answers_are_cached_and_reused(tmp_path, no_network):
    cache = QueryCache(tmp_path)
    first, fake1 = openai_answerer(script=[reply()], cache=cache)
    r1 = first.answer(Q, CTX)
    second, fake2 = openai_answerer(script=[], cache=cache)
    r2 = second.answer(Q, CTX)
    assert fake2.calls == [] and r2.cache_hit and r2.answer == r1.answer
    assert r2.usage.cost_usd == r1.usage.cost_usd                      # original cost still reported
    s = summarize_query_usage([r2])
    assert s["cost_spent_usd"] == 0 and s["cost_if_uncached_usd"] == r1.usage.cost_usd


def test_failed_answers_are_not_cached(tmp_path, no_network):
    cache = QueryCache(tmp_path)
    openai_answerer(script=[reply(finish_reason="length")], cache=cache)[0].answer(Q, CTX)
    again, fake = openai_answerer(script=[reply()], cache=cache)
    assert again.answer(Q, CTX).status == "ok" and len(fake.calls) == 1


def test_different_context_is_a_different_cache_entry(tmp_path, no_network):
    cache = QueryCache(tmp_path)
    openai_answerer(script=[reply()], cache=cache)[0].answer(Q, CTX)
    again, fake = openai_answerer(script=[reply()], cache=cache)
    again.answer(Q, CTX + " More text.")
    assert len(fake.calls) == 1


# ---------------------------------------------------------------- relevance
def test_relevance_checker_openai_and_failure_counts_as_not_relevant(no_network):
    fake = FakeOpenAIClient(script=[make_response(payload={"relevant": True}),
                                    make_response(content="garbage")])
    checker = RelevanceChecker("openai", client=ChatJSONClient(client=fake, sleep=lambda s: None, max_retries=0))
    assert checker(Q, "Robert Zemeckis was born in Chicago.") is True
    assert checker(Q, "Something else.") is False
    assert [r.status for r in checker.results] == ["ok", "malformed"]


def test_mock_backends_are_free_and_deterministic():
    a = Answerer("mock").answer(Q, CTX)
    b = Answerer("mock").answer(Q, CTX)
    assert a.status == "ok" and a.answer == b.answer and a.usage.cost_usd == 0.0
    assert RelevanceChecker("mock")(Q, "the director of Forrest Gump was born in Chicago") is True


# --------------------------------------------------------- config + native L4
def _cfg(tmp_path, **kw):
    base = dict(experiment_id="t", seed=0, dataset="hotpotqa", data_source="mock", num_questions=5,
                strategy="lazygraphrag_native", budget=1.0, embedding_backend="tfidf",
                fast_use_spacy=False, cache_dir=str(tmp_path / "cache"))
    return ExperimentConfig(**{**base, **kw})


def test_builders_follow_the_config(tmp_path, no_network):
    assert build_answerer(_cfg(tmp_path)).backend == "mock"
    dry = build_answerer(_cfg(tmp_path, generation_backend="openai"), dry_run=True)
    assert dry.backend == "openai" and dry.estimate_cost_usd(Q, CTX) > 0
    assert dry.estimate_cost_usd(Q, "") == 0.0
    assert build_relevance_checker(_cfg(tmp_path)).backend == "mock"


def test_native_l4_end_to_end_with_relevance_checker(tmp_path):
    cfg = _cfg(tmp_path, native_relevance_budget=6)
    chunks, questions, _ = build_chunk_manifest("hotpotqa", "mock", 5, 42, 250, 40)
    system = build_native(cfg)
    assert system.index(chunks)["index_cost_usd"] == 0.0
    checker, answerer = build_relevance_checker(cfg), build_answerer(cfg)
    for q in questions:
        found, info = system.retrieve(q["question"], checker)
        assert info["relevance_tests"] <= 6
        r = answerer.answer(q["question"], system.build_context(found, 1500))
        assert r.status in ("ok", "skipped_empty_context")
    assert len(checker.results) <= 6 * len(questions)


def test_demo_cli_graph_path(tmp_path, capsys):
    from extraction.run import main as extraction_cli
    from generation.demo import main as demo_cli
    from graph.build import main as graph_cli
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("\n".join([
        "experiment_id: gen_cli", "seed: 42", "dataset: hotpotqa", "data_source: mock", "num_questions: 5",
        "strategy: random", "budget: 1.0", "embedding_backend: tfidf", "ketrag_mode: tfidf",
        "fast_use_spacy: false", "extraction_backend: mock", "generation_backend: mock",
        f"cache_dir: {tmp_path / 'cache'}", f"output_dir: {tmp_path / 'results'}",
    ]))
    assert demo_cli(["--config", str(cfg)]) == 2                                   # no graph yet
    assert extraction_cli(["--config", str(cfg)]) == 0 and graph_cli(["--config", str(cfg)]) == 0
    capsys.readouterr()
    assert demo_cli(["--config", str(cfg), "--n", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("Predicted:") == 2 and "MOCK, testing only" in out
