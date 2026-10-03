"""Phase 4: the extraction pipeline.

The central claim tested here: ONLY chunks selected by strategy + budget ever reach
the extractor (or the LLM). Proven in layers - see the "PROOF" test groups below.
No test makes a real API call: the real OpenAI extractor is driven by a fake client.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import hashlib
import json

import pytest

import extraction.pipeline as pipeline
from extraction import MockExtractor
from extraction.base_extractor import ExtractionAbort, Extractor
from extraction.cache import ExtractionCache
from extraction.openai_extractor import DisabledClient, OpenAIExtractor
from extraction.pipeline import (BudgetGate, BudgetViolation, CostCapExceeded, plan_extraction,
                                 run_extraction, select_for_extraction)
from extraction.prompts import build_messages
from extraction.run import main as cli_main
from extraction.schemas import ExtractionInput, ExtractionResult
from src.budget import select_top_budget, selected_count
from src.prepare_data import fingerprint
from helpers_extraction import (FakeOpenAIClient, RecordingExtractor, corpus, good_handler, make_error,
                                make_response, rank_with)

pytestmark = pytest.mark.usefixtures("no_network", "no_api_key")

STRATEGIES = ["random", "ketrag", "lazygraphrag", "fastgraphrag"]
BUDGETS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


@pytest.fixture(scope="module")
def data():
    chunks, questions = corpus()
    return chunks, questions, {name: rank_with(name, chunks) for name in STRATEGIES}


def openai_with(client, **kw):
    return OpenAIExtractor(client=client, sleep=lambda s: None, **kw)


# =====================================================================================
# PROOF 1 - for every strategy and every budget, the extractor sees EXACTLY the
#           budget-selected chunks: all of them, and nothing else.
# =====================================================================================
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("budget", BUDGETS)
def test_extractor_receives_exactly_the_budget_selected_chunks(data, strategy, budget):
    chunks, _, rankings = data
    ranked = rankings[strategy]
    recorder = RecordingExtractor()

    run = run_extraction(chunks, ranked, budget, recorder, cache=None)

    expected = select_top_budget(ranked, budget)               # the EXISTING function, computed independently
    unselected = {c.chunk_id for c in chunks} - set(expected)
    assert recorder.seen == expected                           # same chunks, same (rank) order
    assert not set(recorder.seen) & unselected                 # not a single unselected chunk
    assert len(recorder.seen) == selected_count(len(chunks), budget)
    assert [r.chunk_id for r in run.results] == expected
    if budget < 1.0:
        assert unselected, "sanity: the test is only meaningful if some chunks are excluded"


# =====================================================================================
# PROOF 2 - the selection really comes from the existing select_top_budget(), and the
#           gate blocks anything outside it.
# =====================================================================================
def test_pipeline_selection_is_computed_by_the_existing_select_top_budget(data, monkeypatch):
    chunks, _, rankings = data
    ranked = rankings["random"]
    calls = []

    def spy(ranked_ids, budget):
        calls.append((list(ranked_ids), budget))
        return select_top_budget(ranked_ids, budget)

    monkeypatch.setattr(pipeline, "select_top_budget", spy)
    recorder = RecordingExtractor()
    run_extraction(chunks, ranked, 0.25, recorder)
    assert calls[0] == (ranked, 0.25)
    assert recorder.seen == select_top_budget(ranked, 0.25)


def test_a_wrong_sized_selection_is_blocked_before_anything_is_extracted(data, monkeypatch):
    chunks, _, rankings = data
    ranked = rankings["random"]
    monkeypatch.setattr(pipeline, "select_top_budget", lambda ids, b: list(ids)[:2])
    recorder = RecordingExtractor()
    with pytest.raises(BudgetViolation):
        run_extraction(chunks, ranked, 0.50, recorder)
    assert recorder.seen == []


def test_gate_raises_for_any_chunk_outside_the_selected_ids_and_does_not_call_through():
    recorder = RecordingExtractor()
    gate = BudgetGate(recorder, ["allowed-1"])
    gate.extract(ExtractionInput(chunk_id="allowed-1", text="Tom Hanks."))
    with pytest.raises(BudgetViolation):
        gate.extract(ExtractionInput(chunk_id="intruder", text="Robert Zemeckis."))
    assert recorder.seen == ["allowed-1"]


def test_extractor_answering_for_a_different_chunk_is_rejected(data):
    chunks, _, rankings = data

    class Wrong(MockExtractor):
        def extract(self, item):
            return super().extract(ExtractionInput(chunk_id="not-the-requested-id", text=item.text))

    with pytest.raises(BudgetViolation):
        run_extraction(chunks, rankings["random"], 0.10, Wrong())


@pytest.mark.parametrize("bad", [
    lambda ids: ids[:-1],                  # a chunk missing
    lambda ids: ids + ids[:1],             # a chunk twice
    lambda ids: ids + ["made-up-id"],      # an id that is not a chunk
])
def test_a_ranking_that_is_not_a_full_permutation_is_rejected(data, bad):
    chunks, _, rankings = data
    recorder = RecordingExtractor()
    with pytest.raises(ValueError):
        run_extraction(chunks, bad(list(rankings["random"])), 0.25, recorder)
    assert recorder.seen == []


# =====================================================================================
# PROOF 3 - the cache cannot smuggle extractions of unselected chunks into a run.
# =====================================================================================
def test_cache_filled_for_all_chunks_still_yields_only_the_selected_chunks(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["random"]
    cache = ExtractionCache(tmp_path / "cache")
    run_extraction(chunks, ranked, 1.0, MockExtractor(), cache)            # cache now holds EVERY chunk
    assert len(list((tmp_path / "cache").rglob("*.json"))) == len(chunks)

    recorder = RecordingExtractor()
    run = run_extraction(chunks, ranked, 0.10, recorder, cache, run_dir=tmp_path / "run")

    expected = select_top_budget(ranked, 0.10)
    assert recorder.seen == []                                             # everything came from the cache
    assert [r.chunk_id for r in run.results] == expected                   # ...but ONLY the selected chunks
    assert all(r.cache_hit for r in run.results)
    saved = [json.loads(line)["chunk_id"] for line in (tmp_path / "run" / "extractions.jsonl").read_text().splitlines()]
    assert saved == expected                                               # and the saved file agrees
    assert run.summary["n_cache_hits"] == len(expected) < len(chunks)


# =====================================================================================
# PROOF 4 - what is actually sent to the (fake) OpenAI API: only selected chunk texts,
#           no questions, and gold labels cannot change a single byte of a request.
# =====================================================================================
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_openai_requests_contain_only_the_selected_chunks(data, strategy):
    chunks, questions, rankings = data
    ranked = rankings[strategy]
    client = FakeOpenAIClient(handler=good_handler())

    run_extraction(chunks, ranked, 0.25, openai_with(client))

    expected_ids = select_top_budget(ranked, 0.25)
    by_id = {c.chunk_id: c for c in chunks}
    sent = [call["messages"] for call in client.calls]
    assert sent == [build_messages(by_id[i].text) for i in expected_ids]   # exact: these chunks, this order
    assert len(client.calls) == len(expected_ids) < len(chunks)
    for chunk in chunks:                                                    # no unselected chunk was sent
        if chunk.chunk_id not in expected_ids:
            assert build_messages(chunk.text) not in sent
    everything_sent = json.dumps(client.calls)
    for q in questions:                                                     # no question text was sent
        assert q["question"] not in everything_sent


def test_gold_labels_cannot_influence_what_is_sent(data):
    chunks, _, rankings = data
    assert any(c.is_gold_for_question_ids for c in chunks), "sanity: real gold labels exist"
    scrambled = copy.deepcopy(chunks)
    for c in scrambled:
        c.is_gold_for_question_ids = [] if c.is_gold_for_question_ids else ["q1", "q2"]

    real_client, scrambled_client = FakeOpenAIClient(handler=good_handler()), FakeOpenAIClient(handler=good_handler())
    run_extraction(chunks, rankings["random"], 0.50, openai_with(real_client))
    run_extraction(scrambled, rankings["random"], 0.50, openai_with(scrambled_client))
    assert real_client.calls == scrambled_client.calls                      # byte-identical requests


def test_the_saved_run_can_be_audited_after_the_fact(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["ketrag"]
    run = run_extraction(chunks, ranked, 0.25, MockExtractor(), run_dir=tmp_path,
                         run_meta={"run_id": "r1", "strategy": "ketrag", "seed": 0})
    selection = json.loads((tmp_path / "selection.json").read_text())
    extracted = [json.loads(l)["chunk_id"] for l in (tmp_path / "extractions.jsonl").read_text().splitlines()]
    expected = select_top_budget(ranked, 0.25)

    assert selection["selected_chunk_ids"] == expected == extracted
    assert selection["selected_fingerprint"] == fingerprint(expected) == run.summary["selected_fingerprint"]
    assert selection["n_selected"] == len(expected) and selection["n_chunks_total"] == len(chunks)
    assert selection["strategy"] == "ketrag" and selection["budget"] == 0.25


# =====================================================================================
# Caching, cost accounting, resume
# =====================================================================================
def test_rerun_makes_no_api_calls_and_cost_accounting_separates_spent_from_uncached(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["random"]
    cache = ExtractionCache(tmp_path / "cache")
    client = FakeOpenAIClient(handler=good_handler(prompt_tokens=1000, completion_tokens=500))
    per_call = 1000 * 0.15 / 1e6 + 500 * 0.60 / 1e6
    n = selected_count(len(chunks), 0.25)

    first = run_extraction(chunks, ranked, 0.25, openai_with(client), cache)
    assert len(client.calls) == n
    assert first.summary["cost_spent_usd"] == pytest.approx(n * per_call, abs=1e-6)
    assert first.summary["n_extractor_calls"] == n and first.summary["n_cache_hits"] == 0

    second = run_extraction(chunks, ranked, 0.25, openai_with(client), cache)
    assert len(client.calls) == n                                           # ZERO new calls
    assert second.summary["n_cache_hits"] == n and second.summary["n_extractor_calls"] == 0
    assert second.summary["cost_spent_usd"] == 0.0
    assert second.summary["cost_if_uncached_usd"] == pytest.approx(first.summary["cost_if_uncached_usd"])
    assert second.summary["tokens_spent_this_run"] == {"input": 0, "output": 0}


def test_larger_budget_reuses_the_smaller_budgets_extractions(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["random"]
    cache = ExtractionCache(tmp_path / "cache")
    client = FakeOpenAIClient(handler=good_handler())
    run_extraction(chunks, ranked, 0.10, openai_with(client), cache)
    small = len(client.calls)
    run_extraction(chunks, ranked, 0.50, openai_with(client), cache)
    assert len(client.calls) == selected_count(len(chunks), 0.50)           # only the NEW chunks were sent
    assert len(client.calls) - small == selected_count(len(chunks), 0.50) - small


def test_summary_records_model_tokens_cost_runtime_status(data):
    chunks, _, rankings = data
    client = FakeOpenAIClient(handler=good_handler())
    s = run_extraction(chunks, rankings["random"], 0.10, openai_with(client)).summary
    assert s["backend"] == "openai" and s["model"] == "gpt-4o-mini" and s["is_mock"] is False
    assert s["n_ok"] == s["n_processed"] == s["n_selected"] and s["n_failed"] == 0
    assert s["status_counts"] == {"ok": s["n_selected"]}
    assert s["tokens_spent_this_run"]["input"] > 0 and s["cost_spent_usd"] > 0
    assert s["wall_time_seconds"] >= 0 and s["extraction_runtime_seconds_this_run"] >= 0
    assert s["cache_settings"]["prompt_version"] and s["aborted"] is None


def test_mock_runs_are_labelled_mock_in_the_summary(data):
    chunks, _, rankings = data
    s = run_extraction(chunks, rankings["random"], 0.10, MockExtractor()).summary
    assert s["is_mock"] is True and s["backend"] == "mock" and s["cost_spent_usd"] == 0.0


def test_failed_chunks_are_reported_not_cached_and_retried_on_the_next_run(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["random"]
    cache = ExtractionCache(tmp_path / "cache")
    first_id = select_top_budget(ranked, 0.25)[0]
    bad_text = next(c.text for c in chunks if c.chunk_id == first_id)
    state = {"break_it": True}

    def handler(kwargs):
        if state["break_it"] and bad_text in kwargs["messages"][1]["content"]:
            return make_response(content="not json at all")
        return make_response()

    client = FakeOpenAIClient(handler=handler)
    n = selected_count(len(chunks), 0.25)
    first = run_extraction(chunks, ranked, 0.25, openai_with(client, max_retries=1), cache)
    assert first.summary["n_failed"] == 1 and first.summary["failed_chunk_ids"] == [first_id]
    assert first.summary["status_counts"] == {"malformed": 1, "ok": n - 1}
    assert len(list((tmp_path / "cache").rglob("*.json"))) == n - 1          # the failure was NOT cached

    state["break_it"] = False
    calls_before = len(client.calls)
    second = run_extraction(chunks, ranked, 0.25, openai_with(client, max_retries=1), cache)
    assert len(client.calls) - calls_before == 1                             # ONLY the failed chunk re-sent
    assert second.summary["n_failed"] == 0 and second.summary["n_cache_hits"] == n - 1


# =====================================================================================
# Run-level problems: bad key aborts, spending cap, dry run
# =====================================================================================
def test_authentication_error_aborts_immediately_and_saves_partial_results(data, tmp_path):
    chunks, _, rankings = data
    client = FakeOpenAIClient([make_response(), make_response(), make_error("AuthenticationError", 401)])
    with pytest.raises(ExtractionAbort):
        run_extraction(chunks, rankings["random"], 0.25, openai_with(client), run_dir=tmp_path)
    assert len(client.calls) == 3                                            # stopped at the first fatal error
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["aborted"]["reason"] == "fatal_error" and summary["n_processed"] == 2
    assert len((tmp_path / "extractions.jsonl").read_text().splitlines()) == 2


def test_spending_cap_is_checked_before_any_call_is_made(data, tmp_path):
    chunks, _, rankings = data
    client = FakeOpenAIClient(handler=good_handler())
    with pytest.raises(CostCapExceeded, match="nothing was sent"):
        run_extraction(chunks, rankings["random"], 0.25, openai_with(client),
                       max_cost_usd=1e-9, run_dir=tmp_path / "run")
    assert client.calls == [] and not (tmp_path / "run").exists()


def test_spending_cap_stops_mid_run_saves_progress_and_a_rerun_resumes(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["random"]
    cache = ExtractionCache(tmp_path / "cache")
    n = selected_count(len(chunks), 0.25)
    # Each call REALLY costs ~$0.015 (100k input tokens) although the pre-run estimate is tiny,
    # so the pre-flight check passes and the mid-run check has to catch it.
    expensive = FakeOpenAIClient(handler=good_handler(prompt_tokens=100_000, completion_tokens=50))

    with pytest.raises(CostCapExceeded, match="spending cap"):
        run_extraction(chunks, ranked, 0.25, openai_with(expensive), cache,
                       max_cost_usd=0.04, run_dir=tmp_path / "run")
    assert len(expensive.calls) == 3                                         # 3 x $0.015 = $0.045 > cap, so no 4th
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert summary["aborted"]["reason"] == "cost_cap" and summary["n_processed"] == 3

    cheap = FakeOpenAIClient(handler=good_handler())
    resumed = run_extraction(chunks, ranked, 0.25, openai_with(cheap), cache)
    assert len(cheap.calls) == n - 3 and resumed.summary["n_cache_hits"] == 3   # finished work was kept


def test_dry_run_plan_calls_nothing_and_knows_what_is_cached(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["random"]
    cache = ExtractionCache(tmp_path / "cache")
    dry = OpenAIExtractor(client=DisabledClient())                           # any call would raise
    n = selected_count(len(chunks), 0.25)

    before = plan_extraction(chunks, ranked, 0.25, dry, cache)
    assert before["n_selected"] == n and before["n_to_extract"] == n and before["n_already_cached"] == 0
    assert before["estimated_cost_usd"] > 0 and "ESTIMATE" in before["estimate_note"]

    run_extraction(chunks, ranked, 0.25, openai_with(FakeOpenAIClient(handler=good_handler())), cache)
    after = plan_extraction(chunks, ranked, 0.25, dry, cache)
    assert after["n_already_cached"] == n and after["n_to_extract"] == 0 and after["estimated_cost_usd"] == 0.0


def test_select_for_extraction_reports_selection_details(data):
    chunks, _, rankings = data
    sel = select_for_extraction(chunks, rankings["fastgraphrag"], 0.10)
    assert list(sel.selected_ids) == select_top_budget(rankings["fastgraphrag"], 0.10)
    assert [c.chunk_id for c in sel.selected_chunks] == list(sel.selected_ids)
    assert sel.n_total == len(chunks) and sel.fingerprint == fingerprint(list(sel.selected_ids))


def test_extractor_interface_is_abstract():
    with pytest.raises(TypeError):
        Extractor()


# =====================================================================================
# The command line (one config -> one extraction run)
# =====================================================================================
def write_cfg(tmp_path, **overrides):
    settings = {
        "experiment_id": "cli_test", "seed": 42, "dataset": "hotpotqa", "data_source": "mock",
        "num_questions": 5, "chunk_size_words": 250, "chunk_overlap_words": 40,
        "strategy": "random", "budget": 0.25, "ketrag_mode": "tfidf", "fast_use_spacy": "false",
        "extraction_backend": "mock",
        "cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "results"),
    }
    settings.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(f"{k}: {v}" for k, v in settings.items()), encoding="utf-8")
    return str(path)


def only_run_dir(tmp_path):
    (run_dir,) = list((tmp_path / "results" / "extractions").iterdir())
    return run_dir


def test_cli_mock_run_writes_outputs_and_the_second_run_is_all_cache_hits(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert cli_main(["--config", cfg]) == 0
    run_dir = only_run_dir(tmp_path)
    assert {p.name for p in run_dir.iterdir()} == {"extractions.jsonl", "selection.json", "run_summary.json"}
    first = json.loads((run_dir / "run_summary.json").read_text())
    assert first["is_mock"] and first["n_selected"] == 3 and first["n_cache_hits"] == 0
    assert "MOCK" in capsys.readouterr().out

    assert cli_main(["--config", cfg]) == 0
    second = json.loads((run_dir / "run_summary.json").read_text())
    assert second["n_cache_hits"] == 3 and second["n_extractor_calls"] == 0


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    assert cli_main(["--config", write_cfg(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "n_to_extract: 3" in out
    assert not (tmp_path / "results").exists() and not (tmp_path / "cache").exists()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_cli_runs_every_budgeted_strategy_end_to_end(tmp_path, strategy):
    assert cli_main(["--config", write_cfg(tmp_path, strategy=strategy)]) == 0
    summary = json.loads((only_run_dir(tmp_path) / "run_summary.json").read_text())
    assert summary["strategy"] == strategy and summary["n_ok"] == summary["n_selected"] == 3


def test_cli_openai_backend_without_a_key_stops_and_does_not_use_the_mock(tmp_path, capsys):
    code = cli_main(["--config", write_cfg(tmp_path, extraction_backend="openai")])
    assert code == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and "NOT fall back" in err
    assert not (tmp_path / "results").exists() and not (tmp_path / "cache").exists()


def test_cli_openai_dry_run_works_without_a_key_and_shows_an_estimate(tmp_path, capsys):
    code = cli_main(["--config", write_cfg(tmp_path, extraction_backend="openai",
                                           extraction_max_cost_usd=1.0), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0 and "backend: openai" in out and "estimated_cost_usd" in out
    assert "within the cap" in out
    assert not (tmp_path / "results").exists()


def test_cli_rejects_the_native_lazygraphrag_reference(tmp_path, capsys):
    assert cli_main(["--config", write_cfg(tmp_path, strategy="lazygraphrag_native")]) == 2
    assert "not a budgeted selection strategy" in capsys.readouterr().err


# =====================================================================================
# TRUNCATED OUTPUT: a deterministic failure - one call, status "truncated", never retried,
# and handled IDENTICALLY for every strategy (so it cannot favour one of them)
# =====================================================================================
def _doomed(user_message: str) -> bool:
    """Deterministic stand-in for 'this chunk's extraction is too big for the output limit'."""
    return int(hashlib.sha256(user_message.encode()).hexdigest(), 16) % 2 == 0


def truncating_handler(kwargs):
    if _doomed(kwargs["messages"][1]["content"]):
        return make_response(content='{"entities": [{"name": "Tom', finish_reason="length",
                             prompt_tokens=700, completion_tokens=kwargs["max_completion_tokens"])
    return make_response()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_truncation_is_handled_identically_for_every_strategy(data, strategy):
    chunks, _, rankings = data
    ranked = rankings[strategy]
    selected = select_top_budget(ranked, 0.50)
    by_id = {c.chunk_id: c for c in chunks}
    doomed = {i for i in selected if _doomed(build_messages(by_id[i].text)[1]["content"])}
    assert doomed and doomed != set(selected), "sanity: the selection must mix truncating and fine chunks"

    client = FakeOpenAIClient(handler=truncating_handler)
    run = run_extraction(chunks, ranked, 0.50, openai_with(client, max_retries=3))

    assert len(client.calls) == len(selected)                  # exactly ONE call per chunk: no retry cost
    s = run.summary
    assert s["n_truncated"] == len(doomed)
    assert s["status_counts"] == {"truncated": len(doomed), "ok": len(selected) - len(doomed)}
    assert set(s["failed_chunk_ids"]) == doomed and s["n_failed"] == len(doomed)
    for r in run.results:
        if r.chunk_id in doomed:
            assert r.status == "truncated" and r.attempts == 1 and r.entities == []
            assert "not retried" in r.error
        else:
            assert r.status == "ok" and r.attempts == 1
    assert s["usage_complete"] is True                         # a truncated call is still paid for, and counted
    assert s["cost_spent_usd"] > 0 and s["n_api_attempts"] == len(selected)


def test_truncated_chunks_are_not_cached_and_the_next_run_resends_only_those(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["random"]
    cache = ExtractionCache(tmp_path / "cache")
    selected = select_top_budget(ranked, 0.50)
    by_id = {c.chunk_id: c for c in chunks}
    doomed = {i for i in selected if _doomed(build_messages(by_id[i].text)[1]["content"])}

    first_client = FakeOpenAIClient(handler=truncating_handler)
    run_extraction(chunks, ranked, 0.50, openai_with(first_client), cache)
    assert len(list((tmp_path / "cache").rglob("*.json"))) == len(selected) - len(doomed)   # no truncated entry

    second_client = FakeOpenAIClient(handler=good_handler())      # e.g. after raising the output cap
    second = run_extraction(chunks, ranked, 0.50, openai_with(second_client), cache)
    assert len(second_client.calls) == len(doomed)                # ONLY the truncated chunks, once each
    assert second.summary["n_truncated"] == 0 and second.summary["n_failed"] == 0
    assert second.summary["n_cache_hits"] == len(selected) - len(doomed)


def test_transient_errors_are_still_retried_inside_the_pipeline(data):
    chunks, _, rankings = data
    n = selected_count(len(chunks), 0.25)
    client = FakeOpenAIClient(script=[make_error("RateLimitError", 429)], handler=good_handler())
    run = run_extraction(chunks, rankings["random"], 0.25, openai_with(client, max_retries=3))
    assert len(client.calls) == n + 1                              # one extra call: the retried 429
    assert run.summary["n_failed"] == 0 and run.summary["n_ok"] == n


# =====================================================================================
# UNKNOWN USAGE: never silently 0 tokens / $0 anywhere in a run
# =====================================================================================
def without_usage(response):
    response.usage = None
    return response


def no_usage_handler(kwargs):
    return without_usage(make_response())


PER_CALL = 100 * 0.15 / 1e6 + 50 * 0.60 / 1e6                     # make_response default: 100 in / 50 out


def test_a_run_with_complete_usage_reports_no_unknowns(data):
    chunks, _, rankings = data
    s = run_extraction(chunks, rankings["random"], 0.25, openai_with(FakeOpenAIClient(handler=good_handler()))).summary
    assert s["usage_complete"] is True and s["n_usage_unknown"] == 0 and s["usage_unknown_chunk_ids"] == []
    assert s["usage_known_part_lower_bound"] is None
    assert isinstance(s["cost_spent_usd"], float) and s["cost_spent_usd"] > 0


def test_unknown_usage_is_never_reported_as_free_in_the_run_summary(data):
    chunks, _, rankings = data
    ranked = rankings["random"]
    selected = select_top_budget(ranked, 0.25)
    n = len(selected)
    client = FakeOpenAIClient(script=[make_response(), no_usage_handler(None)], handler=good_handler())

    run = run_extraction(chunks, ranked, 0.25, openai_with(client))
    s = run.summary

    assert s["n_ok"] == n                                           # the extraction content is intact
    assert s["usage_complete"] is False
    assert s["n_usage_unknown"] == 1 and s["usage_unknown_chunk_ids"] == [selected[1]]
    # Every total that includes the unknown chunk is None - NOT 0, and NOT a partial sum.
    assert s["cost_spent_usd"] is None and s["cost_if_uncached_usd"] is None
    assert s["tokens_spent_this_run"] == {"input": None, "output": None}
    # What IS known is reported separately and labelled as a lower bound.
    known = s["usage_known_part_lower_bound"]
    assert known["tokens_input_this_run"] == (n - 1) * 100 and known["tokens_output_this_run"] == (n - 1) * 50
    assert known["cost_spent_usd"] == pytest.approx((n - 1) * PER_CALL, abs=1e-6)
    assert known["cost_if_uncached_usd"] == pytest.approx((n - 1) * PER_CALL, abs=1e-6)
    # The per-chunk record says unknown too, and no number was invented for it.
    unknown_result = run.results[1]
    assert unknown_result.usage.known is False and unknown_result.usage.cost_usd is None


def test_results_with_unknown_usage_are_not_cached_so_the_next_run_extracts_them_again(data, tmp_path):
    chunks, _, rankings = data
    ranked = rankings["random"]
    cache = ExtractionCache(tmp_path / "cache")
    n = selected_count(len(chunks), 0.25)

    first = run_extraction(chunks, ranked, 0.25,
                           openai_with(FakeOpenAIClient(script=[no_usage_handler(None)], handler=good_handler())), cache)
    assert first.summary["usage_complete"] is False
    assert len(list((tmp_path / "cache").rglob("*.json"))) == n - 1          # the unknown one was NOT cached

    healed = FakeOpenAIClient(handler=good_handler())
    second = run_extraction(chunks, ranked, 0.25, openai_with(healed), cache)
    assert len(healed.calls) == 1                                           # only that chunk, now with real usage
    assert second.summary["usage_complete"] is True and second.summary["n_cache_hits"] == n - 1
    assert second.summary["cost_if_uncached_usd"] is not None


def test_the_spending_cap_guard_never_counts_unknown_usage_as_zero(data):
    chunks, _, rankings = data
    ranked = rankings["random"]
    selected = select_top_budget(ranked, 0.25)
    by_id = {c.chunk_id: c for c in chunks}
    est = [OpenAIExtractor(client=DisabledClient()).estimate_cost_usd(
               ExtractionInput(chunk_id=i, text=by_id[i].text)) for i in selected[:3]]
    big = 100_000 * 0.15 / 1e6 + 50 * 0.60 / 1e6                            # call 1 REALLY costs this
    # Cap fits call 1 (real cost) + call 2 (unknown, counted at its estimate), but NOT call 3 after that.
    # If unknown were counted as $0, call 3 would still fit under this cap.
    cap = big + max(est[1], est[2]) + 1e-9
    assert big + est[2] <= cap < big + est[1] + est[2]

    client = FakeOpenAIClient(script=[make_response(prompt_tokens=100_000, completion_tokens=50),
                                      no_usage_handler(None)])
    with pytest.raises(CostCapExceeded):
        run_extraction(chunks, ranked, 0.25, openai_with(client), max_cost_usd=cap)
    assert len(client.calls) == 2                                           # stopped BEFORE call 3


# =====================================================================================
# The command line: exit code and wording for truncation / unknown usage
# =====================================================================================
def _use_extractor(monkeypatch, handler):
    extractor = OpenAIExtractor(client=FakeOpenAIClient(handler=handler), sleep=lambda s: None)
    monkeypatch.setattr("extraction.run.build_extractor", lambda cfg, dry_run=False: extractor)


def test_cli_with_unknown_usage_exits_1_and_shows_unknown_never_zero_dollars(tmp_path, monkeypatch, capsys):
    _use_extractor(monkeypatch, no_usage_handler)
    assert cli_main(["--config", write_cfg(tmp_path, extraction_backend="openai")]) == 1
    captured = capsys.readouterr()
    assert "cost spent this run: UNKNOWN" in captured.out and "cost if uncached: UNKNOWN" in captured.out
    assert "cost spent this run: $" not in captured.out                      # never printed as a dollar amount
    assert "no token usage" in captured.err and "lower bound" in captured.err
    summary = json.loads((only_run_dir(tmp_path) / "run_summary.json").read_text())
    assert summary["cost_spent_usd"] is None and summary["usage_complete"] is False


def test_cli_with_truncated_chunks_exits_1_and_tells_you_the_fix(tmp_path, monkeypatch, capsys):
    _use_extractor(monkeypatch, lambda kw: make_response(content='{"entities": [', finish_reason="length"))
    assert cli_main(["--config", write_cfg(tmp_path, extraction_backend="openai")]) == 1
    err = capsys.readouterr().err
    assert "TRUNCATED" in err and "extraction_max_output_tokens" in err
    summary = json.loads((only_run_dir(tmp_path) / "run_summary.json").read_text())
    assert summary["n_truncated"] == summary["n_selected"] == 3 and summary["n_extractor_calls"] == 3
    assert summary["n_api_attempts"] == 3                                   # one attempt each: no retries


def test_cli_with_a_clean_openai_run_still_exits_0(tmp_path, monkeypatch, capsys):
    _use_extractor(monkeypatch, good_handler())
    assert cli_main(["--config", write_cfg(tmp_path, extraction_backend="openai")]) == 0
    out = capsys.readouterr().out
    assert "UNKNOWN" not in out
    summary = json.loads((only_run_dir(tmp_path) / "run_summary.json").read_text())
    assert summary["usage_complete"] is True and summary["cost_spent_usd"] > 0
