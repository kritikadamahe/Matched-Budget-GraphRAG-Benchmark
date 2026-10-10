"""Phase 10 runner: stage orchestration, budget enforcement, gold-label leakage, failure handling,
result persistence, reruns / resumption, caches, and the matrix loop. Mock-backed; no API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pytest

import experiments.runner as runner
from evaluation.schemas import read_predictions
from experiments import (STAGES, Condition, MatrixSpec, generate_conditions, read_condition_result)
from evaluation.judge import Judge
from extraction.mock_extractor import MockExtractor
from generation.answerer import Answerer
from experiments.runner import (Components, MatrixRun, plan_matrix, run_condition, run_matrix, write_manifest,
                                write_results_index)
from src.budget import selected_count
from src.corpus import build_chunk_manifest
from tests.helpers_experiments import (AbortingExtractor, CostlyExtractor, FailingChunksExtractor,
                                       FakeOpenAIAbortingExtractor, FakeOpenAIExtractor, UnknownUsageExtractor, answer_reply, base_cfg,
                                       fake_answerer, fake_judge, judge_verdict, make_ctx, spy_components)
from tests.helpers_extraction import make_error, make_response

COND = Condition("hotpotqa", "random", 0.5, 42)          # 8 chunks in the mock corpus -> 4 selected
N_CHUNKS = 8


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def kinds(events, kind):
    return [e for e in events if e[0] == kind]


# =================================================================== orchestration
def test_one_condition_runs_every_stage_and_writes_every_artifact(tmp_path):
    ctx = make_ctx(tmp_path)
    res = run_condition(COND, ctx)
    d = ctx.condition_dir(COND)
    assert res.status == "completed" and res.clean and res.error is None and res.problems == []
    assert list(res.stages) == list(STAGES) and all(s.status == "ok" for s in res.stages.values())
    for name in ("condition_result.json", "predictions.jsonl", "scored.jsonl", "eval_summary.json",
                 "extraction/extractions.jsonl", "extraction/selection.json", "extraction/run_summary.json",
                 "graph/graph.json", "graph/graph_stats.json"):
        assert (d / name).exists(), name
    assert res == read_condition_result(d)                                           # what was returned is what was saved


def test_the_result_records_identity_config_and_reproducibility_fields(tmp_path):
    ctx = make_ctx(tmp_path)
    res = run_condition(COND, ctx)
    assert (res.condition_id, res.matrix_id) == ("hotpotqa__random__b050__seed42", "t")
    assert (res.dataset, res.strategy, res.budget, res.budget_pct, res.seed) == ("hotpotqa", "random", 0.5, 50, 42)
    assert res.corpus_id == "hotpotqa_mock_3q_seed42_w250o40" and res.run_id.startswith("t__hotpotqa__random__b050__seed42")
    assert res.config["strategy"] == "random" and res.config["budget"] == 0.5 and res.config["seed"] == 42
    assert res.config["experiment_id"] == "t__hotpotqa__random__b050__seed42" and len(res.config_hash) == 16
    assert res.started_at_utc and res.finished_at_utc and res.wall_seconds >= 0


def test_the_stages_call_the_existing_modules_in_order(tmp_path):
    events = []
    ctx = make_ctx(tmp_path, comps=spy_components(events))
    run_condition(COND, ctx)
    names = [e[0] for e in events]
    first = lambda n: names.index(n)
    assert first("load_corpus") < first("rank") < first("build_extractor") < first("extract") \
        < first("make_retriever") < first("retrieve") < first("build_answerer") < first("answer") < first("build_judge")
    assert names.count("rank") == 1 and names.count("build_extractor") == 1 and names.count("build_judge") == 1


def test_stage_records_hold_status_time_and_what_the_module_reported(tmp_path):
    res = run_condition(COND, make_ctx(tmp_path))
    s = res.stages
    assert s["corpus"].details["n_chunks"] == N_CHUNKS and s["corpus"].details["n_questions"] == 3
    assert s["ranking"].details["strategy"] == "random" and s["ranking"].details["n_ranked"] == N_CHUNKS
    assert s["extraction"].details["n_ok"] == 4 and s["extraction"].details["backend"] == "mock"
    assert s["extraction"].details["cost_if_uncached_usd"] == 0.0 and s["extraction"].details["usage_complete"] is True
    assert s["graph"].details["n_nodes"] > 0 and s["graph"].details["provenance_check"] == "passed"
    assert s["retrieval"].details["n_retrieved"] == 3 and s["retrieval"].details["n_failed"] == 0
    assert s["generation"].details["n_predictions"] == 3 and s["generation"].details["status_counts"] == {"ok": 3}
    assert s["evaluation"].details["outcome_counts"]["answered"] == 3
    assert all(isinstance(r.elapsed_seconds, float) and r.elapsed_seconds >= 0 for r in s.values())


def test_predictions_are_evaluation_prediction_records_with_condition_metadata(tmp_path):
    ctx = make_ctx(tmp_path)
    res = run_condition(COND, ctx)
    records = read_predictions(ctx.condition_dir(COND) / "predictions.jsonl")
    assert len(records) == res.n_predictions == 3
    for r in records:
        assert (r.strategy, r.budget, r.seed, r.dataset, r.corpus_id) == ("random", 0.5, 42, "hotpotqa", res.corpus_id)
        assert r.run_id == res.run_id and r.is_mock and r.retrieval is not None and r.generation_settings["backend"] == "mock"
    scored = read(ctx.condition_dir(COND) / "scored.jsonl")
    assert [s["question_id"] for s in scored] == [r.question_id for r in records] and all(s["is_mock"] for s in scored)


def test_headline_metrics_and_cost_are_copied_into_the_result(tmp_path):
    res = run_condition(COND, make_ctx(tmp_path))
    m = res.metrics
    assert m["n_records"] == 3 and m["judge_enabled"] is True and "em" in m and "f1" in m and "judge_accuracy" in m
    assert res.cost["usage_complete"] is True and res.cost["total_cost_if_uncached_usd"] == 0.0
    assert res.cost["stages_included"] == ["extraction", "generation", "judge"]
    assert res.cost["generation_detail"]["n"] == 3


def test_no_judge_skips_the_judge_stage_work(tmp_path):
    events = []
    ctx = make_ctx(tmp_path, comps=spy_components(events), no_judge=True)
    res = run_condition(COND, ctx)
    assert kinds(events, "build_judge") == [] and res.metrics["judge_enabled"] is False
    assert res.cost["stages_included"] == ["extraction", "generation"]
    assert "mock judge backend" not in res.not_reportable_reasons


def test_every_strategy_runs_through_the_same_pipeline(tmp_path):
    for strategy in ("random", "ketrag", "lazygraphrag", "fastgraphrag"):
        res = run_condition(Condition("hotpotqa", strategy, 0.5, 42), make_ctx(tmp_path))
        assert res.status == "completed" and res.n_selected == 4, strategy


# =================================================================== budget enforcement
@pytest.mark.parametrize("budget,expected", [(0.05, 1), (0.10, 1), (0.25, 2), (0.50, 4), (0.75, 6), (1.0, 8)])
def test_only_the_budgeted_number_of_chunks_reaches_the_extractor(tmp_path, budget, expected):
    events = []
    ctx = make_ctx(tmp_path, comps=spy_components(events))
    res = run_condition(Condition("hotpotqa", "random", budget, 42), ctx)
    assert expected == selected_count(N_CHUNKS, budget)
    assert (res.n_chunks_total, res.n_selected_expected, res.n_selected) == (N_CHUNKS, expected, expected)
    extracted = [e[2] for e in kinds(events, "extract")]
    selection = json.loads((ctx.condition_dir(Condition("hotpotqa", "random", budget, 42)) / "extraction" / "selection.json").read_text())
    assert len(extracted) == expected and set(extracted) == set(selection["selected_chunk_ids"])
    assert res.stages["selection"].details["n_selected_expected"] == expected


def test_a_ranking_that_is_not_a_full_permutation_stops_before_any_extractor_exists(tmp_path):
    class Truncated:
        name = "random"

        def rank(self, chunks):
            return [c.chunk_id for c in chunks][:-1]            # one chunk missing

    events = []
    ctx = make_ctx(tmp_path, comps=spy_components(events, strategy_wrap=lambda s: Truncated()))
    res = run_condition(COND, ctx)
    assert res.status == "failed" and res.stages["selection"].status == "failed"
    assert "every chunk_id exactly once" in res.stages["selection"].error
    assert kinds(events, "build_extractor") == [] and kinds(events, "extract") == []
    assert all(res.stages[s].status == "not_run" for s in ("extraction", "graph", "retrieval", "generation", "evaluation"))


def test_selection_is_nested_across_budgets_for_one_strategy(tmp_path):
    ctx = make_ctx(tmp_path)
    ids = {}
    for b in (0.25, 0.5, 1.0):
        c = Condition("hotpotqa", "ketrag", b, 42)
        run_condition(c, ctx)
        ids[b] = json.loads((ctx.condition_dir(c) / "extraction" / "selection.json").read_text())["selected_chunk_ids"]
    assert ids[0.25] == ids[0.5][:2] and ids[0.5] == ids[1.0][:4]               # same ranking, longer prefix


# =================================================================== gold-label leakage
def test_strategy_extraction_and_retrieval_never_see_gold_labels_or_questions(tmp_path):
    real_chunks, questions, _ = build_chunk_manifest("hotpotqa", "mock", 3, 42, 250, 40)
    assert any(c.is_gold_for_question_ids for c in real_chunks)                  # the corpus DOES carry labels ...
    events = []
    ctx = make_ctx(tmp_path, comps=spy_components(events))
    run_condition(COND, ctx)
    (rank,) = kinds(events, "rank")
    assert all(lab == () for lab in rank[2])                                      # ... but the strategy gets none
    (retr,) = kinds(events, "make_retriever")
    assert all(lab == () for lab in retr[1])                                      # nor does retrieval
    assert {e[1] for e in kinds(events, "extract")} == {"ExtractionInput"}        # extractor: (chunk_id, text) only
    assert [e[0] for e in events if e[0] == "rank"] == ["rank"]


def test_retrieval_and_generation_receive_question_text_only(tmp_path):
    _, questions, _ = build_chunk_manifest("hotpotqa", "mock", 3, 42, 250, 40)
    events = []
    run_condition(COND, make_ctx(tmp_path, comps=spy_components(events)))
    asked = {q["question"] for q in questions}
    assert {e[1] for e in kinds(events, "retrieve")} == asked
    assert all(e[1:3] == ("str", "str") and e[3] in asked for e in kinds(events, "answer"))


def test_gold_answers_are_first_read_in_evaluation_after_all_predictions_exist(tmp_path, monkeypatch):
    seen = []
    ctx = make_ctx(tmp_path)
    real = runner.gold_from_questions

    def spy(questions):
        seen.append(((ctx.condition_dir(COND) / "predictions.jsonl").exists(), (ctx.condition_dir(COND) / "scored.jsonl").exists()))
        return real(questions)

    monkeypatch.setattr(runner, "gold_from_questions", spy)
    run_condition(COND, ctx)
    assert seen == [(True, False)]                       # called once: predictions complete, scoring not yet started


def test_no_gold_fields_in_predictions_or_the_condition_result(tmp_path):
    ctx = make_ctx(tmp_path)
    res = run_condition(COND, ctx)

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    forbidden = {"gold", "gold_answer", "gold_aliases", "answer_aliases", "gold_doc_ids", "supporting_paragraphs",
                 "is_gold_for_question_ids", "gold_titles"}
    for line in read(ctx.condition_dir(COND) / "predictions.jsonl"):
        assert not forbidden & set(keys(line))
    assert not forbidden & set(keys(json.loads(res.model_dump_json())))


def test_the_runner_never_imports_evaluation_state_into_earlier_phases():
    root = Path(__file__).resolve().parents[1]
    import re
    pattern = re.compile(r"^\s*(from|import)\s+experiments\b", re.M)
    offenders = [str(p.relative_to(root)) for pkg in ("src", "extraction", "graph", "retrieval", "generation", "strategies",
                                                      "native", "evaluation")
                 for p in (root / pkg).rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


# =================================================================== failures
def test_failed_chunk_extractions_are_recorded_and_the_condition_still_completes(tmp_path):
    ctx = make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: FailingChunksExtractor(every=2)))
    res = run_condition(COND, ctx)
    assert res.status == "completed_with_failures" and not res.clean
    assert res.stages["extraction"].status == "ok" and res.stages["extraction"].details["n_failed"] == 2
    assert any("chunk extraction(s) failed" in p for p in res.problems)
    assert res.stages["graph"].details["n_chunks_skipped_not_ok"] == 2 and res.stages["evaluation"].status == "ok"
    assert res.reportable is False and res.is_mock                                  # still flagged as mock (mock backends)


def test_all_extractions_failing_gives_empty_contexts_and_explicit_abstentions(tmp_path):
    ctx = make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: FailingChunksExtractor(every=1)))
    res = run_condition(COND, ctx)
    assert res.status == "completed_with_failures"
    assert res.stages["retrieval"].details["n_empty_context"] == 3 and res.stages["retrieval"].details["n_empty_graph"] == 3
    assert res.stages["generation"].details["status_counts"] == {"skipped_empty_context": 3}     # no answer call at all
    assert res.metrics["outcome_counts"] == {"answered": 0, "abstained": 3, "empty": 0, "failed": 0}
    assert res.metrics["em"] == 0.0 and res.metrics["n_abstained"] == 3


def test_a_question_whose_retrieval_raises_is_listed_and_has_no_prediction(tmp_path):
    _, questions, _ = build_chunk_manifest("hotpotqa", "mock", 3, 42, 250, 40)
    victim = questions[1]
    ctx = make_ctx(tmp_path, comps=spy_components([], fail_retrieval_on=victim["question"]))
    res = run_condition(COND, ctx)
    assert res.status == "completed_with_failures"
    assert [(f.question_id, f.stage) for f in res.question_failures] == [(victim["question_id"], "retrieval")]
    assert "PageRank did not converge" in res.question_failures[0].error
    assert res.n_questions == 3 and res.n_predictions == 2 and res.stages["retrieval"].details["n_failed"] == 1
    ids = {r.question_id for r in read_predictions(ctx.condition_dir(COND) / "predictions.jsonl")}
    assert victim["question_id"] not in ids and len(ids) == 2
    assert any("no prediction" in p for p in res.problems)


def test_retrieval_failing_for_every_question_fails_the_condition(tmp_path):
    class Boom:
        def retrieve(self, q):
            raise RuntimeError("boom")

    comps = spy_components([])
    comps.make_retriever = lambda cfg, graph, chunks, embedder=None: Boom()
    res = run_condition(COND, make_ctx(tmp_path, comps=comps))
    assert res.status == "failed" and res.stages["retrieval"].status == "failed" and len(res.question_failures) == 3
    assert res.stages["generation"].status == "not_run" and not (make_ctx(tmp_path).condition_dir(COND) / "predictions.jsonl").exists()


def test_failed_answers_are_preserved_with_their_status_and_counted(tmp_path):
    truncated = make_response(content='{"reasoning": "cut', finish_reason="length")
    ctx = make_ctx(tmp_path, comps=spy_components([], answerer=lambda cfg: fake_answerer(script=[truncated] * 3)))
    res = run_condition(COND, ctx)
    assert res.status == "completed_with_failures" and res.stages["generation"].status == "ok"
    assert res.stages["generation"].details["status_counts"] == {"truncated": 3}
    assert res.metrics["n_failed_generations"] == 3 and res.metrics["em"] is None            # not scored, not "wrong"
    assert any("answer generation failed" in p for p in res.problems)
    scored = read(ctx.condition_dir(COND) / "scored.jsonl")
    assert {s["outcome"] for s in scored} == {"failed"} and {s["generation_status"] for s in scored} == {"truncated"}


def test_a_fatal_answer_error_aborts_keeps_partial_predictions_and_skips_evaluation(tmp_path):
    script = [answer_reply("x"), make_error("AuthenticationError", status=401)]
    ctx = make_ctx(tmp_path, comps=spy_components([], answerer=lambda cfg: fake_answerer(script=script)))
    res = run_condition(COND, ctx)
    d = ctx.condition_dir(COND)
    assert res.status == "aborted" and res.stages["generation"].status == "aborted"
    assert res.stages["evaluation"].status == "not_run" and "generation" in res.error
    assert len(read(d / "predictions.jsonl")) == 1 and not (d / "scored.jsonl").exists()      # partial answers kept for debugging
    assert res.n_predictions == 1 and res.reportable is False


def test_a_fatal_extraction_error_aborts_the_condition_and_later_stages_do_not_run(tmp_path):
    ctx = make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: AbortingExtractor(after=1)))
    res = run_condition(COND, ctx)
    assert res.status == "aborted" and res.stages["extraction"].status == "aborted" and "quota" in res.error
    assert res.stages["extraction"].details["aborted"]["reason"] == "fatal_error"             # the partial run's summary was kept
    assert res.stages["extraction"].details["n_processed"] == 1
    assert all(res.stages[s].status == "not_run" for s in ("graph", "retrieval", "generation", "evaluation"))
    assert read_condition_result(ctx.condition_dir(COND)).status == "aborted"


def test_the_spending_cap_stops_before_anything_is_sent(tmp_path):
    events = []
    ctx = make_ctx(tmp_path, comps=spy_components(events, extractor_wrap=lambda e: CostlyExtractor(estimate=1.0)),
                   extraction_max_cost_usd=0.5)
    res = run_condition(COND, ctx)
    assert res.status == "aborted" and "spending cap" in res.error and kinds(events, "extract") == []


def test_a_fatal_judge_error_aborts_but_keeps_em_and_f1(tmp_path):
    ctx = make_ctx(tmp_path, comps=spy_components([], judge=lambda cfg: fake_judge(script=[make_error("AuthenticationError", status=401)])))
    res = run_condition(COND, ctx)
    d = ctx.condition_dir(COND)
    assert res.status == "aborted" and res.stages["evaluation"].status == "aborted"
    scored = read(d / "scored.jsonl")
    assert len(scored) == 3 and all(s["em"] is not None and s["judge_status"] == "not_run" for s in scored)
    assert res.metrics["em"] is not None and res.reportable is False


def test_judge_call_failures_are_counted_not_hidden(tmp_path):
    bad = make_response(content="not json")
    ctx = make_ctx(tmp_path, comps=spy_components([], judge=lambda cfg: fake_judge(handler=lambda kw: bad)))
    res = run_condition(COND, ctx)
    assert res.status == "completed_with_failures" and res.metrics["judge_n_failed"] == 3
    assert any("judge call(s) failed" in p for p in res.problems)


def test_unknown_token_usage_is_a_warning_with_null_totals_never_zero(tmp_path):
    ctx = make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: UnknownUsageExtractor()))
    res = run_condition(COND, ctx)
    assert res.status == "completed"                                              # a data-quality note, not a failure
    assert res.cost["usage_complete"] is False and res.cost["total_cost_if_uncached_usd"] is None
    assert res.cost["total_cost_spent_usd"] is None and res.cost["known_cost_lower_bound_usd"] == 0.0
    assert res.cost["extraction"]["cost_if_uncached_usd"] is None
    assert any("unknown" in w for w in res.warnings)


def test_an_invalid_condition_config_is_a_recorded_failure_not_a_crash(tmp_path):
    ctx = make_ctx(tmp_path, ketrag_mode="faithful", embedding_backend="tfidf")
    res = run_condition(Condition("hotpotqa", "ketrag", 0.5, 42), ctx)
    assert res.status == "failed" and "invalid configuration" in res.error
    assert all(s.status == "not_run" for s in res.stages.values()) and res.run_id is None
    assert read_condition_result(ctx.condition_dir(Condition("hotpotqa", "ketrag", 0.5, 42))).status == "failed"


# =================================================================== mock / reportability
def test_mock_conditions_are_flagged_with_every_reason_and_never_reportable(tmp_path):
    res = run_condition(COND, make_ctx(tmp_path))
    assert res.is_mock and res.reportable is False
    assert res.not_reportable_reasons == ["mock dataset fixture (data_source: mock)", "mock extraction backend",
                                          "mock generation backend", "mock judge backend"]


def test_a_mock_stand_in_under_a_real_config_is_still_not_reportable(tmp_path):
    """Reportability follows what actually ran, not only what the config claims."""
    comps = spy_components([], build_extractor=lambda cfg, dry_run=False: MockExtractor(),
                           build_answerer=lambda cfg, dry_run=False: Answerer("mock"),
                           build_judge=lambda cfg, dry_run=False: Judge("mock"))
    ctx = make_ctx(tmp_path, extraction_backend="openai", generation_backend="openai", judge_backend="openai", comps=comps)
    res = run_condition(COND, ctx)
    assert res.status == "completed" and res.is_mock and not res.reportable
    assert {"mock extraction backend", "mock generation backend", "mock judge backend"} <= set(res.not_reportable_reasons)


def test_a_real_backend_without_a_key_aborts_and_never_falls_back_to_the_mock(tmp_path, no_network, no_api_key, monkeypatch):
    import extraction.openai_extractor as ox
    import generation.llm_client as lc
    monkeypatch.setattr(ox, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(lc, "load_dotenv", lambda *a, **k: False)
    ctx = make_ctx(tmp_path, extraction_backend="openai")                      # the REAL builders, no key
    out = run_matrix([COND], ctx)
    res = out.results[0]
    assert res.status == "aborted" and res.stages["extraction"].status == "aborted"
    assert "NOT fall back" in res.error and out.stopped["reason"] == "aborted"
    assert res.n_predictions == 0 and res.metrics == {} and not (ctx.condition_dir(COND) / "predictions.jsonl").exists()


def test_with_non_mock_backends_and_real_data_the_condition_is_reportable(tmp_path):
    """Plumbing check: non-mock doubles + a non-mock data source -> reportable. (The mock fixture is
    loaded under data_source 'huggingface' only to feed the pipeline.)"""
    comps = spy_components([], build_extractor=lambda cfg, dry_run=False: FakeOpenAIExtractor(),
                           answerer=lambda cfg: fake_answerer(handler=lambda kw: answer_reply("Yes")),
                           judge=lambda cfg: fake_judge(handler=lambda kw: judge_verdict(True)),
                           load_corpus=lambda cfg: build_chunk_manifest(cfg.dataset, "mock", cfg.num_questions, cfg.seed,
                                                                         cfg.chunk_size_words, cfg.chunk_overlap_words))
    ctx = make_ctx(tmp_path, comps=comps, data_source="huggingface", extraction_backend="openai",
                   generation_backend="openai", judge_backend="openai")
    res = run_condition(COND, ctx)
    assert res.status == "completed" and res.is_mock is False and res.reportable is True and res.not_reportable_reasons == []
    assert res.metrics["judge_accuracy"] == 1.0


def test_an_unfinished_condition_is_not_reportable_even_without_mocks(tmp_path):
    comps = spy_components([], build_extractor=lambda cfg, dry_run=False: FakeOpenAIAbortingExtractor(after=0),
                           load_corpus=lambda cfg: build_chunk_manifest(cfg.dataset, "mock", cfg.num_questions, cfg.seed,
                                                                         cfg.chunk_size_words, cfg.chunk_overlap_words))
    res = run_condition(COND, make_ctx(tmp_path, comps=comps, data_source="huggingface", extraction_backend="openai",
                                       generation_backend="openai", judge_backend="openai"))
    assert res.stages["corpus"].status == "ok" and res.stages["extraction"].status == "aborted"      # really got that far
    assert res.status == "aborted" and not res.reportable and not res.is_mock
    assert res.not_reportable_reasons == ["condition did not complete (status aborted)"]


# =================================================================== persistence, reruns, resumption
def test_results_index_has_one_row_per_condition_and_is_rebuilt_not_appended(tmp_path):
    ctx = make_ctx(tmp_path)
    conds = [Condition("hotpotqa", "random", b, 42) for b in (0.25, 0.5)]
    run_matrix(conds, ctx)
    run_matrix(conds, ctx)                                    # again: skipped
    run_matrix(conds, ctx, force=True)                        # again: forced re-run
    rows = read(ctx.matrix_dir / "results.jsonl")
    assert [r["condition_id"] for r in rows] == [c.condition_id for c in conds]
    assert "config" not in rows[0] and rows[0]["config_hash"] and rows[0]["stages"]["extraction"]["details"]


def test_a_completed_condition_is_skipped_on_rerun_and_nothing_is_called(tmp_path):
    ctx = make_ctx(tmp_path)
    first = run_matrix([COND], ctx)
    assert len(first.results) == 1 and first.skipped_completed == []
    events = []
    ctx2 = make_ctx(tmp_path, comps=spy_components(events))
    again = run_matrix([COND], ctx2)
    assert again.results == [] and again.skipped_completed == [COND.condition_id] and events == []


def test_force_reruns_a_completed_condition_and_leaves_no_stale_files(tmp_path):
    ctx = make_ctx(tmp_path)
    run_condition(COND, ctx)
    leftover = ctx.condition_dir(COND) / "extraction" / "leftover_from_old_run.txt"
    leftover.write_text("old")
    events = []
    out = run_matrix([COND], make_ctx(tmp_path, comps=spy_components(events)), force=True)
    assert len(out.results) == 1 and kinds(events, "load_corpus") and not leftover.exists()


def test_an_aborted_condition_is_resumed_and_cached_work_is_not_repaid(tmp_path):
    ctx = make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: AbortingExtractor(after=2)))
    out = run_matrix([COND], ctx)
    assert out.results[0].status == "aborted" and out.stopped["reason"] == "aborted"
    assert (ctx.condition_dir(COND) / "extraction").exists()

    events = []
    out2 = run_matrix([COND], make_ctx(tmp_path, comps=spy_components(events)))               # same cache dir, healthy backend
    res = out2.results[0]
    assert res.status == "completed" and out2.stopped is None
    assert res.stages["extraction"].details["n_cache_hits"] == 2                              # the 2 finished chunks were reused
    assert len(kinds(events, "extract")) == 2                                                  # only the other 2 were extracted
    assert [r["status"] for r in read(ctx.matrix_dir / "results.jsonl")] == ["completed"]


def test_completed_with_failures_is_retried_on_resume(tmp_path):
    ctx = make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: FailingChunksExtractor(every=2)))
    assert run_matrix([COND], ctx).results[0].status == "completed_with_failures"
    out = run_matrix([COND], make_ctx(tmp_path))
    assert out.skipped_completed == [] and out.results[0].status == "completed"                # failed chunks were re-attempted


def test_a_changed_config_makes_a_completed_result_stale_and_it_is_not_overwritten(tmp_path):
    ctx = make_ctx(tmp_path)
    run_matrix([COND], ctx)
    path = ctx.condition_dir(COND) / "condition_result.json"
    before = path.read_bytes()
    events = []
    changed = make_ctx(tmp_path, comps=spy_components(events), retrieval_top_m=7)
    out = run_matrix([COND], changed)
    assert out.stale == [COND.condition_id] and out.results == [] and events == []
    assert path.read_bytes() == before
    forced = run_matrix([COND], changed, force=True)
    assert forced.stale == [] and read_condition_result(ctx.condition_dir(COND)).config["retrieval_top_m"] == 7


def test_an_unreadable_result_file_counts_as_not_done(tmp_path):
    ctx = make_ctx(tmp_path)
    run_condition(COND, ctx)
    (ctx.condition_dir(COND) / "condition_result.json").write_text("{ not json")
    assert read_condition_result(ctx.condition_dir(COND)) is None
    assert len(run_matrix([COND], ctx).results) == 1


def test_a_crashed_condition_without_a_result_file_is_simply_run_again(tmp_path):
    ctx = make_ctx(tmp_path)
    d = ctx.condition_dir(COND)
    (d / "extraction").mkdir(parents=True)
    (d / "extraction" / "garbage.txt").write_text("half-written")
    out = run_matrix([COND], ctx)
    assert out.results[0].status == "completed" and not (d / "extraction" / "garbage.txt").exists()


def test_the_same_config_gives_identical_results_in_a_fresh_directory(tmp_path):
    a = run_condition(Condition("hotpotqa", "ketrag", 0.5, 42), make_ctx(tmp_path / "a"))
    b = run_condition(Condition("hotpotqa", "ketrag", 0.5, 42), make_ctx(tmp_path / "b"))
    assert a.selected_fingerprint == b.selected_fingerprint and a.config_hash == b.config_hash
    assert a.metrics == b.metrics and a.corpus_id == b.corpus_id
    pa = [r.answer.answer for r in read_predictions(make_ctx(tmp_path / "a").condition_dir(COND.__class__("hotpotqa", "ketrag", 0.5, 42)) / "predictions.jsonl")]
    pb = [r.answer.answer for r in read_predictions(make_ctx(tmp_path / "b").condition_dir(COND.__class__("hotpotqa", "ketrag", 0.5, 42)) / "predictions.jsonl")]
    assert pa == pb


def test_condition_directories_never_collide(tmp_path):
    ctx = make_ctx(tmp_path)
    conds = generate_conditions(MatrixSpec(), 42)
    assert len({ctx.condition_dir(c) for c in conds}) == 48


# =================================================================== caches
def test_extraction_cache_is_shared_across_budgets_of_one_strategy(tmp_path):
    events = []
    ctx = make_ctx(tmp_path, comps=spy_components(events))
    small = run_condition(Condition("hotpotqa", "random", 0.5, 42), ctx)       # 4 chunks
    big = run_condition(Condition("hotpotqa", "random", 1.0, 42), ctx)         # the same 4 + 4 more
    assert small.stages["extraction"].details["n_cache_hits"] == 0
    assert big.stages["extraction"].details["n_cache_hits"] == 4 and big.stages["extraction"].details["n_extractor_calls"] == 4
    assert len(kinds(events, "extract")) == 8                                    # 8 chunks extracted once in total, not 12
    # cache hits keep their original cost in the uncached total (so strategies' costs stay comparable)
    assert big.stages["extraction"].details["cost_if_uncached_usd"] is not None


# =================================================================== the matrix loop
def test_ranking_and_corpus_are_computed_once_and_reused_across_budgets(tmp_path):
    events = []
    spec = MatrixSpec(datasets=["hotpotqa", "musique"], strategies=["random", "ketrag"], budgets=[0.25, 0.5, 1.0])
    conds = generate_conditions(spec, 42)
    assert len(conds) == 12
    ctx = make_ctx(tmp_path, comps=spy_components(events))
    out = run_matrix(conds, ctx)
    assert [r.status for r in out.results] == ["completed"] * 12
    assert len(kinds(events, "rank")) == 4                    # 2 datasets x 2 strategies, NOT 12
    assert len(kinds(events, "load_corpus")) == 2             # one per dataset
    reused = [r.stages["ranking"].details["reused_in_memory"] for r in out.results]
    assert reused.count(False) == 4 and reused.count(True) == 8


def test_a_full_48_condition_mock_matrix_completes_cleanly(tmp_path):
    spec = MatrixSpec()
    conds = generate_conditions(spec, 42)
    ctx = make_ctx(tmp_path)
    write_manifest(ctx, spec, conds)
    out = run_matrix(conds, ctx)
    assert len(out.results) == 48 and {r.status for r in out.results} == {"completed"} and out.stopped is None
    rows = read(ctx.matrix_dir / "results.jsonl")
    assert [r["condition_id"] for r in rows] == [c.condition_id for c in conds]
    assert all(r["n_selected"] == r["n_selected_expected"] for r in rows)
    assert not any(r["reportable"] for r in rows) and all(r["is_mock"] for r in rows)
    manifest = json.loads((ctx.matrix_dir / "matrix.json").read_text())
    assert manifest["n_conditions"] == 48 and len(manifest["conditions"]) == 48 and manifest["spec"]["seeds"] is None
    assert run_matrix(conds, make_ctx(tmp_path)).skipped_completed == [c.condition_id for c in conds]


def test_the_matrix_stops_at_the_first_aborted_condition_and_can_be_resumed(tmp_path):
    conds = [Condition("hotpotqa", "random", b, 42) for b in (0.25, 0.5, 1.0)]
    out = run_matrix(conds, make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: AbortingExtractor(after=0))))
    assert [r.status for r in out.results] == ["aborted"] and out.stopped["condition_id"] == conds[0].condition_id
    assert read_condition_result(make_ctx(tmp_path).condition_dir(conds[1])) is None          # later ones never started
    resumed = run_matrix(conds, make_ctx(tmp_path))
    assert [r.status for r in resumed.results] == ["completed"] * 3


def test_a_failed_condition_does_not_stop_the_matrix(tmp_path):
    ctx = make_ctx(tmp_path, ketrag_mode="faithful", embedding_backend="tfidf")
    conds = [Condition("hotpotqa", "ketrag", 0.5, 42), Condition("hotpotqa", "random", 0.5, 42)]
    out = run_matrix(conds, ctx)
    assert [r.status for r in out.results] == ["failed", "completed"] and out.stopped is None


def test_spending_limit_stops_before_the_next_condition(tmp_path):
    conds = [Condition("hotpotqa", "random", b, 42) for b in (0.5, 1.0)]
    ctx = make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: CostlyExtractor()))
    out = run_matrix(conds, ctx, max_total_usd=0.005)           # the first condition spends 4 x $0.01
    assert len(out.results) == 1 and out.stopped["reason"] == "max_total_usd"
    assert out.results[0].cost["total_cost_spent_usd"] == pytest.approx(0.04)


def test_spending_limit_cannot_be_checked_when_cost_is_unknown(tmp_path):
    conds = [Condition("hotpotqa", "random", b, 42) for b in (0.5, 1.0)]
    ctx = make_ctx(tmp_path, comps=spy_components([], extractor_wrap=lambda e: UnknownUsageExtractor()))
    out = run_matrix(conds, ctx, max_total_usd=100.0)
    assert len(out.results) == 1 and "unknown" in out.stopped["message"]
    free = run_matrix(conds, make_ctx(tmp_path / "x", comps=spy_components([], extractor_wrap=lambda e: UnknownUsageExtractor())))
    assert len(free.results) == 2 and free.stopped is None      # without a limit there is nothing to check


def test_write_results_index_ignores_missing_conditions(tmp_path):
    ctx = make_ctx(tmp_path)
    run_condition(COND, ctx)
    path = write_results_index(ctx.matrix_dir, ["nope__x__b005__seed1", COND.condition_id])
    assert [r["condition_id"] for r in read(path)] == [COND.condition_id]


# =================================================================== planning
def test_plan_lists_conditions_with_counts_and_makes_no_calls_and_writes_nothing(tmp_path):
    events = []
    ctx = make_ctx(tmp_path, comps=spy_components(events))
    plan = plan_matrix(generate_conditions(MatrixSpec(), 42), ctx)
    assert plan["n_conditions"] == 48 and {r["status"] for r in plan["conditions"]} == {"pending"}
    row = next(r for r in plan["conditions"] if r["condition_id"] == "hotpotqa__random__b025__seed42")
    assert (row["n_chunks_total"], row["n_selected"], row["n_questions"]) == (8, 2, 3)
    assert kinds(events, "extract") == [] and kinds(events, "answer") == [] and kinds(events, "rank") == []
    assert not ctx.matrix_dir.exists() and not (tmp_path / "cache").exists()


def test_plan_estimates_real_backend_cost_without_a_key_or_network(tmp_path, no_network, no_api_key):
    ctx = make_ctx(tmp_path, extraction_backend="openai", generation_backend="openai", judge_backend="openai")
    plan = plan_matrix(generate_conditions(MatrixSpec(budgets=[0.5, 1.0], strategies=["random"]), 42), ctx)
    t = plan["totals"]
    assert t["est_extraction_usd_sum_of_conditions"] > 0 and t["est_generation_usd_upper_bound"] > 0 and t["est_judge_usd"] > 0
    assert t["extraction_cache_shared_upper_bound_usd"] <= t["est_extraction_usd_sum_of_conditions"]
    assert not ctx.matrix_dir.exists()


def test_plan_reports_stale_and_completed_conditions(tmp_path):
    ctx = make_ctx(tmp_path)
    run_condition(COND, ctx)
    assert plan_matrix([COND], ctx)["conditions"][0]["status"] == "completed"
    changed = make_ctx(tmp_path, retrieval_top_m=7)
    assert plan_matrix([COND], changed)["conditions"][0]["status"] == "stale"


def test_plan_tolerates_a_dataset_that_is_not_downloaded(tmp_path):
    def missing(cfg):
        raise FileNotFoundError("data/raw/hotpotqa not found: run src.prepare_data first")

    ctx = make_ctx(tmp_path, comps=spy_components([], load_corpus=missing))
    plan = plan_matrix([COND], ctx)
    assert plan["conditions"][0]["n_chunks_total"] is None and plan["notes"] and plan["totals"]["est_extraction_usd_sum_of_conditions"] is None
