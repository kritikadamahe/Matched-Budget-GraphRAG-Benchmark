"""Phase 11: per-condition cost / token / runtime measures, missing vs zero, failures, reconciliation.
Matrices come from the real Phase 10 runner with priced non-mock doubles (exact expected totals);
edge cases are made by editing copies of the files. No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pytest

from aggregation.aggregate import aggregate_matrices
from src.corpus import build_chunk_manifest
from tests.helpers_aggregation import (CALL_COST, EXT_COST, EXT_IN, EXT_OUT, EXT_RUNTIME, GEN_RUNTIME, JUDGE_RUNTIME,
                                       FAKE_IN, FAKE_OUT, AbortingPricedExtractor, PartlyUnknownExtractor,
                                       build_matrix, cond_dir, copy_matrix, edit_json, edit_jsonl, priced_components)

CID = "hotpotqa__random__b050__seed42"          # 4 selected chunks, 3 questions; 2 of its extractions are cache hits
N, Q = 4, 3


def records_of(matrix_dir, **kw):
    report, records = aggregate_matrices([matrix_dir], **kw)
    return report, {r.condition_id: r for r in records}


def codes(rec):
    return [i.code for i in rec.issues]


@pytest.fixture(scope="module")
def priced(tmp_path_factory):
    return build_matrix(tmp_path_factory.mktemp("priced"))


@pytest.fixture()
def mutable(priced, tmp_path):
    """A private copy of the priced matrix to damage."""
    return copy_matrix(priced, tmp_path / "m")


# =========================================================================== exact figures
def test_costs_are_exact_and_each_stage_is_separate(priced):
    _, recs = records_of(priced)
    m = recs[CID].measures
    gen = round(Q * CALL_COST, 6)
    assert m["cost_extraction_usd"].status == "measured" and m["cost_extraction_usd"].value == pytest.approx(N * EXT_COST)
    assert m["cost_generation_usd"].value == pytest.approx(gen) and m["cost_judge_usd"].value == pytest.approx(gen)
    assert m["cost_total_usd"].status == "measured"
    assert m["cost_total_usd"].value == pytest.approx(N * EXT_COST + 2 * gen)
    assert m["cost_total_usd"].components == ["extraction", "generation", "judge"]


def test_tokens_are_rebuilt_from_the_per_item_files(priced):
    _, recs = records_of(priced)
    m = recs[CID].measures
    assert (m["tokens_extraction_input"].value, m["tokens_extraction_output"].value) == (N * EXT_IN, N * EXT_OUT)
    assert (m["tokens_generation_input"].value, m["tokens_generation_output"].value) == (Q * FAKE_IN, Q * FAKE_OUT)
    assert (m["tokens_judge_input"].value, m["tokens_judge_output"].value) == (Q * FAKE_IN, Q * FAKE_OUT)
    assert m["tokens_total_input"].value == N * EXT_IN + 2 * Q * FAKE_IN
    assert m["tokens_total_output"].value == N * EXT_OUT + 2 * Q * FAKE_OUT
    assert all(isinstance(m[k].value, int) for k in ("tokens_extraction_input", "tokens_total_output"))


def test_intrinsic_runtime_is_the_sum_of_call_runtimes_plus_the_recomputed_stages(priced):
    _, recs = records_of(priced)
    m = recs[CID].measures
    assert m["runtime_extraction_s"].value == pytest.approx(N * EXT_RUNTIME)
    assert m["runtime_generation_s"].value == pytest.approx(Q * GEN_RUNTIME)
    assert m["runtime_judge_s"].value == pytest.approx(Q * JUDGE_RUNTIME)
    parts = [m[f"runtime_{k}_s"].value for k in ("ranking", "selection", "extraction", "graph", "retrieval", "generation", "judge")]
    assert m["runtime_total_s"].status == "measured" and m["runtime_total_s"].value == pytest.approx(math.fsum(parts), abs=1e-8)
    assert m["runtime_total_s"].value >= N * EXT_RUNTIME + Q * (GEN_RUNTIME + JUDGE_RUNTIME)
    assert "excludes corpus loading" in m["runtime_total_s"].source


def test_figures_that_depend_on_the_cache_are_labelled_and_kept_apart(priced):
    _, recs = records_of(priced)
    m = recs[CID].measures
    dependent = {k for k, v in m.items() if v.cache_dependent}
    assert dependent == {k for k in m if k.startswith(("spent_", "elapsed_")) or k == "wall_seconds"}
    assert not any(m[k].cache_dependent for k in m if k.startswith(("cost_", "tokens_", "runtime_")))
    # the comparable cost is the uncached one; what this run paid is smaller because 2 of 4 chunks were cached
    assert recs[CID].counts["extraction_n_cache_hits"] == 2
    assert m["cost_extraction_usd"].value == pytest.approx(N * EXT_COST)
    assert m["spent_extraction_usd"].value == pytest.approx((N - 2) * EXT_COST)
    assert m["spent_extraction_usd"].value < m["cost_extraction_usd"].value
    assert m["tokens_extraction_input"].value == N * EXT_IN                      # tokens ignore the cache
    assert m["runtime_extraction_s"].value == pytest.approx(N * EXT_RUNTIME)     # so does the call runtime (original kept)
    assert m["elapsed_extraction_s"].value < N * EXT_RUNTIME                     # this run's wall clock does not


def test_counts_failures_metrics_and_configuration_are_preserved(priced):
    _, recs = records_of(priced)
    r = recs[CID]
    raw = __import__("json").loads((cond_dir(priced, CID) / "condition_result.json").read_text())
    assert (r.dataset, r.strategy, r.budget, r.budget_pct, r.seed) == ("hotpotqa", "random", 0.5, 50, 42)
    assert r.config == raw["config"] and r.metrics == raw["metrics"] and r.run_id == raw["run_id"]
    assert r.config_hash == raw["config_hash"] and r.corpus_id == raw["corpus_id"] and r.shared_config_fingerprint
    assert r.counts["n_selected"] == N and r.counts["n_questions"] == Q and r.counts["n_predictions"] == Q
    assert r.counts["judge_n_calls"] == Q and r.counts["n_failed_outcomes"] == 0
    assert r.failures["extraction_status_counts"] == {"ok": N} and r.failures["generation_status_counts"] == {"ok": Q}
    assert r.failures["question_failures"] == [] and r.reconciled and r.issues == [] and r.judge_enabled is True


def test_a_completed_real_condition_is_reportable(priced):
    report, recs = records_of(priced)
    assert recs[CID].reportable and not recs[CID].is_mock and report.reportable


# =========================================================================== zero vs unknown
def test_a_measured_zero_is_a_value_not_a_missing_measurement(tmp_path):
    m = build_matrix(tmp_path, mock=True, strategies=("random",), budgets=(0.5,))
    _, recs = records_of(m)
    meas = next(iter(recs.values())).measures
    for k in ("cost_extraction_usd", "cost_generation_usd", "cost_judge_usd", "cost_total_usd"):
        assert meas[k].status == "measured" and meas[k].value == 0.0 and meas[k].value is not None


def test_unknown_token_usage_is_a_lower_bound_never_zero_and_never_a_complete_total(tmp_path):
    m = build_matrix(tmp_path, strategies=("random",), budgets=(0.5,),
                     comps=priced_components(extractor=PartlyUnknownExtractor))        # 1 of 4 calls has unknown usage
    _, recs = records_of(m)
    r = recs[CID]
    c = r.measures["cost_extraction_usd"]
    assert c.status == "lower_bound" and c.value is None and c.lower_bound == pytest.approx(3 * EXT_COST)
    t = r.measures["tokens_extraction_input"]
    assert t.status == "lower_bound" and t.value is None and t.lower_bound == 3 * EXT_IN and "unknown token usage" in t.reason
    total = r.measures["cost_total_usd"]
    gen = round(Q * CALL_COST, 6)
    assert total.status == "lower_bound" and total.value is None and "extraction: lower_bound" in total.reason
    assert total.lower_bound == pytest.approx(3 * EXT_COST + 2 * gen)
    assert r.measures["tokens_total_input"].status == "lower_bound" and r.measures["tokens_total_input"].value is None
    assert r.measures["cost_extraction_per_selected_chunk_usd"].status == "lower_bound"
    assert r.measures["cost_extraction_per_selected_chunk_usd"].lower_bound == pytest.approx(3 * EXT_COST / N)
    assert r.measures["runtime_extraction_s"].status == "measured"                      # the runtime itself is known
    assert r.status == "completed" and any("unknown" in w for w in r.warnings) and r.reconciled


def test_a_judge_that_was_not_run_is_disabled_not_zero_and_does_not_block_the_total(tmp_path):
    m = build_matrix(tmp_path, strategies=("random",), budgets=(0.5,), no_judge=True)
    _, recs = records_of(m)
    r = recs[CID]
    assert r.judge_enabled is False
    for k in ("cost_judge_usd", "tokens_judge_input", "tokens_judge_output", "runtime_judge_s",
              "cost_judge_per_question_usd", "runtime_judge_per_question_s"):
        assert r.measures[k].status == "disabled" and r.measures[k].value is None, k
    total = r.measures["cost_total_usd"]
    assert total.status == "measured" and total.components == ["extraction", "generation"]
    assert total.value == pytest.approx(N * EXT_COST + round(Q * CALL_COST, 6))
    assert r.measures["runtime_total_s"].components == ["ranking", "selection", "extraction", "graph", "retrieval", "generation"]


def test_a_total_is_never_complete_when_an_enabled_stage_is_unaccounted_for(priced, mutable):
    def drop_judge(doc):
        del doc["cost"]["judge"]
    edit_json(cond_dir(mutable, CID) / "condition_result.json", drop_judge)
    _, recs = records_of(mutable)
    r = recs[CID]
    assert r.measures["cost_judge_usd"].status == "unknown"                           # enabled, but no cost recorded
    assert r.measures["cost_total_usd"].status == "lower_bound" and r.measures["cost_total_usd"].value is None
    assert "judge: unknown" in r.measures["cost_total_usd"].reason


# =========================================================================== failed / aborted / incomplete
def test_an_aborted_condition_keeps_its_partial_figures_as_lower_bounds_and_never_as_zero(tmp_path):
    m = build_matrix(tmp_path, strategies=("random",), budgets=(0.5,),
                     comps=priced_components(extractor=AbortingPricedExtractor))     # 1 chunk extracted, then quota error
    report, recs = records_of(m)
    r = recs[CID]
    raw = __import__("json").loads((cond_dir(m, CID) / "condition_result.json").read_text())
    assert r.status == "aborted" and not r.included and r.exclusion_reason.startswith("aborted:") and "quota" in r.exclusion_reason
    assert raw["stages"]["graph"]["status"] == "not_run" and raw["stages"]["graph"]["elapsed_seconds"] == 0.0   # the default
    for stage in ("graph", "retrieval", "generation", "evaluation"):
        e = r.measures[f"elapsed_{stage}_s"]
        assert e.status == "not_run" and e.value is None and e.lower_bound is None, stage            # NOT 0.0
    assert r.measures["elapsed_extraction_s"].status == "lower_bound" and "aborted" in r.measures["elapsed_extraction_s"].reason
    c = r.measures["cost_extraction_usd"]
    assert c.status == "lower_bound" and c.value is None and c.lower_bound == pytest.approx(EXT_COST)   # 1 chunk was paid for
    assert r.measures["cost_generation_usd"].status == "not_run" and r.measures["cost_judge_usd"].status == "not_run"
    total = r.measures["cost_total_usd"]
    assert total.status == "lower_bound" and total.value is None and total.lower_bound == pytest.approx(EXT_COST)
    assert "generation: not_run" in total.reason
    assert r.measures["runtime_total_s"].status != "measured" and r.measures["runtime_total_s"].value is None
    assert r.measures["tokens_generation_input"].status == "not_run"
    assert r.measures["elapsed_unattributed_s"].status == "not_applicable"
    assert r.measures["wall_seconds"].status == "measured"                           # the wall clock always ran
    assert r.metrics == {} and r.measures["cost_total_per_question_usd"].status != "measured"
    assert [e.kind for e in report.excluded] == ["aborted"] and report.n_included == 0 and not report.reportable


def test_a_condition_that_failed_before_any_stage_has_no_numbers_only_reasons(tmp_path):
    m = build_matrix(tmp_path, strategies=("ketrag",), budgets=(0.5,), ketrag_mode="faithful")   # invalid for ketrag (tfidf)
    report, recs = records_of(m)
    r = recs["hotpotqa__ketrag__b050__seed42"]
    assert r.status == "failed" and not r.included and r.config == {} and r.shared_config_fingerprint is None
    assert "invalid configuration" in r.exclusion_reason
    assert all(v.value is None and v.lower_bound is None for v in r.measures.values() if v.kind != "elapsed" or v.status != "measured")
    assert r.measures["cost_total_usd"].status == "not_run" and r.measures["runtime_total_s"].status != "measured"
    assert [e.kind for e in report.excluded] == ["failed"]


def test_question_failures_are_visible_and_stop_per_question_figures_from_mixing_coverage(tmp_path):
    victim = build_chunk_manifest("hotpotqa", "mock", 3, 42, 250, 40)[1][1]["question"]
    m = build_matrix(tmp_path, strategies=("random",), budgets=(0.5,), comps=priced_components(fail_retrieval_on=victim))
    _, recs = records_of(m)
    r = recs[CID]
    assert r.status == "completed_with_failures" and r.included                       # a measured outcome, included
    assert r.counts["n_questions"] == 3 and r.counts["n_predictions"] == 2 and r.counts["n_question_failures"] == 1
    assert r.failures["question_failures"][0]["stage"] == "retrieval"
    for k in ("cost_total_per_question_usd", "cost_generation_per_question_usd", "runtime_total_per_question_s"):
        assert r.measures[k].status == "not_applicable" and "only 2 of 3 questions" in r.measures[k].reason, k
    assert r.measures["cost_extraction_per_selected_chunk_usd"].status == "measured"      # unaffected: per chunk
    assert r.measures["tokens_generation_input"].value == 2 * FAKE_IN                      # 2 generations really happened


# =========================================================================== normalised columns
def test_normalised_columns_divide_by_denominators_that_exist(priced):
    _, recs = records_of(priced)
    m = recs[CID].measures
    total = m["cost_total_usd"].value
    assert m["cost_extraction_per_selected_chunk_usd"].value == pytest.approx(N * EXT_COST / N)
    assert m["runtime_extraction_per_selected_chunk_s"].value == pytest.approx(EXT_RUNTIME)
    assert m["cost_total_per_question_usd"].value == pytest.approx(total / Q)
    assert m["runtime_generation_per_question_s"].value == pytest.approx(GEN_RUNTIME)
    assert m["cost_total_per_question_usd"].source == "cost_total_usd / n_questions"


def test_a_zero_denominator_is_not_applicable_not_a_division(mutable):
    def zero_selected(doc):
        doc["n_selected"] = 0
    edit_json(cond_dir(mutable, CID) / "condition_result.json", zero_selected)
    _, recs = records_of(mutable)
    m = recs[CID].measures["cost_extraction_per_selected_chunk_usd"]
    assert m.status == "not_applicable" and m.value is None and "n_selected is 0" in m.reason


# =========================================================================== reconciliation: visible, never corrected
def test_token_mismatch_is_an_issue_and_the_recorded_total_is_not_overwritten(mutable):
    edit_jsonl(cond_dir(mutable, CID) / "predictions.jsonl", lambda rows: rows[0]["answer"]["usage"].update(input_tokens=FAKE_IN + 7))
    _, recs = records_of(mutable)
    r = recs[CID]
    issue = next(i for i in r.issues if i.code == "generation_tokens")
    assert issue.severity == "error" and issue.expected == Q * FAKE_IN and issue.actual == Q * FAKE_IN + 7
    assert r.reconciled is False and r.measures["tokens_generation_input"].value == Q * FAKE_IN + 7    # reported as found, flagged
    assert r.measures["cost_generation_usd"].value == pytest.approx(round(Q * CALL_COST, 6))          # recorded cost untouched


def test_cost_mismatch_is_an_issue_and_the_recorded_cost_is_kept(mutable):
    def tamper(doc):
        doc["cost"]["generation"]["cost_if_uncached_usd"] = 0.5
    edit_json(cond_dir(mutable, CID) / "condition_result.json", tamper)
    _, recs = records_of(mutable)
    r = recs[CID]
    issue = next(i for i in r.issues if i.code == "generation_cost")
    assert issue.expected == 0.5 and issue.actual == pytest.approx(round(Q * CALL_COST, 6)) and not r.reconciled
    assert r.measures["cost_generation_usd"].value == 0.5                                  # as recorded, NOT corrected


def test_a_missing_extraction_row_is_an_error_and_the_token_total_becomes_a_lower_bound(mutable):
    edit_jsonl(cond_dir(mutable, CID) / "extraction" / "extractions.jsonl", lambda rows: rows.pop())
    _, recs = records_of(mutable)
    r = recs[CID]
    assert "extraction_row_count" in codes(r) and not r.reconciled
    t = r.measures["tokens_extraction_input"]
    assert t.status == "lower_bound" and t.value is None and t.lower_bound == (N - 1) * EXT_IN
    assert f"{N - 1} item(s) found, {N} expected" in t.reason
    assert r.measures["runtime_extraction_s"].status == "lower_bound"


def test_cache_hit_and_token_this_run_mismatches_are_detected(mutable):
    def flip(rows):
        hit = next(x for x in rows if x["cache_hit"])
        hit["cache_hit"] = False
    edit_jsonl(cond_dir(mutable, CID) / "extraction" / "extractions.jsonl", flip)
    _, recs = records_of(mutable)
    assert {"extraction_cache_hits", "extraction_tokens_this_run"} <= set(codes(recs[CID]))


def test_a_missing_artifact_makes_the_figure_unknown_and_is_an_error(mutable):
    (cond_dir(mutable, CID) / "predictions.jsonl").unlink()
    _, recs = records_of(mutable)
    r = recs[CID]
    assert "artifact_missing" in codes(r) and not r.reconciled
    for k in ("tokens_generation_input", "tokens_generation_output", "runtime_generation_s"):
        assert r.measures[k].status == "unknown" and r.measures[k].value is None and "missing" in r.measures[k].reason, k
    assert r.measures["cost_generation_usd"].status == "measured"                  # the recorded cost is still there
    assert r.measures["runtime_total_s"].status != "measured" and r.counts["generation_n_failed"] is None


def test_an_unreadable_artifact_is_an_error_not_a_crash(mutable):
    (cond_dir(mutable, CID) / "scored.jsonl").write_text("{ not json\n")
    _, recs = records_of(mutable)
    r = recs[CID]
    assert "artifact_unreadable" in codes(r)
    assert r.measures["tokens_judge_input"].status == "unknown" and "unreadable" in r.measures["tokens_judge_input"].reason


def test_a_zero_runtime_on_a_real_call_is_a_warning_that_leaves_the_value_alone(mutable):
    edit_jsonl(cond_dir(mutable, CID) / "extraction" / "extractions.jsonl", lambda rows: rows[0].update(runtime_seconds=0.0))
    _, recs = records_of(mutable)
    r = recs[CID]
    w = next(i for i in r.issues if i.code == "zero_runtime_calls")
    assert w.severity == "warning" and w.actual == 1
    assert r.reconciled is True                                                    # a warning does not flip reconciliation
    assert r.measures["runtime_extraction_s"].status == "measured" and r.measures["runtime_extraction_s"].value == pytest.approx((N - 1) * EXT_RUNTIME)


def test_the_budget_count_is_cross_checked(mutable):
    edit_json(cond_dir(mutable, CID) / "condition_result.json", lambda d: d.update(n_selected_expected=3))
    _, recs = records_of(mutable)
    i = next(i for i in recs[CID].issues if i.code == "budget_mismatch")
    assert (i.expected, i.actual) == (3, N)


def test_judge_call_count_needs_the_evaluation_summary(mutable):
    def drop_n(doc):
        del doc["cost"]["judge"]["n"]
    edit_json(cond_dir(mutable, CID) / "eval_summary.json", drop_n)
    _, recs = records_of(mutable)
    t = recs[CID].measures["tokens_judge_input"]
    assert t.status == "lower_bound" and "expected number of judge calls" in t.reason and t.value is None


def test_ranking_time_comes_from_the_recorded_original_not_the_reused_stage(mutable):
    edit_json(cond_dir(mutable, CID) / "condition_result.json",
              lambda d: d["stages"]["ranking"]["details"].update(ranking_seconds=1.25))
    _, recs = records_of(mutable)
    m = recs[CID].measures
    assert m["runtime_ranking_s"].value == 1.25 and not m["runtime_ranking_s"].cache_dependent
    assert m["elapsed_ranking_s"].value != 1.25 and m["elapsed_ranking_s"].cache_dependent


def test_unattributed_wall_time_is_wall_minus_stage_sum(priced):
    _, recs = records_of(priced)
    m = recs[CID].measures
    stages = math.fsum(m[f"elapsed_{s}_s"].value for s in ("corpus", "ranking", "selection", "extraction", "graph", "retrieval", "generation", "evaluation"))
    assert m["elapsed_unattributed_s"].value == pytest.approx(m["wall_seconds"].value - stages, abs=1e-8)
