"""Phase 12, increment 1: the analysis loader and its gate. Matrices come from the real Phase 10 runner
with free doubles; the 'real' ones are SYNTHETIC (see tests/helpers_analysis.py): they test the code,
they are not results. No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import shutil

import pytest

from analysis.loader import AnalysisRefused, load_analysis_input
from analysis.schemas import (METRICS, PRIMARY_METRICS, SECONDARY_METRICS, QuestionObservation, metric_value)
from tests.helpers_analysis import (Q1_TEXT, cond_dir, copy_matrix, edit_json, edit_jsonl, file_hashes,
                                    mock_matrix, priced_components, real_matrix)

CID = "hotpotqa__random__b050__seed42"
KET = "hotpotqa__ketrag__b050__seed42"


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    return real_matrix(tmp_path_factory.mktemp("real"))


@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    return mock_matrix(tmp_path_factory.mktemp("mock"))


@pytest.fixture
def real_copy(real, tmp_path):
    return copy_matrix(real, tmp_path / "m")


def obs(**kw):
    d = dict(condition_id="c", dataset="hotpotqa", strategy="random", budget_pct=10, seed=42, question_id="q1",
             outcome="answered", generation_status="ok", em=1.0, f1=1.0, judge_correct=True, judge_status="ok")
    d.update(kw)
    return QuestionObservation(**d)


# ============================================================================== metric_value
def test_metric_constants():
    assert PRIMARY_METRICS == ("f1", "judge_accuracy") and SECONDARY_METRICS == ("em",)
    assert set(METRICS) == {"f1", "judge_accuracy", "em"}


def test_metric_value_for_a_measured_answer():
    o = obs(em=0.0, f1=0.4, judge_correct=False)
    assert metric_value(o, "em") == (0.0, None)
    assert metric_value(o, "f1") == (0.4, None)
    assert metric_value(o, "judge_accuracy") == (0.0, None)
    assert metric_value(obs(judge_correct=True), "judge_accuracy") == (1.0, None)


def test_an_abstention_is_a_zero_not_a_missing_value():
    o = obs(outcome="abstained", em=0.0, f1=0.0, judge_correct=False, judge_status="skipped_abstained")
    assert [metric_value(o, m) for m in ("em", "f1", "judge_accuracy")] == [(0.0, None)] * 3


def test_a_failed_generation_is_missing_for_every_metric():
    o = obs(outcome="failed", generation_status="api_error", em=None, f1=None, judge_correct=None,
            judge_status="skipped_failed_generation")
    assert [metric_value(o, m) for m in ("em", "f1", "judge_accuracy")] == [(None, "generation_failed")] * 3


def test_a_question_with_no_prediction_is_missing_for_every_metric():
    o = obs(outcome="no_prediction", generation_status=None, em=None, f1=None, judge_correct=None, judge_status=None)
    assert [metric_value(o, m) for m in ("em", "f1", "judge_accuracy")] == [(None, "no_prediction")] * 3


@pytest.mark.parametrize("status,reason", [("api_error", "judge_failed"), ("malformed", "judge_failed"),
                                           ("refused", "judge_failed"), ("truncated", "judge_failed"),
                                           ("disabled", "judge_disabled"), ("not_run", "judge_not_run")])
def test_judge_problems_are_missing_only_for_the_judge_metric(status, reason):
    o = obs(judge_correct=None, judge_status=status)
    assert metric_value(o, "judge_accuracy") == (None, reason)
    assert metric_value(o, "f1") == (1.0, None)                      # the answer itself was fine


def test_unknown_metric_is_an_error():
    with pytest.raises(ValueError, match="unknown metric"):
        metric_value(obs(), "bleu")


def test_observations_hold_no_gold_fields():
    assert not [f for f in QuestionObservation.model_fields if "gold" in f]


# ============================================================================== a clean synthetic matrix
def test_a_clean_non_mock_matrix_loads(real):
    a = load_analysis_input(real)
    assert a.data_kind == "real" and a.status == "real_complete" and not a.allowed_incomplete
    assert len(a.conditions) == 4                                   # random + ketrag x 25% + 50%
    assert a.question_seed == {"hotpotqa": 42}
    assert len(a.question_ids["hotpotqa"]) == 3
    assert len(a.observations) == 4 * 3
    assert a.judge_enabled is True
    assert a.excluded == []
    assert all(c.selected_chunk_ids is not None and c.n_no_prediction == 0 for c in a.conditions)
    assert any("cannot prove" in c for c in a.caveats)               # the honest limit of the gate is stated


def test_observations_carry_scores_and_retrieved_chunks(real):
    a = load_analysis_input(real)
    o = next(x for x in a.observations if x.condition_id == KET and x.question_id == "q1")
    assert o.outcome in ("answered", "abstained") and o.em is not None and o.f1 is not None
    assert isinstance(o.retrieved_chunk_ids, tuple)
    sel = set(next(c for c in a.conditions if c.condition_id == KET).selected_chunk_ids)
    assert set(o.retrieved_chunk_ids) <= sel                         # retrieval can only return selected chunks


def test_selected_ids_match_the_runner_count(real):
    a = load_analysis_input(real)
    by_id = {r.condition_id: r for r in a.records}
    for c in a.conditions:
        assert len(c.selected_chunk_ids) == by_id[c.condition_id].counts["n_selected"]


def test_loading_is_read_only_and_deterministic(real_copy):
    before = file_hashes(real_copy)
    a1, a2 = load_analysis_input(real_copy), load_analysis_input(real_copy)
    assert file_hashes(real_copy) == before
    assert a1.observations == a2.observations and a1.source_files == a2.source_files


def test_source_files_are_hashed(real):
    a = load_analysis_input(real)
    paths = {s.path for s in a.source_files}
    assert "matrix.json" in paths and f"conditions/{CID}/scored.jsonl" in paths
    assert f"conditions/{CID}/extraction/selection.json" in paths
    for s in a.source_files:
        assert hashlib.sha256((Path(real) / s.path).read_bytes()).hexdigest() == s.sha256


def test_random_shuffles_load_as_separate_conditions_on_one_question_set(tmp_path):
    m = real_matrix(tmp_path, strategies=("random", "ketrag"), budgets=(0.5,), random_seeds=[0, 1, 2])
    a = load_analysis_input(m)
    shuffles = sorted(c.shuffle_seed for c in a.conditions if c.strategy == "random")
    assert shuffles == [0, 1, 2]
    assert len(a.conditions) == 4 and len(a.observations) == 4 * 3          # rows, not extra questions
    assert len({tuple(c.question_ids) for c in a.conditions}) == 1          # the same questions for everyone
    assert all(o.shuffle_seed is None for o in a.observations if o.strategy == "ketrag")


def test_a_question_without_a_prediction_stays_in_the_universe(tmp_path):
    m = real_matrix(tmp_path, comps=priced_components(fail_retrieval_on=Q1_TEXT))
    a = load_analysis_input(m)
    rows = [o for o in a.observations if o.question_id == "q1"]
    assert len(rows) == 4 and all(o.outcome == "no_prediction" for o in rows)
    assert all(c.n_no_prediction == 1 and c.n_questions == 3 for c in a.conditions)
    assert len(a.observations) == 4 * 3                                      # still 3 questions per condition
    assert metric_value(rows[0], "f1") == (None, "no_prediction")
    assert a.status == "real_complete"                                       # a measured failure, not an incomplete run


def test_a_disabled_judge_is_reported(tmp_path):
    a = load_analysis_input(real_matrix(tmp_path, no_judge=True))
    assert a.judge_enabled is False and any("judge was disabled" in c for c in a.caveats)
    o = a.observations[0]
    assert metric_value(o, "judge_accuracy")[1] == "judge_disabled" and metric_value(o, "f1")[0] is not None


# ============================================================================== the mock gate
def test_mock_data_is_refused_by_default(mock):
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(mock)
    assert e.value.kind == "mock"
    assert "never be reported as a finding" in str(e.value) and "--allow-mock-layout-test" in str(e.value)


def test_mock_data_is_accepted_only_as_a_layout_test(mock):
    a = load_analysis_input(mock, allow_mock_layout_test=True)
    assert a.data_kind == "mock" and a.status == "mock_layout_test"
    assert any("not research results" in c for c in a.caveats)
    assert len(a.observations) == 4 * 3


def test_the_mock_flag_never_lets_real_data_look_mock(real):
    a = load_analysis_input(real, allow_mock_layout_test=True)
    assert a.status == "real_complete" and a.data_kind == "real"


def test_one_mock_condition_among_real_ones_is_always_refused(real_copy):
    edit_json(cond_dir(real_copy, CID) / "condition_result.json", lambda d: d.update(is_mock=True))
    for flag in (False, True):
        with pytest.raises(AnalysisRefused) as e:
            load_analysis_input(real_copy, allow_mock_layout_test=flag)
        assert e.value.kind == "mixed" and "never combined" in str(e.value)


def test_a_mock_data_source_in_one_condition_is_mixed(real_copy):
    def f(d):
        d["data_source"] = "mock"
        d["config"]["data_source"] = "mock"
    edit_json(cond_dir(real_copy, CID) / "condition_result.json", f)
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy, allow_mock_layout_test=True)
    assert e.value.kind in ("mixed", "input")                                # the config also differs: either refusal is right


def test_mock_answers_are_caught_even_when_the_flags_were_doctored(real_copy):
    """is_mock is a recorded flag; the per-item backend labels are checked as well."""
    def f(rows):
        for r in rows:
            r["answer"]["backend"] = "mock"
    edit_jsonl(cond_dir(real_copy, CID) / "predictions.jsonl", f)
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy, allow_mock_layout_test=True)
    assert e.value.kind == "mixed"


def test_a_fully_doctored_mock_matrix_is_still_mock(mock, tmp_path):
    """Every condition edited to claim real data, real backends and reportable=true: the per-item mock
    labels (mock answers, mock judge) still expose it."""
    m = copy_matrix(mock, tmp_path / "doctored")
    for p in (m / "conditions").iterdir():
        def f(d):
            d.update(is_mock=False, reportable=True, not_reportable_reasons=[], data_source="huggingface")
            d["config"].update(data_source="huggingface", extraction_backend="openai", generation_backend="openai",
                               judge_backend="openai")
        edit_json(p / "condition_result.json", f)
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(m)
    assert e.value.kind == "mock"
    a = load_analysis_input(m, allow_mock_layout_test=True)
    assert a.status == "mock_layout_test"


# ============================================================================== reportability / completeness
def test_a_missing_condition_refuses_unless_allowed(real_copy):
    shutil.rmtree(cond_dir(real_copy, CID))
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert e.value.kind == "not_reportable" and "matrix incomplete" in str(e.value)
    assert "--allow-incomplete" in str(e.value)
    a = load_analysis_input(real_copy, allow_incomplete=True)
    assert a.status == "real_partial" and a.allowed_incomplete
    assert len(a.conditions) == 3 and any(c.startswith("PARTIAL RUN") for c in a.caveats)
    assert [e["kind"] for e in a.excluded] == ["missing_result"] and a.excluded[0]["condition_id"] == CID


def test_an_aborted_condition_is_listed_and_makes_the_run_partial(real_copy):
    edit_json(cond_dir(real_copy, CID) / "condition_result.json", lambda d: d.update(status="aborted"))
    with pytest.raises(AnalysisRefused):
        load_analysis_input(real_copy)
    a = load_analysis_input(real_copy, allow_incomplete=True)
    assert a.status == "real_partial" and CID not in {c.condition_id for c in a.conditions}
    assert any(e["condition_id"] == CID and e["kind"] == "aborted" for e in a.excluded)
    assert CID not in {o.condition_id for o in a.observations}


def test_without_matrix_json_completeness_cannot_be_proven(real_copy):
    (real_copy / "matrix.json").unlink()
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert "planned conditions are unknown" in str(e.value)
    assert load_analysis_input(real_copy, allow_incomplete=True).status == "real_partial"


def test_allow_incomplete_does_not_waive_other_problems(real_copy):
    edit_json(cond_dir(real_copy, CID) / "condition_result.json",
              lambda d: d.update(reportable=False, not_reportable_reasons=["something else is wrong"]))
    shutil.rmtree(cond_dir(real_copy, KET))
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy, allow_incomplete=True)
    assert e.value.kind == "not_reportable" and "something else is wrong" in str(e.value)


def test_nothing_to_analyse_is_refused_even_when_incomplete_is_allowed(real_copy):
    for p in (real_copy / "conditions").iterdir():
        edit_json(p / "condition_result.json", lambda d: d.update(status="failed"))
    with pytest.raises(AnalysisRefused):
        load_analysis_input(real_copy, allow_incomplete=True)


# ============================================================================== input integrity
def test_not_a_matrix_directory(tmp_path):
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(tmp_path)
    assert e.value.kind == "input" and "not a matrix directory" in str(e.value)


def test_two_results_for_one_cell_are_refused(real_copy):
    shutil.copytree(cond_dir(real_copy, CID), real_copy / "conditions" / "dup_of_random_b050")
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert e.value.kind == "input" and "more than one result" in str(e.value)


def test_a_missing_scored_file_is_refused(real_copy):
    (cond_dir(real_copy, CID) / "scored.jsonl").unlink()
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert CID in str(e.value) and "scored.jsonl" in str(e.value)


def test_scored_and_predictions_must_agree(real_copy):
    edit_jsonl(cond_dir(real_copy, CID) / "scored.jsonl", lambda rows: rows.pop())
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert e.value.kind == "input" and "disagree on the questions" in str(e.value)


def test_a_duplicated_question_is_refused(real_copy):
    edit_jsonl(cond_dir(real_copy, CID) / "scored.jsonl", lambda rows: rows.append(dict(rows[0])))
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert "more than once" in str(e.value)


def test_a_record_from_another_strategy_is_refused(real_copy):
    def f(rows):
        rows[0]["strategy"] = "ketrag"
    edit_jsonl(cond_dir(real_copy, CID) / "scored.jsonl", f)
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert "strategy='ketrag'" in str(e.value)


def test_a_wrong_budget_in_a_record_is_refused(real_copy):
    edit_jsonl(cond_dir(real_copy, CID) / "scored.jsonl", lambda rows: rows[0].update(budget=0.1))
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert "budget=0.1" in str(e.value)


def test_the_selection_file_must_match_the_runner_count(real_copy):
    edit_json(cond_dir(real_copy, CID) / "extraction" / "selection.json",
              lambda d: d.update(selected_chunk_ids=d["selected_chunk_ids"][:-1]))
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert "n_selected" in str(e.value)


def test_a_missing_selection_file_is_a_warning_not_a_refusal(real_copy):
    (cond_dir(real_copy, CID) / "extraction" / "selection.json").unlink()
    a = load_analysis_input(real_copy)
    c = next(c for c in a.conditions if c.condition_id == CID)
    assert c.selected_chunk_ids is None
    assert any(i.code == "no_selection_file" and i.condition_id == CID for i in a.issues)


# ============================================================================== alignment
def test_conditions_must_share_one_question_set(real_copy):
    """Drop q3 from ONE condition consistently (files and the runner's counts): the question sets differ."""
    d = cond_dir(real_copy, CID)
    for name in ("scored.jsonl", "predictions.jsonl"):
        edit_jsonl(d / name, lambda rows: rows.__setitem__(slice(None), [r for r in rows if r["question_id"] != "q3"]))
    edit_json(d / "condition_result.json", lambda r: r.update(n_questions=2, n_predictions=2))
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert e.value.kind == "input" and "do not share one question set" in str(e.value)


def test_a_count_that_disagrees_with_the_files_is_refused(real_copy):
    edit_json(cond_dir(real_copy, CID) / "condition_result.json", lambda r: r.update(n_questions=99))
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(real_copy)
    assert "n_questions=99" in str(e.value)


def test_several_question_seeds_are_refused(tmp_path):
    m = real_matrix(tmp_path, strategies=("random",), budgets=(0.5,), seeds=[42, 43])
    with pytest.raises(AnalysisRefused) as e:
        load_analysis_input(m)
    assert "different question seeds" in str(e.value)


def test_gold_answers_are_not_copied_into_the_loaded_input(real):
    a = load_analysis_input(real)
    blob = json.dumps([o.model_dump(mode="json") for o in a.observations])
    gold = {json.loads(l)["gold_answer"] for l in (Path(real) / "conditions" / CID / "scored.jsonl").read_text().splitlines()}
    assert gold and not any(g in blob for g in gold if len(g) > 3)
