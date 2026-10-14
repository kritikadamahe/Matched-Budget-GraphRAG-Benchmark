"""95% bootstrap confidence intervals over questions (paper: "Statistical Analysis")."""
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bootstrap import bootstrap_ci, condition_cis, paired_bootstrap
from evaluation.schemas import ScoredRecord

GOLD = "G"


def rec(question_id, em, f1, judge_correct=None):
    return ScoredRecord(question_id=question_id, gold_answer=GOLD, predicted_answer="x",
                        generation_status="ok", outcome="answered", em=em, f1=f1,
                        judge_correct=judge_correct, judge_status="not_run" if judge_correct is None else "ok",
                        is_mock_generation=False, is_mock_judge=False)


def test_bootstrap_ci_is_reproducible_and_centred_on_the_mean():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a = bootstrap_ci(values, seed=0)
    b = bootstrap_ci(values, seed=0)
    assert a == b                                              # same seed -> identical interval
    assert a["mean"] == sum(values) / len(values)
    assert a["low"] <= a["mean"] <= a["high"]
    assert a["n"] == len(values) and a["n_resamples"] == 1000 and a["level"] == 0.95


def test_bootstrap_ci_shrinks_for_a_constant_series():
    constant = bootstrap_ci([1.0] * 50, seed=0)
    assert constant["low"] == constant["high"] == 1.0


def test_bootstrap_ci_widens_with_more_variance():
    tight = bootstrap_ci([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], seed=0)
    wide = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], seed=0)
    assert (wide["high"] - wide["low"]) > (tight["high"] - tight["low"])


def test_bootstrap_ci_empty_is_none():
    assert bootstrap_ci([]) is None


def test_different_seeds_give_slightly_different_but_close_intervals():
    values = [0.2, 0.8, 1.0, 0.0, 0.6, 0.4, 1.0, 0.0, 0.3, 0.7]
    a, b = bootstrap_ci(values, seed=0), bootstrap_ci(values, seed=1)
    assert a["mean"] == b["mean"]
    assert abs(a["low"] - b["low"]) < 0.3 and abs(a["high"] - b["high"]) < 0.3


# ------------------------------------------------------------------------------ paired
def test_paired_bootstrap_matches_only_shared_questions():
    a = {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 1.0}
    b = {"q2": 0.0, "q3": 0.0, "q4": 0.0, "q5": 1.0}             # q1, q5 have no counterpart
    out = paired_bootstrap(a, b, seed=0)
    assert out["n_questions"] == 3                                # q2, q3, q4 only
    assert out["diff"] == pytest.approx((0.0 - 0.0 + 1.0 - 0.0 + 1.0 - 0.0) / 3)
    assert out["low"] <= out["diff"] <= out["high"]


def test_paired_bootstrap_identical_series_gives_zero_difference_and_high_p():
    a = {f"q{i}": float(i % 2) for i in range(20)}
    out = paired_bootstrap(a, dict(a), seed=0)
    assert out["diff"] == 0.0 and out["low"] == out["high"] == 0.0 and out["p_value"] == 1.0


def test_paired_bootstrap_no_overlap_is_none():
    assert paired_bootstrap({"q1": 1.0}, {"q2": 1.0}) is None


def test_paired_bootstrap_clear_winner_has_small_p_value():
    a = {f"q{i}": 1.0 for i in range(30)}
    b = {f"q{i}": 0.0 for i in range(30)}
    out = paired_bootstrap(a, b, seed=0)
    assert out["diff"] == 1.0 and out["p_value"] < 0.05


# --------------------------------------------------------------------- condition_cis
def test_condition_cis_covers_em_f1_judge():
    scored = [rec("q1", 1.0, 1.0, True), rec("q2", 0.0, 0.3, False), rec("q3", 1.0, 0.8, True)]
    out = condition_cis(scored)
    assert set(out) == {"em", "f1", "judge_accuracy"}
    assert out["em"]["mean"] == pytest.approx(2 / 3) and out["judge_accuracy"]["mean"] == pytest.approx(2 / 3)
    assert out["f1"]["n"] == 3


def test_condition_cis_skips_failed_generations_and_no_judge_calls():
    scored = [rec("q1", 1.0, 1.0), rec("q2", None, None)]        # q2: failed generation, em/f1 None
    out = condition_cis(scored)
    assert out["em"]["n"] == 1 and out["f1"]["n"] == 1
    assert out["judge_accuracy"] is None                          # no judge_correct anywhere
