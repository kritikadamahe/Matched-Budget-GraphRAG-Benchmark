"""Phase 12, increment 2: per-question arms, shuffle averaging and pairing (analysis/random_avg.py).
Hand-built observations, plus one check on output of the real Phase 10 runner (synthetic 'real-looking'
matrix, see tests/helpers_analysis.py). No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from analysis.random_avg import (INCOMPLETE_SHUFFLES, NO_OBSERVATION, ArmError, build_arm, pair_arms,
                                 shuffle_summary)
from analysis.schemas import QuestionObservation

U = ("q1", "q2", "q3", "q4")


def ob(strategy, q, f1, shuffle=None, *, budget=10, dataset="hotpotqa", em=None, judge=None, outcome=None,
       judge_status=None):
    """One observation. f1=None means a failed generation (outcome 'failed') unless `outcome` is given."""
    if outcome is None:
        outcome = "failed" if f1 is None else "answered"
    if judge_status is None:
        judge_status = "skipped_failed_generation" if outcome == "failed" else "ok"
    return QuestionObservation(
        condition_id=f"{dataset}__{strategy}__b{budget:03d}" + ("" if shuffle is None else f"__rs{shuffle}"),
        dataset=dataset, strategy=strategy, budget_pct=budget, seed=42, shuffle_seed=shuffle, question_id=q,
        outcome=outcome, generation_status="ok" if outcome != "failed" else "api_error",
        em=em if em is not None else f1, f1=f1, judge_correct=judge, judge_status=judge_status)


def random_obs(table, **kw):
    """table: {shuffle_seed: [f1 for q1..q4]} -> observations of Random."""
    return [ob("random", q, v, s, **kw) for s, vals in table.items() for q, v in zip(U, vals)]


def arm(observations, strategy="random", metric="f1", **kw):
    return build_arm(observations, dataset="hotpotqa", strategy=strategy, budget_pct=10, metric=metric, universe=U, **kw)


# ============================================================================== one value per question
def test_each_question_contributes_exactly_one_averaged_score():
    a = arm(random_obs({0: [1, 0, 0.5, 0], 1: [0, 0, 0.5, 1], 2: [1, 0, 0.5, 0]}))
    assert a.values == {"q1": pytest.approx(2 / 3), "q2": 0.0, "q3": 0.5, "q4": pytest.approx(1 / 3)}
    assert a.n == 4 and a.n_universe == 4 and a.n_missing == 0
    assert a.replicates == (0, 1, 2) and a.n_replicates == 3
    assert all(k == 3 for k in a.k_available.values())


@pytest.mark.parametrize("n_shuffles", [1, 2, 3, 6, 25])
def test_the_sample_size_never_grows_with_the_number_of_shuffles(n_shuffles):
    rng = np.random.default_rng(n_shuffles)
    table = {s: list(rng.integers(0, 2, size=4).astype(float)) for s in range(n_shuffles)}
    a = arm(random_obs(table))
    assert a.n == 4 and len(a.values) == 4 and a.n_replicates == n_shuffles          # never 4 * n_shuffles


def test_repeating_identical_shuffles_does_not_change_the_arm():
    row = [1, 0, 0.25, 0.75]
    one, three = arm(random_obs({0: row})), arm(random_obs({0: row, 1: row, 2: row}))
    assert one.values == three.values and one.n == three.n == 4


def test_a_deterministic_strategy_is_one_run_with_one_value_per_question():
    a = arm([ob("ketrag", q, v) for q, v in zip(U, [1, 0, 1, 1])], strategy="ketrag")
    assert a.values == {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 1.0} and a.replicates == (None,) and a.n_partial == 0


def test_the_universe_order_is_kept_and_extra_questions_are_refused():
    a = arm(random_obs({0: [1, 1, 1, 1]}))
    assert list(a.values) == list(U)
    with pytest.raises(ArmError, match="not in the dataset's question set"):
        arm(random_obs({0: [1, 1, 1, 1]}) + [ob("random", "q9", 1.0, 0)])


# ============================================================================== missing shuffles
def test_a_question_missing_in_one_shuffle_is_averaged_over_the_others():
    a = arm(random_obs({0: [1, 0, 1, 0], 1: [0, 0, None, 0], 2: [1, 0, 0, 0]}))
    assert a.values["q3"] == 0.5 and a.k_available == {"q1": 3, "q2": 3, "q3": 2, "q4": 3}
    assert a.n == 4 and a.n_partial == 1 and a.n_missing == 0                  # still one value per question


def test_a_question_missing_in_every_shuffle_has_no_value_and_a_reason():
    a = arm(random_obs({0: [1, 0, None, 0], 1: [0, 0, None, 0]}))
    assert "q3" not in a.values and a.missing == {"q3": "generation_failed"} and a.k_available["q3"] == 0
    assert a.n == 3 and a.n_missing == 1 and a.n_partial == 0
    assert a.missing_reasons() == {"generation_failed": 1}


def test_a_shuffle_with_no_row_for_a_question_counts_as_missing_for_that_shuffle():
    obs = random_obs({0: [1, 0, 1, 0], 1: [1, 0, 1, 0]})
    obs = [o for o in obs if not (o.shuffle_seed == 1 and o.question_id == "q2")]
    a = arm(obs)
    assert a.values["q2"] == 0.0 and a.k_available["q2"] == 1 and a.n_partial == 1
    gone = arm([o for o in obs if o.question_id != "q2" or o.shuffle_seed != 0])
    assert gone.missing == {"q2": NO_OBSERVATION}


def test_missing_reasons_from_different_shuffles_are_combined_deterministically():
    obs = random_obs({0: [1, 0, 1, 0], 1: [1, 0, 1, 0]})
    obs = [o for o in obs if o.question_id != "q4"]
    obs += [ob("random", "q4", None, 0), ob("random", "q4", None, 1, outcome="no_prediction", judge_status=None)]
    obs = [o.model_copy(update={"generation_status": None}) if o.outcome == "no_prediction" else o for o in obs]
    assert arm(obs).missing == {"q4": "generation_failed|no_prediction"}


def test_strict_mode_drops_any_question_missing_in_any_shuffle():
    obs = random_obs({0: [1, 0, 1, 0], 1: [0, 0, None, 0], 2: [1, 0, 0, 0]})
    s = arm(obs, strict=True)
    assert s.missing == {"q3": INCOMPLETE_SHUFFLES} and s.n == 3 and s.strict
    assert set(s.values) == {"q1", "q2", "q4"} and s.values["q1"] == pytest.approx(2 / 3)
    assert arm(obs).n == 4                                                       # the default keeps it


def test_the_judge_metric_has_its_own_missing_pattern():
    obs = [ob("random", q, 1.0, 0, judge=j, judge_status="ok" if j is not None else "api_error")
           for q, j in zip(U, [True, False, None, True])]
    a = arm(obs, metric="judge_accuracy")
    assert a.values == {"q1": 1.0, "q2": 0.0, "q4": 1.0} and a.missing == {"q3": "judge_failed"}
    assert arm(obs, metric="f1").n == 4                                          # the answers themselves were fine


def test_shuffles_can_be_restricted_for_a_sensitivity_analysis():
    obs = random_obs({0: [1, 1, 1, 1], 1: [0, 0, 0, 0], 2: [0, 0, 0, 0]})
    assert arm(obs, shuffles=[0]).values["q1"] == 1.0 and arm(obs, shuffles=[0]).replicates == (0,)
    assert arm(obs, shuffles=[1, 2]).values["q1"] == 0.0
    with pytest.raises(ArmError, match="none of the requested shuffles"):
        arm(obs, shuffles=[7])
    with pytest.raises(ArmError, match="Random only"):
        arm([ob("ketrag", q, 1.0) for q in U], strategy="ketrag", shuffles=[0])


# ============================================================================== inconsistent input
def test_inconsistent_observations_are_errors_not_silent_choices():
    with pytest.raises(ArmError, match="no observations"):
        arm([])
    with pytest.raises(ArmError, match="deterministic but has shuffle seeds"):
        arm([ob("ketrag", q, 1.0, 0) for q in U], strategy="ketrag")
    with pytest.raises(ArmError, match="mixes runs with and without"):
        arm(random_obs({0: [1, 1, 1, 1]}) + [ob("random", q, 1.0, None) for q in U])
    with pytest.raises(ArmError, match="appears twice"):
        arm(random_obs({0: [1, 1, 1, 1]}) + [ob("random", "q1", 0.0, 0)])
    with pytest.raises(ArmError, match="duplicate ids"):
        build_arm(random_obs({0: [1, 1, 1, 1]}), dataset="hotpotqa", strategy="random", budget_pct=10, metric="f1",
                  universe=("q1", "q1", "q2"))


def test_other_cells_are_ignored():
    obs = random_obs({0: [1, 1, 1, 1]}) + random_obs({0: [0, 0, 0, 0]}, budget=25) + \
        [ob("ketrag", q, 0.0) for q in U] + random_obs({0: [0, 0, 0, 0]}, dataset="musique")
    assert arm(obs).values == {q: 1.0 for q in U}


# ============================================================================== deterministic_binary
def test_deterministic_binary_means_single_run_zero_one_values():
    assert arm([ob("ketrag", q, v) for q, v in zip(U, [1, 0, 1, 1])], strategy="ketrag").deterministic_binary
    assert arm(random_obs({0: [1, 0, 1, 1]})).deterministic_binary              # Random with ONE shuffle: a single run
    assert not arm([ob("ketrag", q, v) for q, v in zip(U, [1, 0.5, 1, 1])], strategy="ketrag").deterministic_binary
    assert not arm(random_obs({0: [1, 0, 1, 1], 1: [1, 0, 1, 1], 2: [1, 0, 1, 1]})).deterministic_binary  # an average


# ============================================================================== pairing
def test_complete_case_pairs_only_questions_both_arms_have():
    a = arm(random_obs({0: [1, 0, None, 1], 1: [1, 0, None, 1]}))
    b = arm([ob("ketrag", q, v) for q, v in zip(U, [0, 1, 1, None])], strategy="ketrag")
    s = pair_arms(a, b)
    assert s.question_ids == ("q1", "q2") and s.n_pairs == 2 and s.n_universe == 4 and s.n_dropped == 2
    assert list(s.a) == [1.0, 0.0] and list(s.b) == [0.0, 1.0] and list(s.d) == [1.0, -1.0]
    assert s.n_missing_a == 1 and s.n_missing_b == 1
    assert s.missing_reasons_a == {"generation_failed": 1} and s.missing_reasons_b == {"generation_failed": 1}
    assert s.missing_policy == "complete_case" and (s.strategy_a, s.strategy_b) == ("random", "ketrag")


def test_missing_as_wrong_keeps_the_whole_universe():
    a = arm(random_obs({0: [1, 0, None, 1]}))
    b = arm([ob("ketrag", q, v) for q, v in zip(U, [0, 1, 1, None])], strategy="ketrag")
    s = pair_arms(a, b, missing="as_wrong")
    assert s.question_ids == U and s.n_pairs == 4 and s.n_dropped == 0
    assert list(s.a) == [1.0, 0.0, 0.0, 1.0] and list(s.b) == [0.0, 1.0, 1.0, 0.0]


def test_shuffle_repetitions_never_inflate_the_number_of_pairs():
    k = arm([ob("ketrag", q, v) for q, v in zip(U, [1, 0, 1, 1])], strategy="ketrag")
    for shuffles in (1, 3, 9):
        r = arm(random_obs({s: [1, 0, 0, 1] for s in range(shuffles)}))
        assert pair_arms(r, k).n_pairs == 4


def test_mcnemar_eligibility_follows_the_arms():
    k1 = arm([ob("ketrag", q, v) for q, v in zip(U, [1, 0, 1, 1])], strategy="ketrag")
    k2 = arm([ob("fastgraphrag", q, v) for q, v in zip(U, [0, 0, 1, 1])], strategy="fastgraphrag")
    r3 = arm(random_obs({s: [1, 0, 1, 1] for s in range(3)}))
    assert pair_arms(k1, k2).both_deterministic_binary
    assert not pair_arms(k1, r3).both_deterministic_binary                       # a shuffle average is not binary
    assert not pair_arms(k1, k2, missing="as_wrong").both_deterministic_binary


def test_pairing_refuses_mismatched_arms():
    a = arm(random_obs({0: [1, 1, 1, 1]}))
    k = arm([ob("ketrag", q, 1.0) for q in U], strategy="ketrag")
    with pytest.raises(ArmError, match="with itself"):
        pair_arms(a, a)
    with pytest.raises(ArmError, match="share dataset, budget and metric"):
        pair_arms(a, build_arm([ob("ketrag", q, 1.0, budget=25) for q in U], dataset="hotpotqa", strategy="ketrag",
                               budget_pct=25, metric="f1", universe=U))
    with pytest.raises(ArmError, match="share dataset, budget and metric"):
        pair_arms(a, arm([ob("ketrag", q, 1.0) for q in U], strategy="ketrag", metric="em"))
    with pytest.raises(ArmError, match="one question universe"):
        pair_arms(a, build_arm([ob("ketrag", q, 1.0) for q in U[:3]], dataset="hotpotqa", strategy="ketrag",
                               budget_pct=10, metric="f1", universe=U[:3]))
    with pytest.raises(ArmError, match="unknown missing-data policy"):
        pair_arms(a, k, missing="drop")


# ============================================================================== Random's own variability
def test_shuffle_summary_reports_per_shuffle_means_and_spread():
    obs = random_obs({0: [1, 1, 0, 0], 1: [1, 0, 0, 0], 2: [1, 1, 1, 1]})
    s = shuffle_summary(obs, dataset="hotpotqa", budget_pct=10, metric="f1", universe=U)
    assert s.n_shuffles == 3
    assert s.per_shuffle == ((0, 0.5, 4), (1, 0.25, 4), (2, 1.0, 4))
    assert s.mean == pytest.approx(0.583333333) and s.min == 0.25 and s.max == 1.0
    assert s.sd == pytest.approx(np.std([0.5, 0.25, 1.0], ddof=1), abs=1e-9)


def test_shuffle_summary_with_one_shuffle_has_no_sd_and_with_none_is_an_error():
    s = shuffle_summary(random_obs({0: [1, 0, 0, 0]}), dataset="hotpotqa", budget_pct=10, metric="f1", universe=U)
    assert s.n_shuffles == 1 and s.sd is None and s.mean == 0.25
    with pytest.raises(ArmError, match="no Random shuffles"):
        shuffle_summary([ob("random", q, 1.0) for q in U], dataset="hotpotqa", budget_pct=10, metric="f1", universe=U)


def test_shuffle_summary_counts_measured_questions_per_shuffle():
    s = shuffle_summary(random_obs({0: [1, 0, None, 0], 1: [1, 1, 1, 1]}), dataset="hotpotqa", budget_pct=10,
                        metric="f1", universe=U)
    assert s.per_shuffle[0] == (0, pytest.approx(1 / 3), 3) and s.per_shuffle[1] == (1, 1.0, 4)


# ============================================================================== real runner output
def test_three_real_shuffles_give_one_value_per_question(tmp_path):
    """On output of the real runner (synthetic non-mock matrix, 3 questions): 3 shuffles x 3 questions are
    9 rows, but the Random arm has 3 values, and pairing it with KET-RAG gives at most 3 pairs."""
    from analysis.loader import load_analysis_input
    from tests.helpers_analysis import real_matrix
    a = load_analysis_input(real_matrix(tmp_path, strategies=("random", "ketrag"), budgets=(0.5,),
                                        random_seeds=[0, 1, 2]))
    universe = a.question_ids["hotpotqa"]
    assert sum(1 for o in a.observations if o.strategy == "random") == 9
    r = build_arm(a.observations, dataset="hotpotqa", strategy="random", budget_pct=50, metric="f1", universe=universe)
    k = build_arm(a.observations, dataset="hotpotqa", strategy="ketrag", budget_pct=50, metric="f1", universe=universe)
    assert r.n == 3 == len(universe) and r.replicates == (0, 1, 2) and k.replicates == (None,)
    assert pair_arms(r, k).n_pairs == 3
    assert shuffle_summary(a.observations, dataset="hotpotqa", budget_pct=50, metric="f1",
                           universe=universe).n_shuffles == 3
