"""Phase 12, increment 2: analysis/stats.py. The permutation test is checked against an independent
brute-force enumeration written here (itertools, no shared code), Holm against hand-calculated cases and an
independent formula, McNemar against hand counts and the sign-flip test, the bootstrap against a manual
re-implementation and coverage / calibration simulations (all seeded, so they are deterministic).
No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools

import numpy as np
import pytest
from scipy import stats as sps

from analysis.random_avg import build_arm, pair_arms
from analysis.schemas import QuestionObservation
from analysis.stats import (ArmDescription, StatsError, apply_corrections, bh_adjust, bootstrap_mean_ci,
                            compare_paired, derive_seed, describe_arm, holm_adjust, mcnemar_exact, paired_effect,
                            sign_flip_test)


def brute_force_p(d):
    """Independent exhaustive two-sided sign-flip p-value: enumerate every +/- pattern with itertools."""
    d = [x for x in d if x != 0]
    if not d:
        return 1.0
    obs = abs(sum(d))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        hits += abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-9
    return hits / total


# ============================================================================== sign-flip: exhaustive validation
@pytest.mark.parametrize("d,p", [([1, 2, 3], 2 / 8), ([1, 1, 1, 1], 2 / 16), ([3, -1, 1], 6 / 8),
                                 ([5], 1.0), ([1, -1], 1.0), ([2, 2, -1], 4 / 8)])
def test_sign_flip_matches_hand_calculated_values(d, p):
    for method in ("enumerate", "auto"):
        assert sign_flip_test(d, method=method).p_value == pytest.approx(p, abs=1e-9)
    assert brute_force_p(d) == pytest.approx(p)


def test_sign_flip_enumeration_equals_independent_brute_force_on_random_samples():
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(1, 11))
        kind = trial % 3
        if kind == 0:   d = rng.normal(size=n)                                   # continuous
        elif kind == 1: d = rng.integers(-3, 4, size=n).astype(float)             # integers with zeros and ties
        else:           d = rng.choice([-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0], size=n)   # shuffle-average values
        r = sign_flip_test(d, method="enumerate")
        assert r.p_value == pytest.approx(brute_force_p(d), abs=1e-9), (trial, d)
        assert r.method == "exact_enumeration" and r.n == n and r.n_nonzero == int(np.count_nonzero(d))


def exact_rational_p(d):
    """Exhaustive p-value in exact rational arithmetic (no floating-point noise at all)."""
    from fractions import Fraction
    fr = [Fraction(str(x)) for x in d if x != 0]
    if not fr:
        return 1.0
    obs = abs(sum(fr))
    hits = sum(abs(sum(s * x for s, x in zip(signs, fr))) >= obs for signs in itertools.product((1, -1), repeat=len(fr)))
    return hits / 2 ** len(fr)


@pytest.mark.parametrize("d,p", [([-0.6, 0.2, -0.8, 0.9], 1.0), ([-0.9, 0.7, -0.5, 0.5, 0.9, 0.1], 0.6875),
                                 ([0.1, 0.3, -0.6, 0.6, 0.5, 0.1], 0.40625)])
def test_exact_ties_survive_floating_point_noise(d, p):
    """Decimal differences (like F1 gaps) have sign patterns whose sums are mathematically equal but differ in the
    last bit. A bare >= comparison gets these wrong (0.875, 0.65625 and 0.3125 here); the tolerance must not."""
    assert sign_flip_test(d, method="enumerate").p_value == pytest.approx(p, abs=1e-9)
    assert exact_rational_p(d) == pytest.approx(p, abs=1e-12)


def test_monte_carlo_also_handles_exact_ties_on_decimal_data():
    """Same tie-heavy decimals through the Monte Carlo path: the exact p is 0.6875; a bare >= would give about 0.656."""
    d = [-0.9, 0.7, -0.5, 0.5, 0.9, 0.1]
    mc = sign_flip_test(d, method="monte_carlo", n_permutations=200_000, seed=21)
    assert abs(mc.p_value - 0.6875) < 0.01


def test_enumeration_equals_exact_rational_arithmetic_on_decimal_data():
    import random
    rnd = random.Random(3)
    for _ in range(150):
        d = [rnd.choice([-1, 1]) * rnd.randint(1, 9) / 10 for _ in range(rnd.randint(3, 9))]
        assert sign_flip_test(d, method="enumerate").p_value == pytest.approx(exact_rational_p(d), abs=1e-9), d


def test_zero_differences_do_not_change_the_p_value_but_are_counted():
    base = [0.3, -0.1, 0.25, 0.4, 0.2]
    padded = base + [0.0, 0.0, 0.0]
    a, b = sign_flip_test(base), sign_flip_test(padded)
    assert a.p_value == b.p_value and b.n == 8 and b.n_nonzero == 5
    assert b.mean_diff == pytest.approx(sum(base) / 8, abs=1e-9)


def test_all_zero_differences_give_p_one():
    r = sign_flip_test([0.0, 0.0, 0.0])
    assert r.p_value == 1.0 and r.method == "all_zero" and r.n_nonzero == 0 and r.n == 3
    assert sign_flip_test([]).p_value == 1.0 and sign_flip_test([]).n == 0


def test_the_test_is_symmetric_in_sign_and_order():
    d = np.array([0.4, -0.1, 0.25, 0.3, -0.05, 0.2, 0.35])
    assert sign_flip_test(d).p_value == sign_flip_test(-d).p_value == sign_flip_test(d[::-1]).p_value


def test_the_binomial_shortcut_equals_enumeration_for_equal_magnitudes():
    rng = np.random.default_rng(1)
    for _ in range(40):
        d = rng.choice([-1.0, 0.0, 1.0], size=int(rng.integers(1, 13)), p=[0.35, 0.3, 0.35])
        e, b = sign_flip_test(d, method="enumerate"), sign_flip_test(d, method="binomial")
        assert e.p_value == pytest.approx(b.p_value, abs=1e-9)
        assert b.method == ("exact_binomial" if np.any(d != 0) else "all_zero")      # an all-zero sample is reported as such
    for scale in (1.0, 0.25):                                                     # any common magnitude
        d = scale * np.array([1, 1, -1, 1, 1, -1, 1])
        assert sign_flip_test(d, method="binomial").p_value == pytest.approx(brute_force_p(list(d)), abs=1e-9)


def test_the_binomial_shortcut_refuses_unequal_magnitudes():
    with pytest.raises(StatsError, match="equal"):
        sign_flip_test([1.0, 0.5, -1.0], method="binomial")


def test_auto_picks_the_exact_method_whenever_one_applies():
    rng = np.random.default_rng(2)
    assert sign_flip_test(rng.choice([-1.0, 0.0, 1.0], size=200)).method == "exact_binomial"   # exact at n=200
    assert sign_flip_test(rng.normal(size=10)).method == "exact_enumeration"
    assert sign_flip_test(rng.normal(size=40), n_permutations=999).method == "monte_carlo"
    assert sign_flip_test(rng.normal(size=14), exact_max_n=10, n_permutations=999).method == "monte_carlo"


# ============================================================================== sign-flip: Monte Carlo
def test_monte_carlo_agrees_with_exhaustive_enumeration():
    d = np.random.default_rng(3).normal(0.4, 1.0, size=14)
    exact = sign_flip_test(d, method="enumerate").p_value
    mc = sign_flip_test(d, method="monte_carlo", n_permutations=200_000, seed=11)
    assert mc.method == "monte_carlo" and mc.n_permutations == 200_000
    assert abs(mc.p_value - exact) < 0.01                                          # ~10 standard errors


def test_monte_carlo_agrees_with_the_exact_binomial_on_binary_data():
    d = np.random.default_rng(4).choice([-1.0, 0.0, 1.0], size=60, p=[0.15, 0.5, 0.35])
    exact = sign_flip_test(d, method="binomial").p_value
    assert abs(sign_flip_test(d, method="monte_carlo", n_permutations=200_000, seed=5).p_value - exact) < 0.01


def test_monte_carlo_is_reproducible_and_never_zero():
    d = np.random.default_rng(5).normal(1.5, 1.0, size=30)                         # a very clear difference
    a = sign_flip_test(d, n_permutations=1_000, seed=7)
    b = sign_flip_test(d, n_permutations=1_000, seed=7)
    assert a == b and a.method == "monte_carlo"
    assert a.p_value >= 1 / 1_001 - 1e-9                                           # (1 + 0) / (B + 1): the identity permutation counts
    assert sign_flip_test(d, n_permutations=1_000, seed=8).p_value >= 1 / 1_001 - 1e-9


def test_monte_carlo_p_value_depends_on_the_seed_only_through_sampling_noise():
    d = np.random.default_rng(6).normal(0.15, 1.0, size=50)
    ps = [sign_flip_test(d, n_permutations=4_000, seed=s).p_value for s in range(8)]
    assert len(set(ps)) > 1 and max(ps) - min(ps) < 0.08


def test_type_one_error_is_calibrated_under_the_null():
    """Seeded simulation: symmetric, non-normal differences with mean 0 -> rejection rate at 5% close to 5%."""
    rng = np.random.default_rng(7)
    sims, rejected = 800, 0
    for i in range(sims):
        d = rng.standard_t(df=4, size=40)
        rejected += sign_flip_test(d, n_permutations=499, seed=i).p_value <= 0.05
    assert 0.03 <= rejected / sims <= 0.07, rejected / sims


def test_the_test_has_power_against_a_real_shift():
    rng = np.random.default_rng(8)
    rej = sum(sign_flip_test(rng.normal(0.6, 1.0, size=40), n_permutations=499, seed=i).p_value <= 0.05
              for i in range(200))
    assert rej / 200 > 0.8


def test_invalid_inputs_are_errors():
    with pytest.raises(StatsError, match="NaN"):
        sign_flip_test([1.0, float("nan")])
    with pytest.raises(StatsError, match="unknown method"):
        sign_flip_test([1.0], method="magic")
    with pytest.raises(StatsError, match="n_permutations"):
        sign_flip_test(np.random.default_rng(0).normal(size=30), n_permutations=0)
    with pytest.raises(StatsError, match="one-dimensional"):
        sign_flip_test([[1.0, 2.0]])
    with pytest.raises(StatsError, match="limit 24"):
        sign_flip_test(np.random.default_rng(0).normal(size=30), method="enumerate")


# ============================================================================== McNemar
def test_mcnemar_hand_calculated_case():
    a = [1, 1, 1, 0, 0, 1, 0, 1]
    b = [0, 0, 1, 0, 1, 1, 0, 0]
    r = mcnemar_exact(a, b)
    assert (r.n_a_only, r.n_b_only, r.n_both_correct, r.n_both_wrong, r.n_pairs) == (3, 1, 2, 2, 8)
    assert r.p_value == pytest.approx(10 / 16)                                      # P(X in {0,1,3,4}), X ~ Bin(4, 1/2)


def test_mcnemar_extremes():
    assert mcnemar_exact([1] * 10, [0] * 10).p_value == pytest.approx(2 / 1024)
    assert mcnemar_exact([1, 0, 1], [1, 0, 1]).p_value == 1.0                       # no discordant pair
    assert mcnemar_exact([1, 0], [0, 1]).p_value == 1.0                             # b == c


def test_mcnemar_equals_the_sign_flip_test_on_binary_data():
    """Not independent confirmation: for 0/1 outcomes the two are the SAME test on the SAME pairs, so the
    p-values must be identical (to rounding). This verifies the implementations against each other."""
    rng = np.random.default_rng(9)
    for n in (3, 8, 17, 40, 120, 200):
        for _ in range(8):
            a = rng.integers(0, 2, size=n).astype(float)
            b = np.where(rng.random(n) < 0.6, a, 1 - a)                              # correlated, like two strategies
            mc, sf = mcnemar_exact(a, b), sign_flip_test(a - b)
            assert mc.p_value == pytest.approx(sf.p_value, abs=1e-9), (n, mc, sf)
            assert mc.n_a_only == int(((a - b) == 1).sum()) and mc.n_b_only == int(((a - b) == -1).sum())


def test_mcnemar_says_it_is_not_independent_confirmation():
    assert "not independent confirmation" in mcnemar_exact([1, 0], [0, 0]).note


def test_mcnemar_needs_binary_equal_length_data():
    with pytest.raises(StatsError, match="binary"):
        mcnemar_exact([1.0, 0.5], [0.0, 1.0])
    with pytest.raises(StatsError, match="same length"):
        mcnemar_exact([1, 0], [1])


# ============================================================================== bootstrap
def test_bootstrap_matches_a_manual_reimplementation():
    x = np.array([0.1, 0.4, 0.4, 0.9, 0.0, 0.7, 0.3])
    r = bootstrap_mean_ci(x, n_resamples=1_500, seed=123)
    rng = np.random.default_rng(123)
    means = x[rng.integers(0, x.size, size=(1_500, x.size))].mean(axis=1)           # one block (< 2000 resamples)
    assert r.low == pytest.approx(np.quantile(means, 0.025), abs=1e-9)
    assert r.high == pytest.approx(np.quantile(means, 0.975), abs=1e-9)
    assert r.mean == pytest.approx(x.mean(), abs=1e-9) and r.n == 7 and r.level == 0.95 and r.n_resamples == 1_500


def test_bootstrap_is_deterministic_and_seed_dependent():
    x = np.random.default_rng(10).normal(size=50)
    a, b = bootstrap_mean_ci(x, seed=5), bootstrap_mean_ci(x, seed=5)
    assert a == b and a.n_resamples == 10_000                                         # the default, spanning 5 blocks
    c = bootstrap_mean_ci(x, seed=6)
    assert (c.low, c.high) != (a.low, a.high) and c.mean == a.mean


def test_bootstrap_does_not_depend_on_the_order_of_the_data_for_the_mean():
    x = np.random.default_rng(11).normal(size=30)
    assert bootstrap_mean_ci(x, seed=1).mean == bootstrap_mean_ci(x[::-1], seed=1).mean


def test_bootstrap_interval_properties():
    x = np.random.default_rng(12).normal(0.3, 1.0, size=80)
    r90, r95, r99 = (bootstrap_mean_ci(x, level=l, seed=2) for l in (0.90, 0.95, 0.99))
    assert r95.low <= r95.mean <= r95.high and not r95.degenerate
    assert (r99.high - r99.low) > (r95.high - r95.low) > (r90.high - r90.low)


def test_bootstrap_degenerate_inputs():
    const = bootstrap_mean_ci([0.5] * 10, seed=0)
    assert const.low == const.high == const.mean == 0.5 and const.degenerate
    one = bootstrap_mean_ci([0.7], seed=0)
    assert one.low == one.high == 0.7 and one.degenerate and one.n == 1
    assert bootstrap_mean_ci([], seed=0) is None
    with pytest.raises(StatsError, match="level"):
        bootstrap_mean_ci([1.0, 2.0], level=1.0)
    with pytest.raises(StatsError, match="NaN"):
        bootstrap_mean_ci([1.0, float("nan")])


def test_bootstrap_coverage_is_close_to_nominal():
    """Seeded simulation: the 95% interval covers the true mean (0) in about 95% of normal samples (n = 40)."""
    rng = np.random.default_rng(13)
    sims, covered = 300, 0
    for i in range(sims):
        r = bootstrap_mean_ci(rng.normal(0, 1, size=40), n_resamples=500, seed=i)
        covered += r.low <= 0.0 <= r.high
    assert 0.90 <= covered / sims <= 0.99, covered / sims


def test_derived_seeds_are_stable_and_key_dependent():
    assert derive_seed(0, "a", 1) == derive_seed(0, "a", 1)
    assert len({derive_seed(0, "a", 1), derive_seed(0, "a", 2), derive_seed(1, "a", 1), derive_seed(0, "b", 1)}) == 4
    assert 0 <= derive_seed(0, "x") < 2 ** 64


# ============================================================================== Holm
@pytest.mark.parametrize("p,expected", [
    ([0.01, 0.04, 0.03, 0.005], [0.03, 0.06, 0.06, 0.02]),     # sorted .005x4=.02, .01x3=.03, .03x2=.06, .04x1=.04 -> max(.06) = .06
    ([0.5, 0.6], [1.0, 1.0]),                                  # .5x2 = 1.0 (capped); .6x1 = .6 -> running max 1.0
    ([0.02, 0.02, 0.02], [0.06, 0.06, 0.06]),                  # ties: 3 x .02
    ([0.04], [0.04]),                                          # a family of one is unchanged
    ([0.001, 0.2], [0.002, 0.2]),
    ([0.0, 0.5], [0.0, 0.5]),
])
def test_holm_matches_hand_calculated_cases(p, expected):
    assert holm_adjust(p) == pytest.approx(expected, abs=1e-9)


def test_holm_excludes_untestable_comparisons_from_the_family():
    assert holm_adjust([0.01, None, 0.04]) == [0.02, None, 0.04]                     # m = 2, not 3
    assert holm_adjust([None, None]) == [None, None] and holm_adjust([]) == []


def test_holm_matches_an_independent_formula_and_its_invariants():
    rng = np.random.default_rng(14)
    for _ in range(100):
        m = int(rng.integers(1, 15))
        p = list(np.round(rng.random(m) ** 2, 4))
        adj = holm_adjust(p)
        order = sorted(range(m), key=lambda i: p[i])
        # independent definition: adj_(k) = min(1, max_{j<=k} (m-j+1) p_(j))
        for k, i in enumerate(order):
            want = min(1.0, max((m - j) * p[order[j]] for j in range(k + 1)))
            assert adj[i] == pytest.approx(want, abs=1e-9)
        assert all(a >= x - 1e-12 for a, x in zip(adj, p)) and all(0 <= a <= 1 for a in adj)
        bonf = [min(1.0, m * x) for x in p]
        assert all(a <= b + 1e-12 for a, b in zip(adj, bonf))                         # never more conservative than Bonferroni
        assert [adj[i] for i in order] == sorted(adj[i] for i in order)               # monotone in p


def test_holm_controls_the_family_wise_error_rate_under_the_global_null():
    rng = np.random.default_rng(15)
    sims, m, any_rejection = 4_000, 10, 0
    for _ in range(sims):
        any_rejection += min(holm_adjust(list(rng.random(m)))) <= 0.05
    assert any_rejection / sims <= 0.06                                              # nominal 0.05 (SE ~ 0.0035)


def test_holm_rejects_out_of_range_p_values():
    with pytest.raises(StatsError, match="outside"):
        holm_adjust([0.1, 1.2])
    with pytest.raises(StatsError, match="outside"):
        bh_adjust([-0.1])


def test_benjamini_hochberg_hand_case_and_scipy_cross_check():
    assert bh_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02], abs=1e-9)
    assert bh_adjust([0.01, None, 0.04]) == [0.02, None, 0.04]
    rng = np.random.default_rng(16)
    for _ in range(30):
        p = list(np.round(rng.random(int(rng.integers(1, 20))) ** 2, 5))
        assert bh_adjust(p) == pytest.approx(list(sps.false_discovery_control(p, method="bh")), abs=1e-8)


# ============================================================================== effect sizes
def test_paired_effect_hand_calculated():
    e = paired_effect([1, 0, 1, 1], [0, 0, 1, 0])                                    # d = [1, 0, 0, 1]
    assert (e.n, e.mean_a, e.mean_b, e.mean_diff) == (4, 0.75, 0.25, 0.5)
    assert e.sd_diff == pytest.approx(np.sqrt(1 / 3), abs=1e-9) and e.dz == pytest.approx(0.8660254, abs=1e-6)
    assert (e.n_a_better, e.n_b_better, e.n_tied) == (2, 0, 2)


def test_paired_effect_degenerate_cases():
    one = paired_effect([0.5], [0.2])
    assert one.n == 1 and one.sd_diff is None and one.dz is None and one.mean_diff == pytest.approx(0.3)
    same = paired_effect([0.2, 0.4], [0.2, 0.4])
    assert same.mean_diff == 0.0 and same.dz is None and same.n_tied == 2
    const = paired_effect([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])                         # sd of d is 0: dz undefined, not infinite
    assert const.mean_diff == 1.0 and const.sd_diff == 0.0 and const.dz is None
    empty = paired_effect([], [])
    assert empty.n == 0 and empty.mean_diff is None
    with pytest.raises(StatsError, match="same length"):
        paired_effect([1.0], [1.0, 2.0])


# ============================================================================== putting it together
U = tuple(f"q{i}" for i in range(1, 9))


def obs(strategy, vals, shuffle=None, budget=10, metric_judge=None):
    out = []
    for q, v in zip(U, vals):
        outcome = "failed" if v is None else "answered"
        out.append(QuestionObservation(
            condition_id=f"{strategy}-{shuffle}", dataset="hotpotqa", strategy=strategy, budget_pct=budget, seed=42,
            shuffle_seed=shuffle, question_id=q, outcome=outcome, generation_status="ok", em=v, f1=v,
            judge_correct=None if v is None else bool(v), judge_status="skipped_failed_generation" if v is None else "ok"))
    return out


def arm(observations, strategy, metric="f1", **kw):
    return build_arm(observations, dataset="hotpotqa", strategy=strategy, budget_pct=10, metric=metric, universe=U, **kw)


def test_compare_paired_reports_direction_ci_effect_and_both_tests():
    a = obs("ketrag", [1, 1, 1, 1, 1, 0, 1, 1])
    b = obs("fastgraphrag", [1, 0, 0, 1, 0, 0, 1, 0])
    c = compare_paired(pair_arms(arm(a, "ketrag", "em"), arm(b, "fastgraphrag", "em")), n_resamples=2_000)
    assert (c.strategy_a, c.strategy_b, c.n_pairs, c.n_universe, c.n_dropped) == ("ketrag", "fastgraphrag", 8, 8, 0)
    assert c.effect.mean_diff == pytest.approx(0.5) and c.ci.low <= 0.5 <= c.ci.high and c.ci.n == 8
    assert (c.effect.n_a_better, c.effect.n_b_better, c.effect.n_tied) == (4, 0, 4)
    assert c.sign_flip.method == "exact_binomial" and c.sign_flip.p_value == pytest.approx(2 / 16)
    assert c.mcnemar is not None and c.mcnemar.p_value == c.sign_flip.p_value          # same test on the same pairs
    assert c.mcnemar.n_a_only == 4 and c.mcnemar.n_b_only == 0 and not c.identical and c.flags == []
    flipped = compare_paired(pair_arms(arm(b, "fastgraphrag", "em"), arm(a, "ketrag", "em")), n_resamples=2_000)
    assert flipped.effect.mean_diff == pytest.approx(-0.5) and flipped.sign_flip.p_value == c.sign_flip.p_value


def test_mcnemar_is_only_reported_for_single_run_binary_arms():
    k = obs("ketrag", [1, 0, 1, 1, 0, 1, 1, 0])
    r3 = obs("random", [1, 0, 1, 0, 0, 1, 1, 0], 0) + obs("random", [1, 1, 0, 0, 0, 1, 1, 0], 1) \
        + obs("random", [1, 0, 1, 0, 1, 1, 1, 0], 2)
    shuffle_avg = compare_paired(pair_arms(arm(r3 + k, "random", "em"), arm(r3 + k, "ketrag", "em")), n_resamples=500)
    assert shuffle_avg.mcnemar is None and "shuffle_averaged_arm" in shuffle_avg.flags
    f1 = compare_paired(pair_arms(arm(k, "ketrag", "f1"), arm(obs("fastgraphrag", [.5, 0, 1, .5, 0, 1, 1, 0]), "fastgraphrag")),
                        n_resamples=500)
    assert f1.mcnemar is None                                                           # F1 is not binary
    as_wrong = compare_paired(pair_arms(arm(k, "ketrag", "em"), arm(obs("fastgraphrag", [1, 0, 0, 1, 0, 1, 1, 0]),
                                                                    "fastgraphrag", "em"), missing="as_wrong"), n_resamples=500)
    assert as_wrong.mcnemar is None


def test_the_sample_size_is_the_number_of_questions_not_questions_times_shuffles():
    k = obs("ketrag", [1, 0, 1, 1, 0, 1, 1, 0])
    for n_shuffles in (1, 3, 10):
        r = sum((obs("random", [1, 0, 0, 1, 0, 0, 1, 0], s) for s in range(n_shuffles)), [])
        c = compare_paired(pair_arms(arm(r + k, "random"), arm(r + k, "ketrag")), n_resamples=300)
        assert c.n_pairs == 8 and c.ci.n == 8 and c.effect.n == 8 and c.sign_flip.n == 8


def test_identical_arms_are_flagged_not_forced():
    v = [1, 0, 1, 1, 0, 1, 1, 0]
    c = compare_paired(pair_arms(arm(obs("ketrag", v), "ketrag"), arm(obs("fastgraphrag", v), "fastgraphrag")),
                       n_resamples=300)
    assert c.identical and "all_differences_zero" in c.flags and c.sign_flip.p_value == 1.0
    assert c.effect.mean_diff == 0.0 and c.ci.low == c.ci.high == 0.0 and c.ci.degenerate
    different = compare_paired(pair_arms(arm(obs("ketrag", v), "ketrag"),
                                         arm(obs("fastgraphrag", [1, 0, 1, 1, 0, 1, 1, 1]), "fastgraphrag")), n_resamples=300)
    assert not different.identical and different.effect.mean_diff == pytest.approx(-0.125)


def test_missing_values_are_reported_in_the_comparison():
    a = obs("ketrag", [1, 0, None, 1, 1, 0, 1, 1])
    b = obs("fastgraphrag", [0, 0, 1, None, 1, 0, 0, 1])
    cc = compare_paired(pair_arms(arm(a, "ketrag"), arm(b, "fastgraphrag")), n_resamples=300)
    assert (cc.n_pairs, cc.n_dropped, cc.n_missing_a, cc.n_missing_b) == (6, 2, 1, 1)
    assert "2_questions_dropped_for_missing_values" in cc.flags and cc.missing_policy == "complete_case"
    aw = compare_paired(pair_arms(arm(a, "ketrag"), arm(b, "fastgraphrag"), missing="as_wrong"), n_resamples=300)
    assert (aw.n_pairs, aw.n_dropped, aw.missing_policy) == (8, 0, "as_wrong")


def test_partial_shuffles_are_flagged():
    r = obs("random", [1, 0, 0, 1, 0, 0, 1, 0], 0) + obs("random", [1, 0, None, 1, 0, 0, 1, 0], 1)
    k = obs("ketrag", [1, 0, 1, 1, 0, 1, 1, 0])
    c = compare_paired(pair_arms(arm(r + k, "random"), arm(r + k, "ketrag")), n_resamples=300)
    assert "some_values_average_fewer_shuffles_than_planned" in c.flags and c.n_pairs == 8


def test_a_comparison_with_no_paired_question_cannot_be_made():
    a = obs("ketrag", [1] * 8)
    b = obs("fastgraphrag", [None] * 8)
    with pytest.raises(StatsError, match="nothing to compare"):
        compare_paired(pair_arms(arm(a, "ketrag"), arm(b, "fastgraphrag")))


def test_a_single_pair_is_flagged():
    a = obs("ketrag", [1] + [None] * 7)
    b = obs("fastgraphrag", [0] * 8)
    c = compare_paired(pair_arms(arm(a, "ketrag"), arm(b, "fastgraphrag")), n_resamples=200)
    assert c.n_pairs == 1 and "fewer_than_2_pairs" in c.flags and c.ci.degenerate


def test_results_do_not_depend_on_the_order_comparisons_are_run():
    k = obs("ketrag", [1, 0, 1, 1, 0, 1, 1, 0])
    f = obs("fastgraphrag", [0.5, 0, 1, 0.25, 0, 1, 0.75, 0])
    l = obs("lazygraphrag", [1, 1, 0.5, 1, 0, 0.5, 1, 0.25])
    def run(order):
        arms = {"ketrag": arm(k, "ketrag"), "fastgraphrag": arm(f, "fastgraphrag"), "lazygraphrag": arm(l, "lazygraphrag")}
        return {(x, y): compare_paired(pair_arms(arms[x], arms[y]), n_resamples=1_000, n_permutations=500, exact_max_n=2)
                for x, y in order}
    pairs = [("ketrag", "fastgraphrag"), ("ketrag", "lazygraphrag"), ("fastgraphrag", "lazygraphrag")]
    one, two = run(pairs), run(pairs[::-1])
    for key in pairs:
        assert one[key] == two[key] and one[key].sign_flip.method == "monte_carlo"
    assert one[pairs[0]].ci != one[pairs[1]].ci                                         # different comparisons, different streams


def test_describe_arm_uses_questions_as_the_unit():
    r = sum((obs("random", [1, 0, 0, 1, 0, 0, 1, None], s) for s in range(3)), [])
    d = describe_arm(arm(r, "random"), n_resamples=500)
    assert isinstance(d, ArmDescription)
    assert (d.n, d.n_universe, d.n_missing, d.n_replicates, d.n_partial) == (7, 8, 1, 3, 0)    # 7 questions, not 21
    assert d.ci.n == 7 and d.mean == pytest.approx(3 / 7, abs=1e-9) and d.missing_reasons == {"generation_failed": 1}
    assert describe_arm(arm(r, "random"), n_resamples=500, base_seed=0) == describe_arm(arm(r, "random"), n_resamples=500)


def test_describe_arm_seed_changes_the_resampling_but_not_the_mean():
    # continuous scores: unlike 0/1 data, different resamples give visibly different percentile endpoints
    r = obs("random", [0.91, 0.12, 0.37, 0.64, 0.23, 0.48, 0.85, 0.05], 0)
    a = describe_arm(arm(r, "random"), n_resamples=300, base_seed=0)
    b = describe_arm(arm(r, "random"), n_resamples=300, base_seed=1)
    assert a.mean == b.mean and (a.ci.low, a.ci.high) != (b.ci.low, b.ci.high)


def test_apply_corrections_uses_one_family_and_does_not_mutate():
    comps = []
    for strat, vals in (("ketrag", [1, 1, 1, 1, 1, 1, 1, 0]), ("fastgraphrag", [1, 1, 1, 0, 0, 0, 0, 0]),
                        ("lazygraphrag", [1, 0, 0, 0, 0, 0, 0, 0])):
        comps.append(compare_paired(pair_arms(arm(obs(strat, vals), strat, "em"),
                                              arm(obs("random", [0, 1, 0, 0, 1, 0, 0, 1]), "random", "em")),
                                    n_resamples=300))
    out = apply_corrections(comps, alpha=0.05)
    raw = [c.sign_flip.p_value for c in comps]
    assert [c.p_holm for c in out] == pytest.approx(holm_adjust(raw)) and [c.p_bh for c in out] == pytest.approx(bh_adjust(raw))
    assert all(c.p_holm is None and c.significant_holm is None for c in comps)          # inputs untouched
    assert [c.significant_holm for c in out] == [p <= 0.05 for p in holm_adjust(raw)]
    assert all(o.sign_flip == c.sign_flip and o.effect == c.effect for o, c in zip(out, comps))
    assert apply_corrections([], alpha=0.05) == []
    with pytest.raises(StatsError, match="alpha"):
        apply_corrections(comps, alpha=0.0)
