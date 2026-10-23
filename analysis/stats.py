"""
Statistics for the matched-budget comparison (Phase 12). numpy + scipy only; every random step is seeded
and the seeds are derived from the comparison's own key, so a result never depends on run order.

THE UNIT IS THE QUESTION (see analysis/random_avg.py). All procedures here take per-question values.

PAIRED SIGN-FLIP PERMUTATION TEST  (`sign_flip_test`)
  H0: within a question the two strategies are exchangeable, so each per-question difference d_q is
  equally likely to be +|d_q| or -|d_q| (the distribution of d is symmetric about 0). Statistic: the sum of d
  (equivalently the mean difference). Two-sided p = share of the 2^m sign patterns (m = questions with d != 0;
  zeros never change the sum) whose |sum| is at least the observed one.
    exact_enumeration  all 2^m patterns, m <= `exact_max_n` (default 16)
    exact_binomial     when every nonzero |d| is equal (e.g. binary outcomes: d in {-1, 0, 1}) the sum is
                       2K - m with K ~ Binomial(m, 1/2): exact for any m
    monte_carlo        otherwise: B random sign patterns, p = (1 + #{|T*| >= |T|}) / (B + 1) (the observed
                       pattern counts as one permutation, so p is never 0 and is valid)
    all_zero           every d is 0: p = 1
  Assumptions: questions are independent draws; the symmetry above. It does NOT assume normality and works
  for binary, continuous and shuffle-averaged scores alike. It tests "no difference in mean score on this
  question population", not a difference in any other feature.

EXACT McNEMAR  (`mcnemar_exact`)
  For single-run binary outcomes, on the discordant pairs (b: A right/B wrong, c: A wrong/B right) the exact
  two-sided binomial test of b vs c. IMPORTANT: this is the SAME test as the sign-flip test on those data
  (the sign-flip statistic is b - c, and under H0 b ~ Binomial(b + c, 1/2)), computed from the SAME paired
  observations by a different route. For binary data the two p-values are algebraically identical, so
  McNemar is a cross-check of the implementation and a way to report the discordant counts. It is NOT an
  independent confirmation of a finding.

CONFIDENCE INTERVALS  (`bootstrap_mean_ci`)
  Percentile bootstrap over QUESTIONS (resample the m paired differences with replacement). It captures
  uncertainty about which questions were asked. It does not capture LLM nondeterminism, the one question
  sample (seed), dependence between questions that share a pooled corpus, or, for a Random arm that
  averages a few shuffles, Random's own luck (see random_avg.ShuffleSummary). Approximate near 0 / 100 %.

MULTIPLE COMPARISONS
  Holm-Bonferroni (step-down) controls the family-wise error rate; it makes no independence assumption.
  Benjamini-Hochberg (step-up) controls the false-discovery rate under independence / positive dependence;
  it is reported as exploratory only. The FAMILY is chosen by the caller; p-values that are None (a test that
  could not be run) are excluded from the family size.

EFFECT SIZES  mean difference (percentage points when the metric is a rate), Cohen's d_z = mean(d)/sd(d)
  (a descriptive standardised effect; for binary outcomes it is not a probability), and the counts of
  questions where A is better / B is better / tied.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats as sps

from analysis.random_avg import ArmScores, PairedSample

DEFAULT_RESAMPLES = 10_000
DEFAULT_PERMUTATIONS = 20_000
DEFAULT_EXACT_MAX_N = 16
DEFAULT_LEVEL = 0.95
DEFAULT_ALPHA = 0.05
_BOOTSTRAP_BLOCK = 2_000


class StatsError(ValueError):
    """Invalid input to a statistical procedure."""


def _r(x) -> float:
    return round(float(x), 9)


# ============================================================================== seeds
def derive_seed(base_seed: int, *parts) -> int:
    """A reproducible 64-bit seed from a base seed and a key (SHA-256), independent of call order."""
    h = hashlib.sha256(("|".join([str(int(base_seed))] + [str(p) for p in parts])).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _array(x, name="values") -> np.ndarray:
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    if a.ndim != 1:
        raise StatsError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(a)):
        raise StatsError(f"{name} contains NaN or infinite values (missing data must be removed or scored first)")
    return a


# ============================================================================== bootstrap
class Interval(BaseModel):
    mean: float
    low: float
    high: float
    n: int
    level: float
    n_resamples: int
    degenerate: bool = False       # fewer than 2 values, or no variation: the interval has zero width by construction


def bootstrap_mean_ci(values, *, n_resamples: int = DEFAULT_RESAMPLES, level: float = DEFAULT_LEVEL,
                      seed: int = 0) -> Interval | None:
    """Percentile bootstrap CI of the mean over the items of `values` (None if there are none). Resampling is
    done in fixed blocks of _BOOTSTRAP_BLOCK replicates (bounds memory); the block size is a constant, not a
    parameter, because it fixes the random stream: the same seed always gives the same interval."""
    x = _array(values)
    if x.size == 0:
        return None
    if not (0 < level < 1) or n_resamples < 1:
        raise StatsError("level must be in (0, 1) and n_resamples at least 1")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, _BOOTSTRAP_BLOCK):
        stop = min(start + _BOOTSTRAP_BLOCK, n_resamples)
        means[start:stop] = x[rng.integers(0, x.size, size=(stop - start, x.size))].mean(axis=1)
    alpha = (1 - level) / 2
    degenerate = x.size < 2 or bool(np.all(x == x[0]))
    return Interval(mean=_r(x.mean()), low=_r(np.quantile(means, alpha)), high=_r(np.quantile(means, 1 - alpha)),
                    n=int(x.size), level=level, n_resamples=n_resamples, degenerate=degenerate)


# ============================================================================== sign-flip permutation test
class SignFlipResult(BaseModel):
    p_value: float
    method: str                    # all_zero | exact_binomial | exact_enumeration | monte_carlo
    n: int                         # all paired questions
    n_nonzero: int                 # questions with d != 0 (the only ones that matter)
    n_permutations: int | None = None     # Monte Carlo draws (None for exact methods)
    mean_diff: float


def _tol(abs_sum: float) -> float:
    return 1e-12 * max(1.0, abs_sum)


def _enumerate_p(dnz: np.ndarray) -> float:
    m = dnz.size
    bits = (np.arange(2 ** m)[:, None] >> np.arange(m)[None, :]) & 1
    sums = (2 * bits - 1) @ dnz
    obs = abs(float(dnz.sum()))
    return float(np.count_nonzero(np.abs(sums) >= obs - _tol(float(np.abs(dnz).sum()))) / sums.size)


def _binomial_p(n_pos: int, m: int) -> float:
    t = abs(2 * n_pos - m)
    k = np.arange(m + 1)
    pmf = sps.binom.pmf(k, m, 0.5)
    return float(min(1.0, pmf[np.abs(2 * k - m) >= t].sum()))


def sign_flip_test(d, *, n_permutations: int = DEFAULT_PERMUTATIONS, seed: int = 0,
                   exact_max_n: int = DEFAULT_EXACT_MAX_N, method: str = "auto") -> SignFlipResult:
    """Two-sided paired sign-flip permutation test on the per-question differences `d` (see module docstring).
    `method`: auto | enumerate | binomial | monte_carlo (the explicit ones exist so each can be validated against the
    others; `auto` picks the exact method whenever one applies)."""
    d = _array(d, "differences")
    if method not in ("auto", "enumerate", "binomial", "monte_carlo"):
        raise StatsError(f"unknown method {method!r}")
    n = int(d.size)
    nz = d[d != 0.0]
    m = int(nz.size)
    mean_diff = _r(d.mean()) if n else 0.0
    if m == 0:
        return SignFlipResult(p_value=1.0, method="all_zero", n=n, n_nonzero=0, mean_diff=mean_diff)
    equal_mag = bool(np.all(np.abs(np.abs(nz) - np.abs(nz[0])) <= 1e-12 * abs(nz[0])))
    if method == "binomial" and not equal_mag:
        raise StatsError("the binomial shortcut needs every nonzero |d| to be equal")
    if method == "enumerate" and m > 24:
        raise StatsError(f"exhaustive enumeration of 2^{m} sign patterns is not supported (limit 24)")
    use = method
    if method == "auto":
        use = "binomial" if equal_mag else ("enumerate" if m <= exact_max_n else "monte_carlo")

    if use == "binomial":
        return SignFlipResult(p_value=_r(_binomial_p(int((nz > 0).sum()), m)), method="exact_binomial", n=n,
                              n_nonzero=m, mean_diff=mean_diff)
    if use == "enumerate":
        return SignFlipResult(p_value=_r(_enumerate_p(nz)), method="exact_enumeration", n=n, n_nonzero=m,
                              mean_diff=mean_diff)
    if n_permutations < 1:
        raise StatsError("n_permutations must be at least 1")
    rng = np.random.default_rng(seed)
    obs = abs(float(nz.sum()))
    tol = _tol(float(np.abs(nz).sum()))
    count, step = 0, max(1, min(n_permutations, 4_000_000 // m))
    for start in range(0, n_permutations, step):
        k = min(step, n_permutations - start)
        signs = rng.integers(0, 2, size=(k, m)) * 2 - 1
        count += int(np.count_nonzero(np.abs(signs @ nz) >= obs - tol))
    return SignFlipResult(p_value=_r((1 + count) / (n_permutations + 1)), method="monte_carlo", n=n, n_nonzero=m,
                          n_permutations=n_permutations, mean_diff=mean_diff)


# ============================================================================== McNemar (exact)
class McNemarResult(BaseModel):
    n_pairs: int
    n_both_correct: int            # A = 1, B = 1
    n_both_wrong: int              # A = 0, B = 0
    n_a_only: int                  # b: A = 1, B = 0
    n_b_only: int                  # c: A = 0, B = 1
    p_value: float
    method: str = "exact_binomial_two_sided"
    note: str = ("Computed from the same paired observations as the sign-flip test and, for binary data, "
                 "algebraically identical to it: a consistency check, not independent confirmation.")


def mcnemar_exact(a, b) -> McNemarResult:
    """Exact two-sided McNemar test for paired BINARY outcomes (0/1 arrays of equal length)."""
    a, b = _array(a, "a"), _array(b, "b")
    if a.size != b.size:
        raise StatsError("a and b must have the same length")
    if not (np.all(np.isin(a, (0.0, 1.0))) and np.all(np.isin(b, (0.0, 1.0)))):
        raise StatsError("McNemar's test needs binary (0/1) outcomes; shuffle-averaged or continuous scores are not binary")
    n11, n00 = int(((a == 1) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    disc = n10 + n01
    p = 1.0 if disc == 0 else float(sps.binomtest(n10, disc, 0.5, alternative="two-sided").pvalue)
    return McNemarResult(n_pairs=int(a.size), n_both_correct=n11, n_both_wrong=n00, n_a_only=n10, n_b_only=n01,
                         p_value=_r(min(1.0, p)))


# ============================================================================== multiple comparisons
def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values, in the input order. p-values that are None are not part of the
    family (m counts the others only) and come back as None."""
    idx = [i for i, p in enumerate(p_values) if p is not None]
    for i in idx:
        if not (0.0 <= p_values[i] <= 1.0):
            raise StatsError(f"p-value {p_values[i]!r} is outside [0, 1]")
    m = len(idx)
    out: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: p_values[j]), start=1):
        running = max(running, (m - rank + 1) * p_values[i])
        out[i] = _r(min(1.0, running))
    return out


def bh_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order; None excluded as above."""
    idx = [i for i, p in enumerate(p_values) if p is not None]
    for i in idx:
        if not (0.0 <= p_values[i] <= 1.0):
            raise StatsError(f"p-value {p_values[i]!r} is outside [0, 1]")
    m = len(idx)
    out: list[float | None] = [None] * len(p_values)
    running = 1.0
    for rank, i in reversed(list(enumerate(sorted(idx, key=lambda j: p_values[j]), start=1))):
        running = min(running, m / rank * p_values[i])
        out[i] = _r(min(1.0, running))
    return out
    # (sorted() is stable, so tied p-values keep input order; their adjusted values are equal anyway)


# ============================================================================== effect sizes
class PairedEffect(BaseModel):
    n: int
    mean_a: float | None
    mean_b: float | None
    mean_diff: float | None        # A - B
    sd_diff: float | None          # sample SD (n - 1); None for n < 2
    dz: float | None               # mean_diff / sd_diff; None if sd_diff is None or 0
    n_a_better: int
    n_b_better: int
    n_tied: int


def paired_effect(a, b) -> PairedEffect:
    a, b = _array(a, "a"), _array(b, "b")
    if a.size != b.size:
        raise StatsError("a and b must have the same length")
    d = a - b
    n = int(d.size)
    if n == 0:
        return PairedEffect(n=0, mean_a=None, mean_b=None, mean_diff=None, sd_diff=None, dz=None,
                            n_a_better=0, n_b_better=0, n_tied=0)
    sd = float(d.std(ddof=1)) if n >= 2 else None
    return PairedEffect(n=n, mean_a=_r(a.mean()), mean_b=_r(b.mean()), mean_diff=_r(d.mean()),
                        sd_diff=None if sd is None else _r(sd), dz=_r(d.mean() / sd) if sd else None,
                        n_a_better=int((d > 0).sum()), n_b_better=int((d < 0).sum()), n_tied=int((d == 0).sum()))


# ============================================================================== one arm, described
class ArmDescription(BaseModel):
    dataset: str
    strategy: str
    budget_pct: int
    metric: str
    n: int                         # questions with a value (never questions x shuffles)
    n_universe: int
    n_missing: int
    missing_reasons: dict[str, int] = Field(default_factory=dict)
    n_replicates: int
    n_partial: int
    mean: float | None
    ci: Interval | None


def describe_arm(arm: ArmScores, *, n_resamples: int = DEFAULT_RESAMPLES, level: float = DEFAULT_LEVEL,
                 base_seed: int = 0) -> ArmDescription:
    vals = [arm.values[q] for q in arm.universe if q in arm.values]
    seed = derive_seed(base_seed, "describe", arm.dataset, arm.strategy, arm.budget_pct, arm.metric,
                       "strict" if arm.strict else "all", ",".join(str(s) for s in arm.replicates))
    ci = bootstrap_mean_ci(vals, n_resamples=n_resamples, level=level, seed=seed)
    return ArmDescription(dataset=arm.dataset, strategy=arm.strategy, budget_pct=arm.budget_pct, metric=arm.metric,
                          n=arm.n, n_universe=arm.n_universe, n_missing=arm.n_missing,
                          missing_reasons=arm.missing_reasons(), n_replicates=arm.n_replicates,
                          n_partial=arm.n_partial, mean=None if ci is None else ci.mean, ci=ci)


# ============================================================================== one paired comparison
class PairedComparison(BaseModel):
    dataset: str
    budget_pct: int
    metric: str
    strategy_a: str
    strategy_b: str
    missing_policy: str
    n_universe: int
    n_pairs: int
    n_missing_a: int
    n_missing_b: int
    n_dropped: int
    effect: PairedEffect
    ci: Interval | None                          # paired bootstrap CI of mean(A) - mean(B)
    sign_flip: SignFlipResult
    mcnemar: McNemarResult | None = None         # only for single-run binary arms under complete_case
    identical: bool = False                      # every paired difference is exactly 0
    flags: list[str] = Field(default_factory=list)
    p_holm: float | None = None                  # filled by apply_corrections
    p_bh: float | None = None
    significant_holm: bool | None = None


def compare_paired(sample: PairedSample, *, n_resamples: int = DEFAULT_RESAMPLES,
                   n_permutations: int = DEFAULT_PERMUTATIONS, level: float = DEFAULT_LEVEL, base_seed: int = 0,
                   exact_max_n: int = DEFAULT_EXACT_MAX_N) -> PairedComparison:
    """Mean difference A - B with CI, sign-flip p-value, effect sizes and (when eligible) McNemar, on the paired questions."""
    if sample.n_pairs == 0:
        raise StatsError(f"no paired questions for {sample.dataset}/{sample.budget_pct}%/{sample.metric} "
                         f"{sample.strategy_a} vs {sample.strategy_b}: nothing to compare")
    key = (sample.dataset, sample.budget_pct, sample.metric, sample.strategy_a, sample.strategy_b, sample.missing_policy)
    d = sample.d
    ci = bootstrap_mean_ci(d, n_resamples=n_resamples, level=level, seed=derive_seed(base_seed, "bootstrap", *key))
    sf = sign_flip_test(d, n_permutations=n_permutations, seed=derive_seed(base_seed, "permutation", *key),
                        exact_max_n=exact_max_n)
    mc = mcnemar_exact(sample.a, sample.b) if sample.both_deterministic_binary else None
    flags = []
    if sample.n_pairs < 2:
        flags.append("fewer_than_2_pairs")
    if sf.method == "all_zero":
        flags.append("all_differences_zero")
    if sample.n_dropped:
        flags.append(f"{sample.n_dropped}_questions_dropped_for_missing_values")
    if sample.n_partial_a or sample.n_partial_b:
        flags.append("some_values_average_fewer_shuffles_than_planned")
    if len(sample.replicates_a) > 1 or len(sample.replicates_b) > 1:
        flags.append("shuffle_averaged_arm")
    return PairedComparison(
        dataset=sample.dataset, budget_pct=sample.budget_pct, metric=sample.metric, strategy_a=sample.strategy_a,
        strategy_b=sample.strategy_b, missing_policy=sample.missing_policy, n_universe=sample.n_universe,
        n_pairs=sample.n_pairs, n_missing_a=sample.n_missing_a, n_missing_b=sample.n_missing_b,
        n_dropped=sample.n_dropped, effect=paired_effect(sample.a, sample.b), ci=ci, sign_flip=sf, mcnemar=mc,
        identical=sf.method == "all_zero", flags=flags)


def apply_corrections(comparisons: Sequence[PairedComparison], *, alpha: float = DEFAULT_ALPHA) -> list[PairedComparison]:
    """Copies of `comparisons` with Holm- and BH-adjusted p-values over THIS list as one family. The caller must
    pass exactly one pre-specified family (e.g. all pairwise comparisons for one dataset and metric)."""
    if not (0 < alpha < 1):
        raise StatsError("alpha must be in (0, 1)")
    raw = [c.sign_flip.p_value for c in comparisons]
    holm, bh = holm_adjust(raw), bh_adjust(raw)
    return [c.model_copy(update={"p_holm": h, "p_bh": q, "significant_holm": None if h is None else h <= alpha})
            for c, h, q in zip(comparisons, holm, bh)]
