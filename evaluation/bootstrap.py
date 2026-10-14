"""
Bootstrap confidence intervals over questions (paper: "Statistical Analysis").

200 questions per dataset give noisy averages, so every quality number gets a 95% interval:
resample the per-question scores with replacement many times, recompute the mean each time,
and take the 2.5th and 97.5th percentiles. A fixed RNG seed makes the interval reproducible.

paired_bootstrap compares two conditions that answered the SAME questions (e.g. two strategies
at one budget): it resamples questions, keeping each question's pair of scores together, so the
question-to-question difficulty cancels out.

Note: this captures uncertainty about WHICH questions were asked, not Random's luck in picking
chunks; the latter is covered by repeating Random with several shuffle seeds (random_seeds).
"""

from __future__ import annotations

import numpy as np

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 0


def bootstrap_ci(values, n_resamples: int = DEFAULT_RESAMPLES, level: float = 0.95,
                 seed: int = DEFAULT_SEED) -> dict | None:
    """{"mean", "low", "high", "n"} for the mean of `values`; None if there are no values."""
    x = np.asarray([float(v) for v in values], dtype=float)
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    alpha = (1 - level) / 2
    return {"mean": round(float(x.mean()), 6), "low": round(float(np.quantile(means, alpha)), 6),
            "high": round(float(np.quantile(means, 1 - alpha)), 6), "n": int(x.size),
            "level": level, "n_resamples": n_resamples}


def paired_bootstrap(a: dict, b: dict, n_resamples: int = DEFAULT_RESAMPLES, level: float = 0.95,
                     seed: int = DEFAULT_SEED) -> dict | None:
    """Difference mean(a) - mean(b) over the questions both have ({question_id: score}).
    Returns the difference, its interval and a two-sided bootstrap p-value; None if no overlap."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = np.asarray([float(a[q]) - float(b[q]) for q in common], dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    alpha = (1 - level) / 2
    p = 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {"diff": round(float(d.mean()), 6), "low": round(float(np.quantile(means, alpha)), 6),
            "high": round(float(np.quantile(means, 1 - alpha)), 6), "p_value": round(min(1.0, p), 6),
            "n_questions": int(d.size), "level": level, "n_resamples": n_resamples}


def condition_cis(scored) -> dict:
    """95% intervals for one condition's EM, F1 and judge accuracy, over the same records the
    headline numbers use (failed generations have em=None and are left out, as in the summary)."""
    em = [r.em for r in scored if r.em is not None]
    f1 = [r.f1 for r in scored if r.f1 is not None]
    judge = [1.0 if r.judge_correct else 0.0 for r in scored if r.judge_correct is not None]
    return {"em": bootstrap_ci(em), "f1": bootstrap_ci(f1), "judge_accuracy": bootstrap_ci(judge)}
