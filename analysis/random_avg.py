"""
Per-question scores of one strategy at one (dataset, budget, metric), and pairing of two strategies (Phase 12).

THE UNIT IS THE QUESTION. An "arm" has AT MOST ONE value per question, however many times the strategy
was run. Random is run once per shuffle seed on the SAME questions; its arm value for a question is the
MEAN of that question's score over the shuffles that have a measured score. Deterministic strategies are
run once, so their value is that single score. Consequences, all tested:

  * the number of values in an arm is the number of questions, never questions x shuffles;
  * adding shuffles (or repeating identical ones) cannot raise n, only change the averaged values;
  * a question missing in SOME shuffles is averaged over the others and counted in `k_available`
    (so `n_partial` shows how many questions rest on fewer shuffles than planned);
  * a question missing in ALL shuffles has no value and a reason; with `strict=True` a question that is
    missing in ANY shuffle has no value (reason "incomplete_shuffles"): a sensitivity analysis.

MISSING IS NOT ZERO. `pair_arms(..., missing="complete_case")` (the primary analysis) pairs only the
questions where both arms have a value and reports how many were dropped; `missing="as_wrong"` (the
sensitivity analysis) keeps every question of the universe and scores a missing value 0.

`deterministic_binary` marks arms whose values are single 0/1 observations (EM, judge accuracy of a
strategy that ran once). Only such pairs are eligible for McNemar's test; a shuffle average is not binary.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

from analysis.schemas import QuestionObservation, metric_value

NO_OBSERVATION = "no_observation"          # a replicate has no row at all for the question
INCOMPLETE_SHUFFLES = "incomplete_shuffles"  # strict mode: some shuffles missing
MissingPolicy = Literal["complete_case", "as_wrong"]


class ArmError(ValueError):
    """The observations cannot form a valid arm / pair (an inconsistency, not a missing measurement)."""


def _r(x: float) -> float:
    return round(float(x), 9)


# ============================================================================== one arm
@dataclass(frozen=True)
class ArmScores:
    dataset: str
    strategy: str
    budget_pct: int
    metric: str
    universe: tuple[str, ...]
    values: dict[str, float]                 # question_id -> value: ONE per question that has any
    missing: dict[str, str]                  # question_id -> reason (the questions without a value)
    replicates: tuple[int | None, ...]       # the shuffle seeds used ((None,) for a single run)
    k_available: dict[str, int]              # question_id -> replicates that contributed a value
    strict: bool = False

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def n_universe(self) -> int:
        return len(self.universe)

    @property
    def n_missing(self) -> int:
        return len(self.missing)

    @property
    def n_replicates(self) -> int:
        return len(self.replicates)

    @property
    def n_partial(self) -> int:
        """Questions whose value rests on fewer replicates than planned (only meaningful when n_replicates > 1)."""
        return sum(1 for k in self.k_available.values() if 0 < k < self.n_replicates)

    @property
    def deterministic_binary(self) -> bool:
        """Single-run 0/1 observations: eligible for McNemar's test."""
        return self.n_replicates == 1 and self.n > 0 and all(v in (0.0, 1.0) for v in self.values.values())

    def missing_reasons(self) -> dict[str, int]:
        return dict(sorted(Counter(self.missing.values()).items()))


def build_arm(observations: Iterable[QuestionObservation], *, dataset: str, strategy: str, budget_pct: int,
              metric: str, universe: Sequence[str], strict: bool = False,
              shuffles: Sequence[int] | None = None) -> ArmScores:
    """The arm of `strategy` at (dataset, budget_pct) for `metric`. `universe` is the dataset's question set
    (AnalysisInput.question_ids). `shuffles` restricts Random to the given shuffle seeds (a sensitivity analysis)."""
    universe = tuple(universe)
    uni = set(universe)
    if len(uni) != len(universe):
        raise ArmError("the question universe has duplicate ids")
    cell = [o for o in observations if o.dataset == dataset and o.strategy == strategy and o.budget_pct == budget_pct]
    if not cell:
        raise ArmError(f"no observations for {dataset}/{strategy}/{budget_pct}%")

    ids = sorted({o.shuffle_seed for o in cell}, key=lambda s: (s is not None, s))
    if strategy != "random" and ids != [None]:
        raise ArmError(f"{strategy} is deterministic but has shuffle seeds {ids}; only Random is repeated")
    if None in ids and len(ids) > 1:
        raise ArmError(f"{strategy} mixes runs with and without a shuffle seed ({ids})")
    if shuffles is not None:
        if strategy != "random":
            raise ArmError("`shuffles` applies to Random only")
        ids = [s for s in ids if s in set(shuffles)]
        if not ids:
            raise ArmError(f"none of the requested shuffles {sorted(set(shuffles))} exist for {dataset}/random/{budget_pct}%")
        cell = [o for o in cell if o.shuffle_seed in set(ids)]

    cells: dict[tuple, QuestionObservation] = {}
    for o in cell:
        if o.question_id not in uni:
            raise ArmError(f"{o.condition_id}: question {o.question_id!r} is not in the dataset's question set")
        k = (o.shuffle_seed, o.question_id)
        if k in cells:
            raise ArmError(f"{o.condition_id}: question {o.question_id!r} appears twice for shuffle {o.shuffle_seed}")
        cells[k] = o

    values, missing, k_avail = {}, {}, {}
    for q in universe:
        got, reasons = [], []
        for sid in ids:
            o = cells.get((sid, q))
            if o is None:
                reasons.append(NO_OBSERVATION)
                continue
            v, why = metric_value(o, metric)
            (got if v is not None else reasons).append(v if v is not None else why)
        k_avail[q] = len(got)
        if got and not (strict and reasons):
            values[q] = math.fsum(got) / len(got)             # one value per question, however many replicates
        else:
            missing[q] = INCOMPLETE_SHUFFLES if got else "|".join(sorted(set(reasons)))
    return ArmScores(dataset=dataset, strategy=strategy, budget_pct=budget_pct, metric=metric, universe=universe,
                     values=values, missing=missing, replicates=tuple(ids), k_available=k_avail, strict=strict)


# ============================================================================== pairing
@dataclass(frozen=True)
class PairedSample:
    dataset: str
    budget_pct: int
    metric: str
    strategy_a: str
    strategy_b: str
    missing_policy: str
    question_ids: tuple[str, ...]
    a: np.ndarray = field(repr=False)
    b: np.ndarray = field(repr=False)
    n_universe: int = 0
    n_missing_a: int = 0                     # universe questions where arm A has no value
    n_missing_b: int = 0
    n_dropped: int = 0                       # universe questions NOT in the sample (complete_case only)
    missing_reasons_a: dict = field(default_factory=dict)
    missing_reasons_b: dict = field(default_factory=dict)
    replicates_a: tuple = (None,)
    replicates_b: tuple = (None,)
    n_partial_a: int = 0
    n_partial_b: int = 0
    both_deterministic_binary: bool = False

    @property
    def n_pairs(self) -> int:
        return len(self.question_ids)

    @property
    def d(self) -> np.ndarray:
        """Per-question differences A - B."""
        return self.a - self.b


def pair_arms(a: ArmScores, b: ArmScores, *, missing: MissingPolicy = "complete_case") -> PairedSample:
    """Pair two arms of one cell question by question. The sample size is the number of QUESTIONS paired."""
    if (a.dataset, a.budget_pct, a.metric) != (b.dataset, b.budget_pct, b.metric):
        raise ArmError("arms must share dataset, budget and metric to be paired")
    if a.universe != b.universe:
        raise ArmError("arms must share one question universe to be paired")
    if a.strategy == b.strategy:
        raise ArmError("cannot pair an arm with itself")
    if missing not in ("complete_case", "as_wrong"):
        raise ArmError(f"unknown missing-data policy {missing!r}")
    if missing == "complete_case":
        ids = tuple(q for q in a.universe if q in a.values and q in b.values)
        av, bv = [a.values[q] for q in ids], [b.values[q] for q in ids]
    else:
        ids = a.universe
        av, bv = [a.values.get(q, 0.0) for q in ids], [b.values.get(q, 0.0) for q in ids]
    return PairedSample(
        dataset=a.dataset, budget_pct=a.budget_pct, metric=a.metric, strategy_a=a.strategy, strategy_b=b.strategy,
        missing_policy=missing, question_ids=tuple(ids), a=np.asarray(av, dtype=float), b=np.asarray(bv, dtype=float),
        n_universe=a.n_universe, n_missing_a=a.n_missing, n_missing_b=b.n_missing, n_dropped=a.n_universe - len(ids),
        missing_reasons_a=a.missing_reasons(), missing_reasons_b=b.missing_reasons(),
        replicates_a=a.replicates, replicates_b=b.replicates, n_partial_a=a.n_partial, n_partial_b=b.n_partial,
        both_deterministic_binary=a.deterministic_binary and b.deterministic_binary and missing == "complete_case")


# ============================================================================== Random's own variability
@dataclass(frozen=True)
class ShuffleSummary:
    """How much Random's luck moves the condition-level mean at one (dataset, budget, metric). With few
    shuffles the SD is a crude estimate (n_shuffles is reported so it cannot be mistaken for a precise one)."""
    dataset: str
    budget_pct: int
    metric: str
    per_shuffle: tuple[tuple[int, float | None, int], ...]   # (shuffle_seed, mean over measured questions, n measured)
    n_shuffles: int
    mean: float | None
    sd: float | None             # sample SD (n-1) of the per-shuffle means; None with fewer than 2 shuffles
    min: float | None
    max: float | None


def shuffle_summary(observations: Iterable[QuestionObservation], *, dataset: str, budget_pct: int, metric: str,
                    universe: Sequence[str]) -> ShuffleSummary:
    obs = list(observations)
    shuffle_ids = sorted({o.shuffle_seed for o in obs if o.dataset == dataset and o.strategy == "random"
                          and o.budget_pct == budget_pct and o.shuffle_seed is not None})
    if not shuffle_ids:
        raise ArmError(f"no Random shuffles for {dataset}/{budget_pct}% (the matrix has no random_seeds)")
    per = []
    for sid in shuffle_ids:
        arm = build_arm(obs, dataset=dataset, strategy="random", budget_pct=budget_pct, metric=metric,
                        universe=universe, shuffles=[sid])
        per.append((sid, _r(math.fsum(arm.values.values()) / arm.n) if arm.n else None, arm.n))
    means = [m for _s, m, _n in per if m is not None]
    return ShuffleSummary(
        dataset=dataset, budget_pct=budget_pct, metric=metric, per_shuffle=tuple(per), n_shuffles=len(per),
        mean=_r(math.fsum(means) / len(means)) if means else None,
        sd=_r(statistics.stdev(means)) if len(means) >= 2 else None,
        min=min(means) if means else None, max=max(means) if means else None)
