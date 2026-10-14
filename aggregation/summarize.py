"""
Group statistics (Phase 11).

GROUP LEVELS
  cell              (dataset, strategy, budget_pct)   THE matched-budget comparison unit. It collapses only
                    seeds. To compare strategies fairly, put cells that share dataset AND budget_pct side by side.
  dataset_strategy  (dataset, strategy)               descriptive rollup: collapses budgets.
  dataset_budget    (dataset, budget_pct)             descriptive rollup: collapses strategies.
  --group-by FIELDS custom levels over dataset, strategy, budget_pct, seed.

Quality metrics (EM / F1 / judge) are reported ONLY for groups that keep dataset, strategy and budget_pct
fixed. A rollup that mixes budgets or strategies has no matched comparison behind it, so averaging quality
over it would be misleading: its `metrics` is null and says why. Cost, token and runtime figures are
additive, so rollups report them (sum / mean), always next to how many records stand behind them.

STATISTICS RULES (see schemas.Stat)
  - Only INCLUDED conditions (completed or completed_with_failures) are summarised; aborted / failed ones are
    counted in n_conditions / n_excluded with their reasons.
  - A record contributes to sum / mean / min / max only if its measure is `measured`. Partial (`lower_bound`)
    records never enter the mean; they raise `sum_lower_bound` instead. `sum` is null unless every applicable
    record is measured. `std` is null with fewer than two values (never 0).
  - Ratios (normalised columns) have no sum.
"""

from __future__ import annotations

import math
import statistics

from aggregation.schemas import ConditionRecord, GroupSummary, MetricStat, Stat
from aggregation.measures import _r

GROUP_FIELDS = ("dataset", "strategy", "budget_pct", "seed", "shuffle_seed")
DEFAULT_LEVELS = (
    ("cell", ("dataset", "strategy", "budget_pct")),
    ("dataset_strategy", ("dataset", "strategy")),
    ("dataset_budget", ("dataset", "budget_pct")),
)
MATCHED = {"dataset", "strategy", "budget_pct"}
ADDITIVE_KINDS = ("cost", "spent", "tokens", "runtime", "elapsed")
COUNT_KEYS = ("n_chunks_total", "n_selected", "n_questions", "n_predictions", "n_question_failures",
              "extraction_n_failed", "extraction_n_cache_hits", "generation_n_failed", "n_answered", "n_abstained",
              "n_empty", "n_failed_outcomes", "judge_n_calls", "judge_n_failed")
METRIC_KEYS = ("em", "f1", "em_failures_as_wrong", "f1_failures_as_wrong", "judge_accuracy",
               # evidence recall (did the strategy pick / retrieve the gold chunks?) and Random's exact expectation
               "selection_recall", "selection_all_gold_rate", "retrieval_recall", "retrieval_all_gold_rate",
               "expected_random_selection_recall")


def parse_group_by(spec: str) -> tuple[str, tuple[str, ...]]:
    fields = tuple(f.strip() for f in spec.split(",") if f.strip())
    bad = [f for f in fields if f not in GROUP_FIELDS]
    if not fields or bad or len(set(fields)) != len(fields):
        raise ValueError(f"--group-by needs unique fields from {GROUP_FIELDS}; got {spec!r}")
    return "+".join(fields), fields


def _metric_or_evidence(r, key):
    """METRIC_KEYS mixes runner-computed metrics (em, f1, judge_accuracy, ...) with evidence-recall
    figures (selection_recall, ...), which live in separate ConditionRecord fields; metrics itself is
    left exactly as the runner wrote it."""
    return r.metrics[key] if key in r.metrics else r.evidence.get(key)


def stat_of(measures, *, additive: bool) -> Stat:
    n = len(measures)
    known = [m.value for m in measures if m.status == "measured"]
    lbs = [m.lower_bound for m in measures if m.status == "lower_bound" and m.lower_bound is not None]
    n_lb = sum(1 for m in measures if m.status == "lower_bound")
    n_unknown = sum(1 for m in measures if m.status in ("unknown", "not_run"))
    n_na = sum(1 for m in measures if m.status in ("disabled", "not_applicable"))
    complete = n > 0 and len(known) + n_na == n and len(known) > 0
    all_int = all(isinstance(v, int) for v in known + lbs)
    add = (lambda xs: sum(xs)) if all_int else (lambda xs: _r(math.fsum(xs)))
    return Stat(
        n_records=n, n_known=len(known), n_lower_bound=n_lb, n_unknown=n_unknown, n_not_applicable=n_na, complete=complete,
        sum=add(known) if (additive and complete) else None,
        sum_lower_bound=add(known + lbs) if (additive and (known or lbs)) else None,
        mean=_r(math.fsum(known) / len(known)) if known else None,
        min=min(known) if known else None, max=max(known) if known else None,
        std=_r(statistics.stdev(known)) if len(known) >= 2 else None,
    )


def metric_stat(values) -> MetricStat:
    known = [v for v in values if v is not None]
    return MetricStat(
        n_records=len(values), n_known=len(known), n_missing=len(values) - len(known),
        mean=_r(math.fsum(known) / len(known)) if known else None, min=min(known) if known else None,
        max=max(known) if known else None, std=_r(statistics.stdev(known)) if len(known) >= 2 else None)


def summarize_groups(records: list[ConditionRecord], levels=DEFAULT_LEVELS) -> list[GroupSummary]:
    out: list[GroupSummary] = []
    for level, fields in levels:
        groups: dict[tuple, list[ConditionRecord]] = {}
        for rec in records:
            groups.setdefault(tuple(getattr(rec, f) for f in fields), []).append(rec)
        for key in sorted(groups):
            out.append(_summarize_one(level, fields, key, groups[key]))
    return out


def _summarize_one(level, fields, key, members: list[ConditionRecord]) -> GroupSummary:
    inc = [r for r in members if r.included]
    exc = [r for r in members if not r.included]
    matched = MATCHED <= set(fields)
    reasons: dict[str, int] = {}
    for r in exc:
        reasons[r.status] = reasons.get(r.status, 0) + 1

    names = sorted({n for r in inc for n in r.measures})
    stats = {}
    for name in names:
        series = [r.measures[name] for r in inc if name in r.measures]
        kind = series[0].kind
        stats[name] = stat_of(series, additive=kind in ADDITIVE_KINDS)

    counts = {k: sum(r.counts[k] for r in inc if r.counts.get(k) is not None) for k in COUNT_KEYS}
    nr_reasons: list[str] = []
    for r in inc:
        for x in r.not_reportable_reasons:
            if x not in nr_reasons:
                nr_reasons.append(x)
    if exc:
        nr_reasons.append(f"{len(exc)} condition(s) in this group are excluded ({', '.join(f'{k}: {v}' for k, v in sorted(reasons.items()))})")
    if not inc:
        nr_reasons.append("no included conditions")
    if any(not r.reconciled for r in inc):
        nr_reasons.append("per-item cross-checks failed for some conditions (see issues)")

    if matched:
        metrics = {k: metric_stat([_metric_or_evidence(r, k) for r in inc]) for k in METRIC_KEYS}
        comparison, note = "matched_budget_cell", (
            "Same dataset, strategy and budget; only seeds (and Random's repeated shuffles) are collapsed: the "
            "mean and std over shuffles show Random's luck. Compare strategies by placing cells with "
            "the same dataset and budget_pct side by side.")
        omitted = None
    else:
        metrics = None
        comparison, note = "descriptive_rollup", (
            "Collapses " + " and ".join(x for x in ("budgets" if "budget_pct" not in fields else None,
                                                    "strategies" if "strategy" not in fields else None,
                                                    "datasets" if "dataset" not in fields else None) if x) +
            ". Descriptive only: cost / runtime / token sums are meaningful, a matched-budget comparison is not.")
        omitted = ("quality metrics are not aggregated here: averaging EM / F1 / judge across different budgets, "
                   "strategies or datasets is not a matched comparison")
    return GroupSummary(
        level=level, group_by=list(fields), comparison=comparison, comparison_note=note,
        key=dict(zip(fields, key)), n_conditions=len(members), n_included=len(inc), n_excluded=len(exc),
        excluded_reasons=dict(sorted(reasons.items())),
        statuses=dict(sorted({s: sum(1 for r in members if r.status == s) for s in {r.status for r in members}}.items())),
        seeds=sorted({r.seed for r in members}),
        shuffle_seeds=sorted({r.shuffle_seed for r in members if r.shuffle_seed is not None}),
        n_mock=sum(1 for r in inc if r.is_mock),
        n_reportable=sum(1 for r in inc if r.reportable), reportable=bool(inc) and not nr_reasons,
        not_reportable_reasons=nr_reasons, n_unreconciled=sum(1 for r in inc if not r.reconciled),
        config_fingerprints=sorted({r.shared_config_fingerprint for r in inc if r.shared_config_fingerprint}),
        condition_ids=sorted(r.condition_id for r in members), counts=counts,
        n_conditions_with_failures=sum(1 for r in inc if r.status == "completed_with_failures"),
        measures=stats, metrics=metrics, metrics_omitted_reason=omitted)
