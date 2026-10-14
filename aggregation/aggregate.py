"""
Aggregate the Phase 10 results: cost, tokens and runtime per condition and per group (Phase 11).

    # default: every group level, written next to the results
    python -m aggregation.aggregate --matrix-dir results/experiments/benchmark_mock
    python -m aggregation.aggregate --config configs/benchmark_mock.yaml          # same, found via the config
    python -m aggregation.aggregate --matrix-dir DIR --group-by dataset,strategy,budget_pct,seed

Input : <matrix_dir>/conditions/*/condition_result.json (+ the per-item files next to them) and matrix.json.
Output: <matrix_dir>/aggregate/   (or --out-dir; required with several --matrix-dir)
    conditions.jsonl   one flat row per condition (cost / tokens / runtime Measures, metrics, counts,
                       failures, config): the table Phase 12 reads
    conditions.csv     the same, flattened (<measure> and <measure>__status columns; empty cell = no value)
    aggregate.json     groups (cell / dataset_strategy / dataset_budget / custom), excluded conditions,
                       duplicates, issues, reportability, methodology

NOTHING is invented: a missing measurement is null with a status and a reason, never 0. Aborted / failed /
missing conditions stay visible with their reasons. Mock results are marked and never reportable.
Duplicate cells and mixed configurations are errors unless you opt in (--on-duplicate latest,
--allow-mixed-config), and the report then says so.

Exit codes: 0 everything planned was aggregated cleanly | 1 aggregated with problems (excluded or missing
conditions, issues) | 2 stopped (bad input, duplicates, mixed configs, nothing to aggregate).
Phase 12 should call assert_reportable(report) before presenting any number as a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from aggregation.loader import AggregationError, check_configs, discover, resolve_duplicates
from aggregation.measures import build_record
from aggregation.schemas import AggregateReport, ConditionRecord, ExcludedCondition, Issue
from aggregation.summarize import DEFAULT_LEVELS, parse_group_by, summarize_groups
from evaluation.schemas import write_jsonl
from evaluation.score import NotReportableError

METHODOLOGY = {
    "measure_status": {
        "measured": "complete; value is set (0.0 here is a measured zero)",
        "lower_bound": "partial; lower_bound is the known part, value is null",
        "unknown": "cannot be reconstructed (see reason); value is null",
        "not_run": "the stage never ran; value is null",
        "disabled": "switched off on purpose (judge); value is null",
        "not_applicable": "no valid denominator / coverage; value is null",
    },
    "cost_usd": "cost_<stage>_usd = the earlier phases' cost_if_uncached (cache hits at their original cost): the "
                "comparable figure. cost_total_usd = extraction + generation + judge (judge only if enabled); it is "
                "measured only if every required stage is measured, otherwise a lower bound or unknown.",
    "spent_usd": "spent_*: money paid in THIS run. Depends on cache state and run order: cache_dependent, not for "
                 "comparing conditions.",
    "tokens": "rebuilt from the per-item files (extractions.jsonl, predictions.jsonl, scored.jsonl), which keep the "
              "original usage on cache hits. Unknown usage makes the figure a lower bound, never zero.",
    "runtime_intrinsic": "runtime_total_s = ranking (original time) + selection + extraction call runtimes + graph + "
                         "retrieval + generation call runtimes + judge call runtimes. Call runtimes are the original "
                         "per-call times, kept on cache hits. Excludes corpus loading, EM/F1 scoring and file writing.",
    "runtime_this_run": "wall_seconds and elapsed_<stage>_s: what this run measured. cache_dependent. A stage that did "
                        "not run is not_run, not 0.",
    "normalised": "per selected chunk divides by n_selected; per question by n_questions, only when every question "
                  "has a prediction. No other denominators are used.",
    "statistics": "per group, over INCLUDED conditions: mean / min / max over measured values only; sum only if every "
                  "applicable record is measured; sum_lower_bound otherwise; std needs two values. Counts "
                  "(n_records, n_known, ...) accompany every statistic.",
    "groups": "cell (dataset, strategy, budget_pct) is the matched-budget comparison unit; other levels are "
              "descriptive rollups and carry no aggregated quality metrics.",
}


# ------------------------------------------------------------------------------ the aggregation
def aggregate_matrices(matrix_dirs, *, on_duplicate: str = "error", allow_mixed_config: bool = False,
                       group_by: list[str] | None = None):
    """Aggregate one or more matrix directories. Returns (AggregateReport, [ConditionRecord]).
    Raises AggregationError if the input cannot be aggregated honestly."""
    matrix_dirs = [Path(d) for d in matrix_dirs]
    levels = list(DEFAULT_LEVELS) + [parse_group_by(g) for g in (group_by or [])]
    disc = discover(matrix_dirs)
    kept, dropped, duplicates = resolve_duplicates(disc.conditions, on_duplicate)
    fingerprints, mixed, differences = check_configs(kept, allow_mixed_config)

    records = sorted((build_record(lc) for lc in kept),
                     key=lambda r: (r.dataset, r.strategy, r.budget_pct, r.seed,
                                    -1 if r.shuffle_seed is None else r.shuffle_seed, r.matrix_id))
    included = [r for r in records if r.included]

    excluded = list(disc.excluded) + dropped
    for r in records:
        if not r.included:
            excluded.append(ExcludedCondition(
                kind="aborted" if r.status == "aborted" else "failed", condition_id=r.condition_id, matrix_id=r.matrix_id,
                dataset=r.dataset, strategy=r.strategy, budget_pct=r.budget_pct, seed=r.seed,
                shuffle_seed=r.shuffle_seed, status=r.status,
                reason=r.exclusion_reason or r.status))
    issues: list[Issue] = list(disc.issues) + [i for r in records for i in r.issues]

    planned = disc.planned_cells
    included_cells = {(r.dataset, r.strategy, r.budget_pct, r.seed, r.shuffle_seed) for r in included}
    complete = None if planned is None else planned <= included_cells

    reasons: list[str] = []
    if not included:
        reasons.append("no included conditions")
    for r in included:
        for x in r.not_reportable_reasons:
            if x not in reasons:
                reasons.append(x)
    if planned is None:
        reasons.append("the planned conditions are unknown (no matrix.json): completeness cannot be verified")
    elif not complete:
        reasons.append(f"matrix incomplete: {len(planned - included_cells)} of {len(planned)} planned condition(s) "
                       f"have no included result")
    if mixed:
        reasons.append("conditions with different configurations were combined (--allow-mixed-config)")
    if duplicates:
        reasons.append("duplicate results were resolved with --on-duplicate latest")
    if any(not r.reconciled for r in included):
        reasons.append("per-item cross-checks failed for some conditions (see issues)")

    n_errors = sum(1 for i in issues if i.severity == "error")
    report = AggregateReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        matrix_dirs=[str(d) for d in matrix_dirs], matrix_ids=disc.matrix_ids,
        options={"on_duplicate": on_duplicate, "allow_mixed_config": allow_mixed_config,
                 "levels": [{"level": name, "group_by": list(fields)} for name, fields in levels]},
        n_conditions_loaded=len(records), n_included=len(included), n_excluded=len(excluded),
        n_planned=None if planned is None else len(planned), matrix_complete=complete,
        statuses=dict(sorted({s: sum(1 for r in records if r.status == s) for s in {r.status for r in records}}.items())),
        is_mock=any(r.is_mock for r in included), n_mock=sum(1 for r in included if r.is_mock),
        n_reportable_conditions=sum(1 for r in included if r.reportable),
        reportable=not reasons, not_reportable_reasons=reasons,
        shared_config_fingerprints=fingerprints, mixed_config=mixed, config_differences=differences,
        excluded=excluded, duplicates=duplicates, issues=issues, n_errors=n_errors, n_warnings=len(issues) - n_errors,
        methodology=METHODOLOGY, groups=summarize_groups(records, levels))
    return report, records


def assert_reportable(report, *, require_reconciled: bool = True) -> None:
    """Call before presenting aggregate numbers as results (Phase 12). Accepts an AggregateReport or the dict
    loaded from aggregate.json."""
    d = report.model_dump(mode="json") if isinstance(report, AggregateReport) else report
    reasons = list(d.get("not_reportable_reasons", [])) if not d.get("reportable", False) else []
    if not d.get("reportable", False) and not reasons:
        reasons = ["not reportable"]
    if require_reconciled and d.get("n_errors", 0):
        reasons.append(f"{d['n_errors']} unresolved error issue(s) (see issues)")
    if reasons:
        raise NotReportableError("these aggregates are not reportable: " + "; ".join(reasons))


# ------------------------------------------------------------------------------ outputs
def _flat_value(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return v


def csv_rows(records: list[ConditionRecord]):
    """(header, rows): scalar identity / outcome / count / metric columns, then <measure> and
    <measure>__status for every measure. An empty cell means no value (see the __status column)."""
    ident = ["condition_id", "matrix_id", "run_id", "dataset", "data_source", "strategy", "budget", "budget_pct", "seed",
             "shuffle_seed",
             "corpus_id", "config_hash", "shared_config_fingerprint", "status", "included", "exclusion_reason",
             "is_mock", "reportable", "reconciled", "judge_enabled"]
    count_keys = sorted({k for r in records for k in r.counts})
    metric_keys = sorted({k for r in records for k, v in r.metrics.items() if not isinstance(v, (dict, list))})
    evidence_keys = sorted({k for r in records for k, v in r.evidence.items() if not isinstance(v, (dict, list))})
    ci_keys = sorted({name for r in records for name, ci in r.ci95.items() if ci})
    measure_keys = list(records[0].measures) if records else []
    header = (ident + ["n_issues", "not_reportable_reasons"] + count_keys + [f"metric_{k}" for k in metric_keys]
             + [f"evidence_{k}" for k in evidence_keys])
    for name in ci_keys:
        header += [f"{name}_ci95_low", f"{name}_ci95_high"]
    for k in measure_keys:
        header += [k, f"{k}__status"]
    rows = []
    for r in records:
        row = [_flat_value(getattr(r, k)) for k in ident]
        row += [len(r.issues), " | ".join(r.not_reportable_reasons)]
        row += [_flat_value(r.counts.get(k)) for k in count_keys]
        row += [_flat_value(r.metrics.get(k)) for k in metric_keys]
        row += [_flat_value(r.evidence.get(k)) for k in evidence_keys]
        for name in ci_keys:
            ci = r.ci95.get(name)
            row += [_flat_value(ci["low"]), _flat_value(ci["high"])] if ci else ["", ""]
        for k in measure_keys:
            m = r.measures.get(k)
            row += ["", ""] if m is None else [_flat_value(m.value), m.status]
        rows.append(row)
    return header, rows


def write_outputs(out_dir: str | Path, report: AggregateReport, records: list[ConditionRecord]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "conditions.jsonl", records)
    header, rows = csv_rows(records)
    tmp = out_dir / f"conditions.csv.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, out_dir / "conditions.csv")
    tmp = out_dir / f"aggregate.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_dir / "aggregate.json")
    return out_dir


# ------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", action="append", default=[], help="a Phase 10 matrix directory (repeatable)")
    ap.add_argument("--config", help="config YAML: aggregates <output_dir>/experiments/<experiment_id>")
    ap.add_argument("--out-dir", help="where to write the outputs (default: <matrix_dir>/aggregate)")
    ap.add_argument("--on-duplicate", choices=("error", "latest"), default="error",
                    help="two results for one cell: error (default) or keep the latest finished")
    ap.add_argument("--allow-mixed-config", action="store_true",
                    help="aggregate conditions with different configs anyway (result is marked not reportable)")
    ap.add_argument("--group-by", action="append", default=[], metavar="FIELDS",
                    help="extra group level, comma-separated from dataset,strategy,budget_pct,seed (repeatable)")
    args = ap.parse_args(argv)

    try:
        dirs = [Path(d) for d in args.matrix_dir]
        if args.config:
            from experiments.conditions import load_matrix
            from experiments.run import _resolve
            cfg, _spec = load_matrix(args.config)
            dirs.append(_resolve(cfg.output_dir) / "experiments" / cfg.experiment_id)
        if not dirs:
            raise AggregationError("give --matrix-dir DIR or --config CONFIG")
        if args.out_dir is None and len(dirs) != 1:
            raise AggregationError("--out-dir is required when aggregating several matrix directories")
        report, records = aggregate_matrices(dirs, on_duplicate=args.on_duplicate,
                                             allow_mixed_config=args.allow_mixed_config, group_by=args.group_by)
    except (AggregationError, ValueError, FileNotFoundError) as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2

    out_dir = write_outputs(Path(args.out_dir) if args.out_dir else dirs[0] / "aggregate", report, records)
    print(f"aggregated {report.n_conditions_loaded} condition(s): included {report.n_included}, excluded {report.n_excluded}"
          f"  statuses {report.statuses}")
    print(f"output: {out_dir}  (conditions.jsonl, conditions.csv, aggregate.json)")
    total = [r.measures["cost_total_usd"] for r in records if r.included]
    print(f"  cost_total_usd per included condition: "
          f"{sum(1 for m in total if m.status == 'measured')} measured, "
          f"{sum(1 for m in total if m.status != 'measured')} not measured")
    if report.is_mock:
        print("  MOCK RESULTS - testing only, NOT reportable.")
    elif not report.reportable:
        print("  NOT REPORTABLE: " + "; ".join(report.not_reportable_reasons), file=sys.stderr)
    for e in report.excluded:
        print(f"  EXCLUDED {e.condition_id}: {e.kind}: {e.reason[:160]}", file=sys.stderr)
    for i in report.issues:
        print(f"  {i.severity.upper()} {i.code}{' [' + i.condition_id + ']' if i.condition_id else ''}: {i.message}", file=sys.stderr)
    problems = report.n_excluded or report.issues or report.matrix_complete is False
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
