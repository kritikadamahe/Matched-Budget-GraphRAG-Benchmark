"""
Run the matched-budget experiment matrix (Phase 10).

    # 1. PLAN (free: no API call, no key needed): lists the 48 conditions, counts, estimated cost
    python -m experiments.run --config configs/benchmark_mock.yaml --dry-run

    # 2. A small, FREE end-to-end run on the bundled mock data (mock backends, never reportable)
    python -m experiments.run --config configs/benchmark_mock.yaml --datasets hotpotqa --budgets 25,100

    # 3. A real, PAID pilot on a few conditions (needs real backends in the config, OPENAI_API_KEY,
    #    and --confirm-paid)
    python -m experiments.run --config CONFIG --datasets hotpotqa --strategies random --budgets 5 --confirm-paid

The full 48-condition benchmark is never started by accident: a run whose config uses a real
(openai) backend refuses to start without --confirm-paid, and a missing API key stops the whole run
before any condition begins (it never falls back to the mock).

The matrix is the `matrix:` block of the config YAML (default: 2 datasets x 4 strategies x 6 budgets
= 48). Selecting a subset:  --datasets  --strategies  --budgets (5,10 or 5%,10% or 0.05)  --condition ID
(repeatable)  --limit N.   --force redoes completed conditions.   --no-judge scores EM / F1 only.
--max-total-usd X stops BEFORE a condition once the money spent in this run reaches X (or cannot be known).

Output:  <output_dir>/experiments/<experiment_id>/  (or --out-dir)
    matrix.json   results.jsonl   conditions/<condition_id>/{condition_result.json, predictions.jsonl,
    scored.jsonl, eval_summary.json, extraction/, graph/}

Exit codes: 0 every selected condition completed cleanly (or was already complete) |
            1 finished with problems (failed items, a failed condition, or a stale result) |
            2 stopped (bad arguments, paid run without --confirm-paid, missing key, a run-level error).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from experiments.conditions import condition_cfg, generate_conditions, load_matrix, select_conditions
from experiments.runner import (RunnerContext, paid_backends, plan_matrix, preflight, run_matrix, write_manifest)
from extraction.base_extractor import ExtractionAbort
from src.corpus import PROJECT_ROOT

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_budget(token: str) -> float:
    """'5' / '5%' = 5 percent; '0.05' = 5 percent; '1' = 100 percent (a bare value <= 1 is a fraction)."""
    token = token.strip()
    if token.endswith("%"):
        return float(token[:-1]) / 100
    v = float(token)
    return v / 100 if v > 1 else v


def _csv(value: str | None) -> list[str] | None:
    return None if value is None else [x.strip() for x in value.split(",") if x.strip()]


def _money(v) -> str:
    return "UNKNOWN" if v is None else f"${v:.4f}"


def _print_plan(plan: dict, ctx: RunnerContext, args, base_cfg) -> None:
    print(f"DRY RUN - nothing is sent to any API, nothing is written.  matrix: {ctx.matrix_id}")
    print(f"{'condition_id':<42}{'status':<24}{'chunks':>7}{'select':>7}{'quest':>6}{'extract$':>10}{'answer$':>9}{'judge$':>8}")

    def cell(v, width, money=False):
        return f"{'-':>{width}}" if v is None else (f"{v:>{width}.4f}" if money else f"{v:>{width}}")

    for r in plan["conditions"]:
        print(f"{r['condition_id']:<42}{r['status']:<24}{cell(r['n_chunks_total'], 7)}{cell(r['n_selected'], 7)}"
              f"{cell(r['n_questions'], 6)}{cell(r['est_extraction_usd'], 10, True)}"
              f"{cell(r['est_generation_usd'], 9, True)}{cell(r['est_judge_usd'], 8, True)}")
    t = plan["totals"]
    print(f"\n{plan['n_conditions']} condition(s).")
    print(f"  extraction, sum over conditions : {_money(t['est_extraction_usd_sum_of_conditions'])}")
    print(f"  extraction, caches shared (upper bound: every chunk of a dataset extracted at most once): "
          f"{_money(t['extraction_cache_shared_upper_bound_usd'])}")
    print(f"  answer generation (upper bound) : {_money(t['est_generation_usd_upper_bound'])}")
    print(f"  judge                           : {_money(t['est_judge_usd'])}")
    for n in plan["notes"]:
        print(f"  NOTE: {n}")
    print(f"  {plan['estimate_note']}")
    paid = paid_backends(base_cfg, args.no_judge)
    print("  backends: " + (", ".join(paid) + "  -> a real run costs money and needs --confirm-paid" if paid
                            else "all mock - a real run is free (and its results are never reportable)"))
    if args.max_total_usd is not None:
        est = t["extraction_cache_shared_upper_bound_usd"]
        parts = [est, t["est_generation_usd_upper_bound"], t["est_judge_usd"]]
        if any(p is None for p in parts):
            print(f"  spending limit ${args.max_total_usd}: cannot be compared (some estimates are unavailable)")
        else:
            total = sum(parts)
            print(f"  spending limit ${args.max_total_usd}: estimated total ${total:.4f} is "
                  f"{'within' if total <= args.max_total_usd else 'ABOVE'} the limit")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config YAML (shared ExperimentConfig + optional matrix: block)")
    ap.add_argument("--dry-run", action="store_true", help="list the conditions and estimated cost; call nothing")
    ap.add_argument("--datasets", help="comma-separated subset, e.g. hotpotqa")
    ap.add_argument("--strategies", help="comma-separated subset, e.g. random,ketrag")
    ap.add_argument("--budgets", help="comma-separated subset, e.g. 5,25,100")
    ap.add_argument("--condition", action="append", help="run only this condition id (repeatable)")
    ap.add_argument("--limit", type=int, help="run only the first N selected conditions")
    ap.add_argument("--force", action="store_true", help="redo conditions that are already completed")
    ap.add_argument("--no-judge", action="store_true", help="score EM and F1 only (no judge calls)")
    ap.add_argument("--confirm-paid", action="store_true", help="required to start a run that uses a real (openai) backend")
    ap.add_argument("--max-total-usd", type=float, help="stop before a condition once this much has been spent in this run")
    ap.add_argument("--out-dir", help="matrix output folder (default: <output_dir>/experiments/<experiment_id>)")
    args = ap.parse_args(argv)

    try:
        base_cfg, spec = load_matrix(args.config)
        if not _SAFE_ID.match(base_cfg.experiment_id):
            raise ValueError(f"experiment_id {base_cfg.experiment_id!r} must only use letters, digits, '.', '_', '-'")
        if args.max_total_usd is not None and args.max_total_usd <= 0:
            raise ValueError("--max-total-usd must be positive")
        all_conditions = generate_conditions(spec, base_cfg.seed)
        selected = select_conditions(
            all_conditions, datasets=_csv(args.datasets), strategies=_csv(args.strategies),
            budgets=None if args.budgets is None else [parse_budget(b) for b in _csv(args.budgets)],
            ids=args.condition, limit=args.limit)
        if not selected:
            raise ValueError("the selection contains no conditions")
    except (ValueError, FileNotFoundError) as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2

    matrix_dir = Path(args.out_dir) if args.out_dir else _resolve(base_cfg.output_dir) / "experiments" / base_cfg.experiment_id
    ctx = RunnerContext(base_cfg=base_cfg, matrix_id=base_cfg.experiment_id, matrix_dir=matrix_dir,
                        no_judge=args.no_judge, log=lambda m: print(m, flush=True))

    if args.dry_run:
        try:
            _print_plan(plan_matrix(selected, ctx), ctx, args, base_cfg)
        except (ExtractionAbort, ValueError, FileNotFoundError, ImportError) as e:
            print(f"STOPPED: {e}", file=sys.stderr)
            return 2
        return 0

    paid = paid_backends(base_cfg, args.no_judge)
    if paid and not args.confirm_paid:
        print("STOPPED: this config uses real (paid) backends: " + ", ".join(paid) + ".\n"
              "  Nothing was run. Review the cost first:  python -m experiments.run --config "
              f"{args.config} --dry-run\n  then add --confirm-paid to start (use --datasets/--strategies/--budgets "
              "for a small pilot first).", file=sys.stderr)
        return 2
    try:
        preflight(condition_cfg(base_cfg, selected[0], ctx.matrix_id), ctx)
    except (ExtractionAbort, ValueError, ImportError) as e:
        print(f"STOPPED before running anything: {e}", file=sys.stderr)
        return 2

    write_manifest(ctx, spec, all_conditions)
    out = run_matrix(selected, ctx, force=args.force, max_total_usd=args.max_total_usd, all_conditions=all_conditions)

    by_status: dict[str, int] = {}
    for r in out.results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print(f"\nran {len(out.results)} condition(s) {by_status}; skipped (already completed): "
          f"{len(out.skipped_completed)}; stale (refused): {len(out.stale)}")
    print(f"output: {matrix_dir}   (results.jsonl, matrix.json, conditions/)")
    if any(r.is_mock for r in out.results):
        print("MOCK RESULTS - testing only, NOT reportable (see not_reportable_reasons in each result).")
    for r in out.results:
        for p in r.problems:
            print(f"  {r.condition_id}: {p}", file=sys.stderr)
    if out.stale:
        print("  STALE: completed under a different config, left untouched: " + ", ".join(out.stale) +
              "  (use --force to redo, or a new experiment_id)", file=sys.stderr)
    if out.stopped:
        print(f"STOPPED EARLY: {out.stopped['message']}  (re-run the same command to resume)", file=sys.stderr)
        return 2
    return 1 if (out.stale or any(r.status != "completed" for r in out.results)) else 0


if __name__ == "__main__":
    sys.exit(main())
