"""
Phase 12 command line: load a finished Phase 10 matrix, validate it, and write the summary, tables and figures.

    # Is this matrix analysable? Writes NOTHING (exit 0 = yes, 2 = refused):
    python -m analysis.run --matrix-dir results/experiments/benchmark_real --validate-only

    # The analysis of a complete real matrix -> <matrix-dir>/analysis/real/{summary.json, tables/, figures/}
    python -m analysis.run --matrix-dir results/experiments/benchmark_real

    # Same, found via the config that produced the matrix
    python -m analysis.run --config configs/benchmark_real.yaml

    # A matrix that is missing planned conditions: explicit opt-in, output marked PARTIAL RUN
    python -m analysis.run --matrix-dir DIR --allow-incomplete                 -> <out>/real_partial/

    # Mock data (free runs): refused unless you say it is only a layout test; output marked MOCK - NOT RESULTS
    python -m analysis.run --matrix-dir results/experiments/benchmark_mock --allow-mock-layout-test   -> <out>/mock_layout_test/

THIS MODULE ONLY CONNECTS THE OTHER PARTS. It computes no statistic itself: analysis.loader gates and loads,
analysis.evidence computes the evidence metrics (when gold chunks are available), analysis.summary builds the summary
(through analysis.random_avg and analysis.stats), analysis.tables and analysis.plots render it. It never calls an API,
downloads a dataset (unless you pass --allow-rebuild, see below) or touches Git (the commit hash is read from .git).

ORDER OF WORK. (1) arguments are validated; (2) the matrix passes the loader's gate (mock / reportable / aligned /
reconciled); (3) evidence metrics are computed (--evidence) and must reconcile with the runner's stored figures;
(4) the validation report is built (completeness, question alignment, Random shuffles, evidence, input hashes, git
commit); (5) with --validate-only the report is printed and the run ends; otherwise (6) the summary, tables and figures
are built IN MEMORY and only then (7) written. Every refusal happens before anything is written: a refused run leaves
no file behind.

OUTPUT FOLDERS. <out-dir> (default <matrix-dir>/analysis) gets one sub-folder per data status and never mixes them:
    mock_layout_test/   mock data (layout test only; files start with MOCK_; MOCK_NOT_RESULTS.txt marker)
    real_partial/       a real but incomplete matrix (files start with PARTIAL_; PARTIAL_RUN.txt marker)
    real/               a complete real matrix
Each holds summary.json, tables/, figures/ and run_manifest.json (arguments, checks, SHA-256 of every file written).
Nothing in the outputs depends on the clock, so the same input and options give byte-identical files.

EVIDENCE (--evidence off|auto|require, default auto). Gold chunk labels are needed for evidence metrics. They are read
from data/corpus/<corpus_id>/chunks.json (written by `python -m src.prepare_data`); with `auto` a missing file means the
evidence figures and tables are skipped with a note, with `require` it stops the run. A wrong corpus or a failed
reconciliation with the runner's figures always stops the run. --allow-rebuild lets the evidence step rebuild the corpus
from the saved config, which for data_source huggingface may DOWNLOAD the dataset: it is off by default.

--metrics selects which quality metrics are written (default: f1, judge_accuracy, em). Hypothesis families are per
metric, so a selection never changes a p-value. Warnings are not filtered by the selection.

EXIT CODES: 0 clean | 1 finished (or validated) with warnings: a partial run, attrition above 2%, skipped tests, evidence
issues, excluded conditions | 2 stopped: bad arguments, refused input (mock without opt-in, mixed, not reportable,
misaligned, unreadable), failed validation. Exit 2 writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from analysis import evidence as ev
from analysis import plots as pl
from analysis import tables as tb
from analysis.loader import AnalysisRefused, load_analysis_input
from analysis.schemas import METRICS, PRIMARY_METRICS, AnalysisInput
from analysis.summary import (AnalysisSummary, Headline, POLICIES, build_summary, read_git_commit, summary_to_json)
from aggregation.loader import AggregationError

EXIT_OK, EXIT_WARN, EXIT_STOP = 0, 1, 2
MOCK_MARKER = "MOCK_NOT_RESULTS.txt"
PARTIAL_MARKER = "PARTIAL_RUN.txt"
MIN_RESAMPLES = 100


class UsageError(ValueError):
    """Invalid command-line arguments."""


@dataclass
class Check:
    name: str
    status: str            # pass | warn | fail | info
    detail: str


@dataclass
class Validation:
    checks: list[Check] = field(default_factory=list)

    def add(self, name, status, detail):
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.status == "warn" for c in self.checks)

    def as_dicts(self) -> list[dict]:
        return [{"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks]


# ============================================================================== arguments
def _csv(value: str | None) -> list[str] | None:
    return None if value is None else [x.strip() for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m analysis.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("input")
    src.add_argument("--matrix-dir", help="a finished Phase 10 matrix directory")
    src.add_argument("--config", help="the config that produced the matrix: analyses <output_dir>/experiments/<experiment_id>")
    gate = ap.add_argument_group("what may be analysed")
    gate.add_argument("--validate-only", action="store_true", help="run the gate and the checks, print the report, write nothing")
    gate.add_argument("--allow-incomplete", action="store_true",
                      help="accept a real matrix that is missing planned conditions (output is marked PARTIAL RUN)")
    gate.add_argument("--allow-mock-layout-test", action="store_true",
                      help="accept mock data for a LAYOUT test only (output goes to mock_layout_test/, marked MOCK - NOT RESULTS)")
    out = ap.add_argument_group("output")
    out.add_argument("--out-dir", help="output root (default: <matrix-dir>/analysis); status sub-folders are created inside it")
    out.add_argument("--no-figures", action="store_true", help="write the summary and tables only")
    out.add_argument("--formats", default=",".join(tb.FORMATS), help="table formats, comma-separated from csv,md,tex")
    out.add_argument("--figure-formats", default=",".join(pl.FORMATS), help="figure formats, comma-separated from png,pdf")
    out.add_argument("--dpi", type=int, default=pl.DPI, help="PNG resolution (default 300)")
    stat = ap.add_argument_group("statistics")
    stat.add_argument("--metrics", help="quality metrics to write, comma-separated from " + ",".join(METRICS) + " (default: all)")
    stat.add_argument("--alpha", type=float, default=0.05, help="significance level for Holm, in (0, 1) (default 0.05)")
    stat.add_argument("--n-resamples", type=int, default=10_000, help=f"bootstrap resamples (>= {MIN_RESAMPLES}; default 10000)")
    stat.add_argument("--n-permutations", type=int, default=20_000,
                      help=f"sign-flip permutations when the exact test is infeasible (>= {MIN_RESAMPLES}; default 20000)")
    stat.add_argument("--seed", type=int, default=0, help="base seed for the bootstrap and permutation draws (default 0)")
    evg = ap.add_argument_group("evidence metrics (need gold chunk labels)")
    evg.add_argument("--evidence", choices=("off", "auto", "require"), default="auto",
                     help="auto: compute when gold labels are on disk, else skip with a note; require: stop if unavailable")
    evg.add_argument("--corpus-root", help="folder holding <corpus_id>/chunks.json (default: <project>/data/corpus)")
    evg.add_argument("--allow-rebuild", action="store_true",
                     help="rebuild the corpus from the saved config if chunks.json is missing (a huggingface rebuild may DOWNLOAD data)")
    return ap


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:                               # argparse reports its own errors; keep main() returning an int
        raise UsageError("invalid arguments (see above)") if e.code not in (0, None) else SystemExit(0)
    if bool(args.matrix_dir) == bool(args.config):
        raise UsageError("give exactly one of --matrix-dir or --config")
    if not (0 < args.alpha < 1):
        raise UsageError("--alpha must be strictly between 0 and 1")
    for name in ("n_resamples", "n_permutations"):
        if getattr(args, name) < MIN_RESAMPLES:
            raise UsageError(f"--{name.replace('_', '-')} must be at least {MIN_RESAMPLES}")
    if args.seed < 0:
        raise UsageError("--seed must be >= 0")
    if args.dpi < 50 or args.dpi > 1200:
        raise UsageError("--dpi must be between 50 and 1200")
    args.metrics = _csv(args.metrics)
    if args.metrics is not None:
        bad = [m for m in args.metrics if m not in METRICS]
        if bad or not args.metrics or len(set(args.metrics)) != len(args.metrics):
            raise UsageError(f"--metrics must be a non-empty list of unique values from {METRICS}; got {args.metrics}")
    for opt, allowed in (("formats", tb.FORMATS), ("figure_formats", pl.FORMATS)):
        vals = _csv(getattr(args, opt))
        if not vals or [v for v in vals if v not in allowed]:
            raise UsageError(f"--{opt.replace('_', '-')} must be a non-empty list from {list(allowed)}; got {getattr(args, opt)!r}")
        setattr(args, opt, vals)
    return args


def resolve_matrix_dir(args: argparse.Namespace) -> Path:
    if args.matrix_dir:
        return Path(args.matrix_dir)
    from experiments.conditions import load_matrix
    from experiments.run import _resolve
    cfg, _spec = load_matrix(args.config)
    return _resolve(cfg.output_dir) / "experiments" / cfg.experiment_id


# ============================================================================== evidence
def run_evidence(analysis: AnalysisInput, args, validation: Validation, notes: list[str]) -> ev.EvidenceReport | None:
    """The evidence report, or None when it is off / unavailable (auto). Wrong corpus or failed reconciliation -> failed check."""
    if args.evidence == "off":
        validation.add("evidence", "info", "evidence metrics switched off (--evidence off)")
        return None
    try:
        gold = {d: ev.find_gold_source(analysis, d, corpus_root=args.corpus_root, allow_rebuild=args.allow_rebuild)
                for d in sorted({c.dataset for c in analysis.conditions})}
    except ev.EvidenceError as e:
        if args.evidence == "require":
            validation.add("evidence", "fail", f"gold chunk labels are required but unavailable: {e}")
        else:
            validation.add("evidence", "info", "gold chunk labels not found: evidence metrics skipped "
                           "(run `python -m src.prepare_data --config <config>` to write them)")
            notes.append("evidence metrics were skipped: gold chunk labels are not on disk")
        return None
    try:
        report = ev.compute_evidence(analysis, gold_sources=gold)
    except ev.EvidenceError as e:
        validation.add("evidence", "fail", f"the gold labels do not belong to this matrix: {e}")
        return None
    errors = [i for i in report.issues if i.severity == "error"]
    if not report.reconciled or errors:
        validation.add("evidence", "fail", f"evidence does not reconcile with the runner's stored figures: "
                       f"{len(errors)} error(s), e.g. {errors[0].code + ': ' + errors[0].message if errors else 'n/a'}")
        return None
    warns = [i for i in report.issues if i.severity == "warning"]
    validation.add("evidence", "warn" if warns else "pass",
                   f"evidence reconciled with the runner for {len(report.conditions)} conditions (gold: "
                   f"{'; '.join(f'{d}: {o}' for d, o in sorted(report.gold_origin.items()))})"
                   + (f"; {len(warns)} warning(s)" if warns else ""))
    return report


# ============================================================================== validation
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_input(analysis: AnalysisInput, validation: Validation, git_commit: str | None) -> None:
    rep = analysis.report
    # completeness
    if analysis.status == "mock_layout_test":
        validation.add("data status", "info", "MOCK data accepted for a layout test only: outputs are marked MOCK — NOT RESULTS")
    elif analysis.status == "real_partial":
        validation.add("data status", "warn", "PARTIAL real run accepted (--allow-incomplete): outputs are marked PARTIAL RUN")
    else:
        validation.add("data status", "pass", "real data, every planned condition present and reconciled")
    planned = rep.n_planned
    if planned is None:
        validation.add("matrix completeness", "warn" if analysis.status != "mock_layout_test" else "info",
                       f"the planned number of conditions is unknown (no matrix.json); {rep.n_included} included")
    elif rep.n_included == planned and rep.matrix_complete:
        validation.add("matrix completeness", "pass", f"{rep.n_included} of {planned} planned conditions included")
    else:
        validation.add("matrix completeness", "warn", f"{rep.n_included} of {planned} planned conditions included "
                       f"({rep.n_excluded} excluded, matrix_complete={rep.matrix_complete})")
    if analysis.excluded:
        validation.add("excluded conditions", "warn",
                       "; ".join(f"{e.get('condition_id')}: {e.get('kind')}" for e in analysis.excluded[:5])
                       + (" …" if len(analysis.excluded) > 5 else ""))
    # alignment (the loader refuses a misaligned matrix; this re-checks what it returned)
    bad = []
    for c in analysis.conditions:
        if set(c.question_ids) != set(analysis.question_ids.get(c.dataset, [])):
            bad.append(c.condition_id)
    if bad:
        validation.add("question alignment", "fail", f"{len(bad)} condition(s) do not share the dataset's question set: {bad[:3]}")
    else:
        validation.add("question alignment", "pass", "; ".join(
            f"{d}: {len(q)} questions, question seed {analysis.question_seed.get(d)}" for d, q in sorted(analysis.question_ids.items())))
    # Random shuffles: the same shuffle seeds at every budget
    for d in sorted(analysis.question_ids):
        by_b: dict[int, set] = {}
        for c in analysis.conditions:
            if c.dataset == d and c.strategy == "random":
                by_b.setdefault(c.budget_pct, set()).add(c.shuffle_seed)
        if by_b and len({frozenset(s) for s in by_b.values()}) > 1:
            validation.add("random shuffles", "warn", f"{d}: Random has different shuffle seeds at different budgets: "
                           + "; ".join(f"{b}%: {sorted(s, key=str)}" for b, s in sorted(by_b.items())))
        elif by_b:
            s = sorted(next(iter(by_b.values())), key=str)
            validation.add("random shuffles", "pass", f"{d}: Random shuffle seeds {s} at every budget")
    # provenance: the input files must still be what was loaded
    changed = []
    for f in analysis.source_files:
        p = Path(analysis.matrix_dir) / f.path
        if not p.is_file() or _sha256(p) != f.sha256:
            changed.append(f.path)
    if changed:
        validation.add("provenance", "fail", f"{len(changed)} input file(s) changed or vanished since they were loaded: {changed[:3]}")
    else:
        validation.add("provenance", "pass", f"{len(analysis.source_files)} input files hashed (SHA-256) and unchanged")
    validation.add("git commit", "pass" if git_commit else "info",
                   git_commit if git_commit else "no git repository found (provenance records no commit)")
    if analysis.judge_enabled is False:
        validation.add("judge", "info", "the judge was disabled: judge accuracy is not analysed")
    for c in analysis.caveats:
        validation.add("caveat", "info", c)


# ============================================================================== metric selection
def select_metrics(summary: AnalysisSummary, wanted: list[str]) -> AnalysisSummary:
    """Drop everything about the metrics that were not selected. Nothing is recomputed; the one derived figure, the
    headline count of significant primary tests, is recounted from the retained comparison rows (and equals the
    summary's own when every metric is kept)."""
    analysed = summary.metrics["analysed"]
    missing = [m for m in wanted if m not in analysed]
    if missing:
        raise UsageError("metric(s) not available in this matrix: " + ", ".join(
            f"{m} ({summary.skipped_metrics.get(m, 'not analysed')})" for m in missing))
    keep = [m for m in analysed if m in wanted]
    if keep == analysed:
        return summary
    comps = [c for c in summary.comparisons if c.metric in keep]
    headline = summary.headline
    if summary.status != "mock_layout_test":
        prim = [c for c in comps if c.analysis == POLICIES["complete_case"] and c.role == "primary" and c.p_value is not None]
        sig = sum(1 for c in prim if c.significant_holm)
        lead = "PARTIAL RUN. " if summary.status == "real_partial" else ""
        alpha = summary.provenance.settings["alpha"]
        headline = Headline(statement=f"{lead}{sig} of {len(prim)} primary paired comparisons (complete-case; selected metrics: "
                                      f"{', '.join(m for m in keep if m in PRIMARY_METRICS) or 'none'}) pass Holm at alpha={alpha}. "
                                      "This is a mechanical count, not an interpretation.",
                            n_primary_tests=len(prim), significant_primary_tests=sig)
    return summary.model_copy(update={
        "arms": [r for r in summary.arms if r.arm.metric in keep], "comparisons": comps,
        "families": [f for f in summary.families if f.metric in keep],
        "shuffle_variability": [v for v in summary.shuffle_variability if v.metric in keep],
        "anchor_checks": [a for a in summary.anchor_checks if a.metric in keep], "headline": headline,
        "metrics": {**summary.metrics, "analysed": keep, "primary": [m for m in keep if m in PRIMARY_METRICS],
                    "secondary": [m for m in keep if m not in PRIMARY_METRICS]}})


# ============================================================================== output
def _marker_text(status: str) -> str | None:
    if status == "mock_layout_test":
        return ("MOCK — NOT RESULTS\n\nEverything in this folder was produced from MOCK data for a layout test.\n"
                "It contains no research findings, no valid comparisons and no significance claims.\n"
                "Do not cite, report or compare these files.\n")
    if status == "real_partial":
        return ("PARTIAL RUN\n\nThe matrix this analysis read is missing planned conditions (--allow-incomplete).\n"
                "Results are for inspection only until the matrix is complete.\n")
    return None


def _clear_previous(base: Path) -> None:
    """Remove what an EARLIER run of this tool left in `base` (only the names this tool writes), so a re-run with other
    options (e.g. --no-figures, fewer metrics) cannot leave stale files that no longer match summary.json."""
    import shutil
    for name in ("tables", "figures"):
        shutil.rmtree(base / name, ignore_errors=True)
    for name in ("summary.json", "run_manifest.json", MOCK_MARKER, PARTIAL_MARKER):
        (base / name).unlink(missing_ok=True)


def write_outputs(summary: AnalysisSummary, figset, tables, out_root: Path, args, validation: Validation) -> list[Path]:
    from analysis.summary import write_summary
    base = out_root / pl.output_subdir(summary.status)
    base.mkdir(parents=True, exist_ok=True)
    _clear_previous(base)
    written: list[Path] = []
    marker = _marker_text(summary.status)
    if marker:
        p = base / (MOCK_MARKER if summary.status == "mock_layout_test" else PARTIAL_MARKER)
        p.write_text(marker, encoding="utf-8", newline="\n")
        written.append(p)
    written.append(write_summary(summary, base / "summary.json"))
    written += tb.write_tables(tables, base / "tables", formats=args.formats)
    if figset is not None:
        written += pl.write_figures(figset, out_root, formats=args.figure_formats, dpi=args.dpi)
    manifest = {
        "status": summary.status, "banner": summary.banner, "matrix_dir_name": summary.provenance.matrix_dir_name,
        "input_digest": summary.provenance.input_digest, "git_commit": summary.provenance.git_commit,
        "options": {"metrics": summary.metrics["analysed"], "alpha": args.alpha, "n_resamples": args.n_resamples,
                    "n_permutations": args.n_permutations, "seed": args.seed, "evidence": args.evidence,
                    "formats": args.formats, "figure_formats": None if figset is None else args.figure_formats,
                    "dpi": args.dpi, "figures": figset is not None},
        "checks": validation.as_dicts(), "skipped_figures": {} if figset is None else figset.skipped,
        "files": [{"path": p.relative_to(base).as_posix() if p.is_relative_to(base) else p.as_posix(),
                   "sha256": _sha256(p), "bytes": p.stat().st_size} for p in sorted(written)],
    }
    mp = base / "run_manifest.json"
    mp.write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return written + [mp]


# ============================================================================== main
def _print_report(validation: Validation, stream=None):
    stream = sys.stdout if stream is None else stream            # looked up at call time (not bound at import)
    for c in validation.checks:
        print(f"  [{c.status.upper():4}] {c.name}: {c.detail}", file=stream)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except UsageError as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return EXIT_STOP

    try:
        matrix_dir = resolve_matrix_dir(args)
        analysis = load_analysis_input(matrix_dir, allow_mock_layout_test=args.allow_mock_layout_test,
                                       allow_incomplete=args.allow_incomplete)
    except AnalysisRefused as e:
        print(f"STOPPED: analysis refused ({e.kind}):", file=sys.stderr)
        for r in e.reasons:
            print(f"  - {r}", file=sys.stderr)
        if e.kind == "mock":
            print("  (mock data is for layout tests only: pass --allow-mock-layout-test to produce clearly marked, "
                  "non-reportable output)", file=sys.stderr)
        return EXIT_STOP
    except (AggregationError, ValueError, FileNotFoundError, OSError) as e:
        print(f"STOPPED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_STOP

    validation, notes = Validation(), []
    validate_input(analysis, validation, read_git_commit(Path(__file__).resolve().parents[1]))
    evidence = run_evidence(analysis, args, validation, notes)
    print(f"matrix: {matrix_dir}  status: {analysis.status}")
    _print_report(validation)
    if validation.failed:
        for c in validation.checks:
            if c.status == "fail":
                print(f"  FAILED {c.name}: {c.detail}", file=sys.stderr)
        print("STOPPED: validation failed; nothing was written", file=sys.stderr)
        return EXIT_STOP
    if analysis.status == "mock_layout_test":
        print("  MOCK — NOT RESULTS: layout test only; no findings, no significance claims")
    if args.validate_only:
        print("validate-only: nothing was written")
        return EXIT_WARN if validation.warned else EXIT_OK

    try:
        summary = build_summary(analysis, evidence, base_seed=args.seed, n_resamples=args.n_resamples,
                                n_permutations=args.n_permutations, alpha=args.alpha,
                                git_commit=read_git_commit(Path(__file__).resolve().parents[1]))
        if args.metrics is not None:
            summary = select_metrics(summary, args.metrics)
        tables = tb.build_tables(summary)
        figset = None if args.no_figures else pl.build_figures(summary)
    except UsageError as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return EXIT_STOP

    out_root = Path(args.out_dir) if args.out_dir else matrix_dir / "analysis"
    try:
        paths = write_outputs(summary, figset, tables, out_root, args, validation)
    except OSError as e:
        print(f"STOPPED: could not write the outputs: {e}", file=sys.stderr)
        return EXIT_STOP

    base = out_root / pl.output_subdir(summary.status)
    print(f"wrote {len(paths)} files to {base}  (summary.json, tables/, "
          f"{'figures/, ' if figset is not None else ''}run_manifest.json)")
    print(f"  {summary.headline.statement}")
    for n in notes:
        print(f"  note: {n}")
    for w in summary.warnings:
        print(f"  warning: {w}", file=sys.stderr)
    if figset is not None and figset.skipped:
        for name, why in sorted(figset.skipped.items()):
            print(f"  figure not drawn: {name}: {why}")
    return EXIT_WARN if (validation.warned or summary.warnings) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
