"""
Tables from an `AnalysisSummary` (Phase 12, substep 4A): CSV, Markdown and LaTeX.

`build_tables(summary)` returns plain `Table` objects (raw values, no formatting); `render_csv`, `render_markdown`
and `render_latex` turn one into text; `write_tables` writes them. Nothing here computes statistics: every number
comes from the summary, so the tables cannot disagree with the JSON.

CSV keeps full precision and leaves a missing value EMPTY (never 0). Markdown and LaTeX round to 4 decimals, show a
missing value as "n/a" and print p-values below 0.0001 as "<0.0001".

STATUS IS ON EVERY OUTPUT. Each CSV row carries `data_status` and `watermark` columns. Markdown starts with a banner
line and LaTeX with a comment line plus the banner in the caption. For mock data the banner is "MOCK — NOT RESULTS";
for a partial real run it is the partial-run banner. Mock comparison tables have empty inferential columns because
the summary suppresses them (see analysis/summary.py): a mock table cannot show a p-value or a significance flag.

Output is deterministic: fixed row order (the summary's), no timestamps, "\\n" line endings.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from analysis.summary import AnalysisSummary, ArmRow

FORMATS = ("csv", "md", "tex")


@dataclass(frozen=True)
class Table:
    name: str                      # file stem
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple, ...]
    notes: tuple[str, ...] = ()
    status: str = "real_complete"
    banner: str | None = None


# ============================================================================== builders
def _arm_cells(r: ArmRow) -> tuple:
    a = r.arm
    return (a.dataset, a.metric, r.role, a.strategy, a.budget_pct, a.n, a.n_universe, a.n_missing,
            a.mean, None if a.ci is None else a.ci.low, None if a.ci is None else a.ci.high, a.n_replicates,
            a.n_partial, r.attrition_warning)


_ARM_COLS = ("dataset", "metric", "role", "strategy", "budget_pct", "n", "n_universe", "n_missing", "mean",
             "ci_low", "ci_high", "n_replicates", "n_partial", "attrition_warning")


def build_tables(summary: AnalysisSummary) -> list[Table]:
    s, b = summary.status, summary.banner
    mock = s == "mock_layout_test"
    ci_pct = f"{summary.provenance.settings['ci_level']:.0%}"
    out: list[Table] = []

    def add(name, title, columns, rows, notes=()):
        out.append(Table(name=name, title=title, columns=tuple(columns), rows=tuple(tuple(r) for r in rows),
                         notes=tuple(notes), status=s, banner=b))

    # 1. long performance table (every arm)
    add("performance", "Quality metrics by dataset, strategy and budget", _ARM_COLS,
        [_arm_cells(r) for r in summary.arms],
        [f"mean = mean over the n questions that have a measured value; ci = percentile bootstrap {ci_pct} interval "
         "of that mean (resampling questions).",
         "n counts questions, never questions x shuffles. Random's value per question is its mean over the shuffles "
         "that have a measured value (n_replicates = shuffles used).",
         "A missing value is not a zero: n_missing questions have no value for this metric (see missing_data).",
         "F1 and judge accuracy are the primary metrics; EM is secondary."])

    # 2. wide performance table: strategy x budget of the mean
    budgets = sorted({r.arm.budget_pct for r in summary.arms})
    wide_rows = []
    keys = []
    for r in summary.arms:
        k = (r.arm.dataset, r.arm.metric, r.arm.strategy)
        if k not in keys:
            keys.append(k)
    lookup = {(r.arm.dataset, r.arm.metric, r.arm.strategy, r.arm.budget_pct): r.arm.mean for r in summary.arms}
    for ds, metric, strat in keys:
        wide_rows.append((ds, metric, strat, *[lookup.get((ds, metric, strat, x)) for x in budgets]))
    add("performance_wide", "Mean quality metric by strategy (columns: budget %)",
        ("dataset", "metric", "strategy", *[f"budget_{x}pct" for x in budgets]), wide_rows,
        ["Means only; sample sizes and intervals are in the performance table."])

    # 3. paired comparisons
    comp_cols = ("analysis", "dataset", "metric", "role", "budget_pct", "strategy_a", "strategy_b", "n_universe",
                 "n_pairs", "n_dropped", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "dz", "n_a_better",
                 "n_b_better", "n_tied", "test_method", "p_value", "p_holm", "p_bh", "significant_holm", "mcnemar_p",
                 "flags", "suppressed", "skipped_reason")
    add("comparisons", "Paired strategy comparisons (A minus B)", comp_cols,
        [(c.analysis, c.dataset, c.metric, c.role, c.budget_pct, c.strategy_a, c.strategy_b, c.n_universe, c.n_pairs,
          c.n_dropped, c.mean_a, c.mean_b, c.mean_diff, c.ci_low, c.ci_high, c.dz, c.n_a_better, c.n_b_better,
          c.n_tied, c.test_method, c.p_value, c.p_holm, c.p_bh, c.significant_holm, c.mcnemar_p, ";".join(c.flags),
          c.suppressed, c.skipped_reason) for c in summary.comparisons],
        (["MOCK DATA: inferential columns are intentionally empty; no comparison is reported."] if mock else
         ["Paired by question. p_value: paired sign-flip permutation test (two-sided); p_holm / p_bh adjust within one "
          "family = one (dataset, metric, analysis) over budgets below the anchor budget.",
          "ci: percentile bootstrap interval of mean(A) - mean(B). dz: mean difference / SD of differences.",
          "primary_complete_case drops questions with a missing value in either arm; sensitivity_missing_as_wrong "
          "scores them 0. Only the first is the primary analysis.",
          "mcnemar_p appears only for single-run binary arms and is algebraically identical to the sign-flip test "
          "(a consistency check, not independent confirmation)."]))

    # 4. families
    add("families", "Multiple-testing families", ("dataset", "metric", "role", "analysis", "n_tests", "n_tests_computed",
                                                   "n_tests_skipped", "alpha", "inference_suppressed", "complete"),
        [(f.dataset, f.metric, f.role, f.analysis, f.n_tests, f.n_tests_computed, f.n_tests_skipped, f.alpha,
          f.inference_suppressed, f.complete) for f in summary.families],
        ["Holm and Benjamini-Hochberg are applied inside one family only. The 100% budget is an anchor, not a test."])

    # 5. missing data
    miss = [r for r in summary.arms if r.arm.n_missing]
    add("missing_data", "Questions without a measured value", ("dataset", "metric", "strategy", "budget_pct", "n_universe",
                                                               "n_missing", "fraction_missing", "reasons", "attrition_warning"),
        [(r.arm.dataset, r.arm.metric, r.arm.strategy, r.arm.budget_pct, r.arm.n_universe, r.arm.n_missing,
          r.attrition_fraction, "; ".join(f"{k}={v}" for k, v in r.arm.missing_reasons.items()), r.attrition_warning)
         for r in miss],
        ["Empty table = every question has a value in every arm." if not miss else
         "Missing is not zero. The primary analysis is complete-case; a sensitivity analysis scores missing as wrong.",
         "attrition_warning: more than 2% of the questions have no value."])

    # 6. Random's variability
    add("random_variability", "Random strategy: variability across shuffles",
        ("dataset", "metric", "budget_pct", "n_shuffles", "mean", "sd", "min", "max", "per_shuffle"),
        [(v.dataset, v.metric, v.budget_pct, v.n_shuffles, v.mean, v.sd, v.min, v.max,
          "; ".join(f"seed {p['shuffle_seed']}: {p['mean']} (n={p['n']})" for p in v.per_shuffle))
         for v in summary.shuffle_variability],
        ["Each shuffle is a different Random selection on the SAME questions; sd is the sample SD of the per-shuffle means "
         "and is a crude estimate when n_shuffles is small. This is Random's selection luck, not question sampling error."])

    # 7. cost / tokens / runtime
    add("cost_runtime", "Cost, tokens and runtime (mean over measured conditions)",
        ("dataset", "strategy", "budget_pct", "measure", "n_records", "n_known", "n_lower_bound", "n_unknown", "mean",
         "min", "max"),
        [(c.dataset, c.strategy, c.budget_pct, c.measure, c.n_records, c.n_known, c.n_lower_bound, c.n_unknown, c.mean,
          c.min, c.max) for c in summary.cost],
        ["Means use measured values only; lower bounds and unknowns are counted, never mixed into the mean. "
         "Cached extraction can make cost and runtime depend on run order."])

    # 8. evidence
    if summary.evidence:
        add("evidence", "Evidence metrics (gold chunks, computed post hoc)", _ARM_COLS,
            [_arm_cells(r) for r in summary.evidence],
            ["selection_recall = |gold ∩ selected| / |gold|; retrieval_recall = |gold ∩ retrieved| / |gold| (unconditional); "
             "conditional_retrieval_recall = |gold ∩ retrieved| / |gold ∩ selected|.",
             "Questions without gold chunks, or where a metric's condition is not met, are undefined (missing), not 0."])

    # 9. anchor
    add("anchor_checks", f"Anchor budget ({summary.provenance.settings['anchor_budget_pct']}%) consistency",
        ("dataset", "metric", "budget_pct", "strategy_a", "strategy_b", "n_pairs", "identical_outcomes",
         "selected_chunks_identical", "note"),
        [(a.dataset, a.metric, a.budget_pct, a.strategy_a, a.strategy_b, a.n_pairs, a.identical_outcomes,
          a.selected_chunks_identical, a.note) for a in summary.anchor_checks],
        ["A sanity check, not a result: at full budget every strategy sees the same chunks. A mismatch is reported, never forced."])

    # 10. provenance
    p = summary.provenance
    prov_rows = [("status", summary.status), ("data_kind", summary.data_kind), ("banner", summary.banner),
                 ("git_commit", p.git_commit), ("matrix_dir_name", p.matrix_dir_name),
                 ("matrix_ids", ";".join(p.matrix_ids)),
                 ("shared_config_fingerprints", ";".join(p.shared_config_fingerprints)),
                 ("question_seed", ";".join(f"{k}={v}" for k, v in p.question_seed.items())),
                 ("input_digest_sha256", p.input_digest), ("n_input_files", len(p.source_files)),
                 *[(f"setting.{k}", v) for k, v in p.settings.items()],
                 *[(f"version.{k}", v) for k, v in p.versions.items()],
                 ("analysis_schema_version", p.analysis_schema_version), ("summary_schema_version", p.summary_schema_version)]
    add("provenance", "Provenance", ("key", "value"), prov_rows,
        ["input_digest_sha256 hashes every input file (path and SHA-256); the per-file hashes are in summary.json.",
         "git_commit is read from .git without running git; empty if unavailable."])
    return out


# ============================================================================== renderers
def _flag_cols(t: Table) -> list[str]:
    return ["data_status", "watermark"]


def render_csv(t: Table) -> str:
    """Full precision, empty = missing; first columns data_status and watermark."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([*_flag_cols(t), *t.columns])
    for row in t.rows:
        w.writerow([t.status, t.banner or "", *[_csv(v) for v in row]])
    return buf.getvalue()


def _csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _fmt(v, col: str) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if col.startswith(("p_", "mcnemar_p")) and v < 1e-4:
            return "<0.0001"
        return f"{v:.4f}"
    return str(v)


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render_markdown(t: Table) -> str:
    lines = []
    if t.banner:
        lines += [f"**{t.banner}**", ""]
    lines += [f"### {t.title}", ""]
    lines.append("| " + " | ".join(t.columns) + " |")
    lines.append("|" + "|".join("---" for _ in t.columns) + "|")
    for row in t.rows:
        lines.append("| " + " | ".join(_md_escape(_fmt(v, c)) for v, c in zip(row, t.columns)) + " |")
    if not t.rows:
        lines.append("| " + " | ".join("—" if i == 0 else "" for i in range(len(t.columns))) + " |")
    if t.notes:
        lines.append("")
        lines += [f"- {n}" for n in t.notes]
    return "\n".join(lines) + "\n"


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(s: str) -> str:
    s = "".join(_TEX.get(ch, ch) for ch in s)
    return (s.replace("—", "---").replace("∩", r"$\cap$").replace("≥", r"$\geq$").replace("≤", r"$\leq$")
             .replace("×", r"$\times$"))


def render_latex(t: Table) -> str:
    """A standalone `table` environment using only `\\hline` (no packages beyond LaTeX itself); long tables
    should be resized by the author (e.g. \\small, longtable) - the layout is deliberately plain."""
    cap = (t.banner + ". " if t.banner else "") + t.title
    lines = []
    if t.banner:
        lines.append(f"% {t.banner.replace('—', '---')}")
    lines += [r"\begin{table}[ht]", r"\centering", r"\small",
              r"\begin{tabular}{" + "l" * len(t.columns) + "}", r"\hline",
              " & ".join(_tex(c) for c in t.columns) + r" \\", r"\hline"]
    for row in t.rows:
        lines.append(" & ".join(_tex(_fmt(v, c)) for v, c in zip(row, t.columns)) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\caption{" + _tex(cap) + "}", r"\end{table}"]
    if t.notes:
        lines += ["% notes:"] + ["% " + _tex(n).replace("\n", " ") for n in t.notes]
    return "\n".join(lines) + "\n"


_RENDER = {"csv": render_csv, "md": render_markdown, "tex": render_latex}


def write_tables(tables: Sequence[Table], out_dir: str | Path, formats: Sequence[str] = FORMATS) -> list[Path]:
    """Write every table in every requested format to `out_dir` (created if needed); returns the paths written."""
    bad = [f for f in formats if f not in _RENDER]
    if bad:
        raise ValueError(f"unknown table format(s) {bad}; expected {list(FORMATS)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in tables:
        for f in formats:
            p = out_dir / f"{t.name}.{f}"
            p.write_text(_RENDER[f](t), encoding="utf-8", newline="\n")
            paths.append(p)
    return paths
