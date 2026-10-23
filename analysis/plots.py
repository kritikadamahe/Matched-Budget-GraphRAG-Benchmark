"""
Figures from an `AnalysisSummary` (Phase 12, substep 4B).

`build_figures(summary)` returns a `FigureSet`: matplotlib figures plus, for each, the exact data that was drawn
(`PlotFigure.series`), and a dict of figures that could not be drawn with the reason. `write_figures` saves PNG (300 dpi)
and PDF.

NO NEW STATISTICS. Every plotted number is copied from the summary (arm means and their bootstrap intervals, paired
mean differences and their intervals, per-shuffle means, cost / token / runtime means, evidence means). This module
imports neither `analysis.stats` nor `analysis.random_avg`, and does no arithmetic beyond choosing axis limits.

MISSING IS NOT ZERO. A value the summary holds as None becomes NaN: the point and its interval are not drawn and
a line breaks there. Nothing is interpolated or filled in. Points that cannot be drawn are counted in `PlotFigure.notes`.

ABSTENTION RATE is plotted from `summary.abstention` (counted in analysis.summary from the per-question outcomes). A
summary without those rows (an older one) skips the figure with ABSTENTION_REASON instead of computing anything here.

STATUS ON EVERY FIGURE.
  mock_layout_test  a large diagonal "MOCK — NOT RESULTS" watermark, a red banner at the top, and "MOCK" in the footer.
                    Inference is already suppressed in the summary, so the comparison figure shows differences without
                    intervals and without any significance marking.
  real_partial      an orange "PARTIAL RUN" banner at the top and a lighter diagonal mark.
  real_complete     no banner; the footer still names the status, the input digest and the git commit.

OUTPUT SEPARATION. `figure_dir(out_root, status)` puts mock figures in `<out_root>/mock_layout_test/figures`, partial runs
in `<out_root>/real_partial/figures` and complete real runs in `<out_root>/real/figures`; mock file names also start
with "MOCK_" (partial: "PARTIAL_"). `write_figures` always uses that function: mock figures cannot land in the real folder.

COLOUR. Strategies keep one colour for ever (fixed order from the reference categorical palette, validated for CVD
separation), plus a distinct marker and line style, so colour is never the only cue. Matplotlib's Agg canvas is used
directly (no pyplot), so no display is needed and figure state is not global.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis.summary import MOCK_BANNER, PARTIAL_BANNER, AnalysisSummary, ArmRow  # noqa: E402

DPI = 300
FORMATS = ("png", "pdf")
MOCK_DIR, PARTIAL_DIR, REAL_DIR = "mock_layout_test", "real_partial", "real"

STRATEGY_STYLE = {                       # fixed per strategy: colour (categorical slots 1-4), marker, line style
    "random":       ("#2a78d6", "o", "-"),
    "ketrag":       ("#eb6834", "s", "--"),
    "lazygraphrag": ("#1baf7a", "^", "-."),
    "fastgraphrag": ("#e87ba4", "D", ":"),
}
FALLBACK_STYLE = ("#52514e", "x", "-")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"
MOCK_RED, PARTIAL_ORANGE = "#c0201f", "#c26a00"
DATASET_NAMES = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}
METRIC_LABELS = {
    "f1": "F1 (token overlap, 0–1)", "em": "Exact match (fraction of questions, 0–1)",
    "judge_accuracy": "Judge accuracy (fraction judged correct, 0–1)",
    "selection_recall": "Selection recall (fraction of gold chunks selected)",
    "retrieval_recall": "Retrieval recall (fraction of gold chunks in context)",
    "conditional_retrieval_recall": "Retrieval recall given selected",
    "all_gold_selected": "All gold chunks selected (fraction of questions)",
    "all_gold_retrieved": "All gold chunks retrieved (fraction of questions)",
    "all_gold_retrieved_given_selected": "All gold retrieved given all selected",
}
METRIC_SHORT = {"f1": "F1", "em": "Exact match", "judge_accuracy": "Judge accuracy"}
ABSTENTION_REASON = ("the summary has no abstention rows, and the abstention rate cannot be plotted without computing a "
                     "new statistic in the plotting module")


# ============================================================================== records
@dataclass
class PlotFigure:
    name: str
    title: str
    status: str
    banner: str | None
    fig: Figure
    series: list[dict] = field(default_factory=list)   # exactly what was drawn (None = not drawn / missing)
    notes: list[str] = field(default_factory=list)


@dataclass
class FigureSet:
    status: str
    figures: list[PlotFigure]
    skipped: dict[str, str]

    def by_name(self) -> dict[str, PlotFigure]:
        return {f.name: f for f in self.figures}


# ============================================================================== helpers
def _style(strategy: str):
    return STRATEGY_STYLE.get(strategy, FALLBACK_STYLE)


def _ds(name: str) -> str:
    return DATASET_NAMES.get(name, name)


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def _fmt_budget_axis(ax, budgets: Sequence[int]):
    ax.set_xticks(list(budgets))
    ax.set_xticklabels([f"{b}" for b in budgets], fontsize=8)
    ax.set_xlabel("Extraction budget (% of corpus chunks selected for graph extraction)", fontsize=8.5, color=INK2)


def _style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


TOP_IN, BOTTOM_IN = 1.75, 0.85          # room above / below the panels, in inches (title block, legend; notes, footer)


def _new_figure(n_panels_w: int, n_panels_h: int = 1, *, panel_w=4.8, panel_h=4.0):
    w, h = max(9.0, panel_w * n_panels_w), panel_h * n_panels_h + TOP_IN + BOTTOM_IN
    fig = Figure(figsize=(w, h), layout="constrained")
    FigureCanvasAgg(fig)
    fig.get_layout_engine().set(rect=(0.0, BOTTOM_IN / h, 1.0, 1 - (TOP_IN + BOTTOM_IN) / h))
    fig.set_facecolor("#fcfcfb")
    axes = fig.subplots(n_panels_h, n_panels_w, squeeze=False)
    for ax in axes.ravel():
        _style_axes(ax)
        ax.set_facecolor("#fcfcfb")
    return fig, axes


def _decorate(fig: Figure, summary: AnalysisSummary, title: str, subtitle: str, notes: Sequence[str]):
    """Status banner, title, subtitle, watermark, notes and provenance footer, positioned in inches so they never collide
    with the panels (which live between BOTTOM_IN and TOP_IN)."""
    status, p = summary.status, summary.provenance
    w, h = fig.get_size_inches()
    top = lambda inches: 1 - inches / h               # y (figure fraction) `inches` below the top edge
    chars = int(w * 15)                               # characters per line at ~8 pt
    if status == "mock_layout_test":
        fig.text(0.5, top(0.22), f"{MOCK_BANNER}  ·  layout test only, no findings", fontsize=12, fontweight="bold",
                 color="white", ha="center", va="center", bbox=dict(facecolor=MOCK_RED, edgecolor="none", boxstyle="round,pad=0.35"))
        fig.text(0.5, 0.5, MOCK_BANNER, fontsize=min(60, 5.2 * w), fontweight="bold", color=MOCK_RED, alpha=0.22,
                 rotation=28, ha="center", va="center", zorder=100)
    elif status == "real_partial":
        fig.text(0.5, top(0.22), PARTIAL_BANNER, fontsize=11, fontweight="bold", color="white", ha="center", va="center",
                 bbox=dict(facecolor=PARTIAL_ORANGE, edgecolor="none", boxstyle="round,pad=0.35"))
        fig.text(0.5, 0.5, "PARTIAL RUN", fontsize=min(54, 4.8 * w), fontweight="bold", color=PARTIAL_ORANGE, alpha=0.12,
                 rotation=28, ha="center", va="center", zorder=100)
    fig.text(0.012, top(0.62), title, fontsize=13, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.012, top(0.86), "\n".join(textwrap.wrap(subtitle, chars)[:3]), fontsize=8.5, color=INK2, ha="left", va="top")
    prefix = "MOCK — NOT RESULTS · " if status == "mock_layout_test" else ("PARTIAL RUN · " if status == "real_partial" else "")
    foot = (f"{prefix}status: {status} · input digest {p.input_digest[:12] or 'n/a'} · git "
            f"{(p.git_commit or 'n/a')[:10]} · numpy {p.versions.get('numpy', '?')}")
    fig.text(0.012, 0.1 / h, foot, fontsize=7, color=INK2, ha="left", va="bottom")
    if notes:
        fig.text(0.012, 0.3 / h, "\n".join(textwrap.wrap(notes[0], int(chars * 1.15))[:3]), fontsize=7, color=INK2,
                 ha="left", va="bottom")


def _legend(fig, strategies: Sequence[str], extra: Sequence[Line2D] = ()):
    handles = [Line2D([0], [0], color=_style(s)[0], marker=_style(s)[1], linestyle=_style(s)[2], linewidth=1.6,
                      markersize=6, label=s) for s in strategies] + list(extra)
    _, h = fig.get_size_inches()
    fig.legend(handles=handles, loc="center right", ncol=max(1, len(handles)), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.995, 1 - 1.5 / h))


def _arms(summary: AnalysisSummary, rows: Sequence[ArmRow], dataset: str, metric: str) -> dict[str, dict[int, ArmRow]]:
    out: dict[str, dict[int, ArmRow]] = {}
    for r in rows:
        if r.arm.dataset == dataset and r.arm.metric == metric:
            out.setdefault(r.arm.strategy, {})[r.arm.budget_pct] = r
    return out


def _order(strategies) -> list[str]:
    known = [s for s in STRATEGY_STYLE if s in strategies]
    return known + sorted(set(strategies) - set(known))


def _ci_label(summary) -> str:
    return f"{summary.provenance.settings['ci_level']:.0%} bootstrap CI"


def _missing_note(rows: Sequence[ArmRow], metric: str) -> list[str]:
    n = sum(1 for r in rows if r.arm.metric == metric and r.arm.n_missing)
    return ([f"Missing data: {n} (strategy, budget) cells have questions without a measured value; their means use measured "
             "questions only (complete-case). No missing value is drawn as 0."] if n else [])


def _draw_line(ax, fig_rec: PlotFigure, panel: str, label: str, strategy: str, xs, ys, los=None, his=None, **extra):
    color, marker, ls = _style(strategy)
    y = np.array([_nan(v) for v in ys])
    x = np.array([float(v) for v in xs])
    ax.plot(x, y, color=color, marker=marker, linestyle=ls, linewidth=1.6, markersize=5.5, markeredgecolor="#fcfcfb",
            markeredgewidth=0.8, zorder=3)
    lo = hi = None
    if los is not None:
        lo, hi = np.array([_nan(v) for v in los]), np.array([_nan(v) for v in his])
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.any():
            ax.fill_between(x, lo, hi, where=ok, color=color, alpha=0.16, linewidth=0, zorder=2)
    fig_rec.series.append({"panel": panel, "label": label, "strategy": strategy, "x": [float(v) for v in xs],
                           "y": list(ys), "lo": None if los is None else list(los), "hi": None if his is None else list(his),
                           **extra})


# ============================================================================== quality vs budget
def _quality_figure(summary: AnalysisSummary, metric: str, rows: Sequence[ArmRow], name: str, title: str,
                    datasets: Sequence[str], *, ylabel: str | None = None) -> PlotFigure:
    fig, axes = _new_figure(len(datasets))
    rec = PlotFigure(name=name, title=title, status=summary.status, banner=summary.banner, fig=fig)
    strategies: set[str] = set()
    for ax, ds in zip(axes[0], datasets):
        arms = _arms(summary, rows, ds, metric)
        budgets = sorted({b for s in arms.values() for b in s})
        for strat in _order(arms):
            strategies.add(strat)
            xs = budgets
            cells = [arms[strat].get(b) for b in xs]
            nrep = max((c.arm.n_replicates for c in cells if c), default=1)
            _draw_line(ax, rec, ds, strat, strat, xs, [c.arm.mean if c else None for c in cells],
                       [c.arm.ci.low if c and c.arm.ci else None for c in cells],
                       [c.arm.ci.high if c and c.arm.ci else None for c in cells],
                       n=[c.arm.n if c else None for c in cells], n_replicates=nrep)
        _fmt_budget_axis(ax, budgets)
        ax.set_ylabel(ylabel or METRIC_LABELS.get(metric, metric), fontsize=8.5, color=INK2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{_ds(ds)}  (n = {summary.design['datasets'][ds]['n_questions']} questions)", fontsize=9.5, color=INK,
                     loc="left")
    _legend(fig, _order(strategies))
    notes = _missing_note(rows, metric)
    rec.notes = notes
    _decorate(fig, summary, title, f"Line = mean over questions; band = {_ci_label(summary)} of that mean (resampling questions). "
              "Random: each question's value is its mean over the Random shuffles.", notes)
    return rec


# ============================================================================== measures (cost / runtime / tokens)
def _measure_by_budget(summary, measure: str, dataset: str) -> dict[str, dict[int, object]]:
    out: dict[str, dict[int, object]] = {}
    for c in summary.cost:
        if c.dataset == dataset and c.measure == measure:
            out.setdefault(c.strategy, {})[c.budget_pct] = c
    return out


def _measure_figure(summary, name, title, measures: Sequence[tuple[str, str]], datasets) -> PlotFigure | None:
    if not summary.cost:
        return None
    fig, axes = _new_figure(len(datasets), len(measures), panel_h=3.4)
    rec = PlotFigure(name=name, title=title, status=summary.status, banner=summary.banner, fig=fig)
    strategies: set[str] = set()
    n_unknown = 0
    for i, (measure, ylabel) in enumerate(measures):
        for ax, ds in zip(axes[i], datasets):
            by = _measure_by_budget(summary, measure, ds)
            budgets = sorted({b for s in by.values() for b in s})
            for strat in _order(by):
                strategies.add(strat)
                cells = [by[strat].get(b) for b in budgets]
                n_unknown += sum(1 for c in cells if c is not None and c.mean is None)
                _draw_line(ax, rec, f"{ds}:{measure}", strat, strat, budgets, [c.mean if c else None for c in cells],
                           n_known=[c.n_known if c else None for c in cells], n_records=[c.n_records if c else None for c in cells])
            _fmt_budget_axis(ax, budgets or [0])
            ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
            ax.set_title(_ds(ds), fontsize=9.5, color=INK, loc="left")
    _legend(fig, _order(strategies))
    notes = ["Means over conditions with a measured value (Random: over its shuffle runs); lower bounds and unknowns are never "
             "drawn" + (f" ({n_unknown} unmeasured cells left blank)." if n_unknown else ".")]
    rec.notes = notes
    _decorate(fig, summary, title, "Each point is the mean over the measured conditions of one (strategy, budget); no interval is "
              "available for these measures. Cached extraction can make cost and runtime depend on run order.", notes)
    return rec


# ============================================================================== cost vs quality
def _cost_quality_figure(summary, metric: str, datasets) -> PlotFigure | None:
    cost = {(c.dataset, c.strategy, c.budget_pct): c for c in summary.cost if c.measure == "cost_total_usd"}
    if not cost:
        return None
    fig, axes = _new_figure(len(datasets))
    name = f"cost_vs_quality_{metric}"
    title = f"{METRIC_SHORT.get(metric, metric)} versus total cost"
    rec = PlotFigure(name=name, title=title, status=summary.status, banner=summary.banner, fig=fig)
    strategies: set[str] = set()
    omitted = 0
    for ax, ds in zip(axes[0], datasets):
        arms = _arms(summary, summary.arms, ds, metric)
        for strat in _order(arms):
            xs, ys, bs = [], [], []
            for b in sorted(arms[strat]):
                a, c = arms[strat][b], cost.get((ds, strat, b))
                if a.arm.mean is None or c is None or c.mean is None:
                    omitted += 1
                    continue
                xs.append(c.mean), ys.append(a.arm.mean), bs.append(b)
            if not xs:
                continue
            strategies.add(strat)
            color, marker, ls = _style(strat)
            ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, linewidth=1.2, markersize=5.5, markeredgecolor="#fcfcfb",
                    zorder=3)
            for x, y, b in zip(xs, ys, bs):
                ax.annotate(f"{b}%", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6.5, color=INK2)
            rec.series.append({"panel": ds, "label": strat, "strategy": strat, "x": xs, "y": ys, "budget_pct": bs})
        ax.set_xlabel("Mean total cost per condition (USD; extraction + generation + judge)", fontsize=8.5, color=INK2)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=8.5, color=INK2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(_ds(ds), fontsize=9.5, color=INK, loc="left")
    _legend(fig, _order(strategies))
    notes = [f"Labels = budget %. {omitted} (strategy, budget) cells without both a cost and a quality value are not drawn."]
    notes += _missing_note(summary.arms, metric)
    rec.notes = notes
    _decorate(fig, summary, title, "Quality is the mean over questions (no interval shown); cost is the mean over measured conditions. "
              "Cost depends on the extraction cache.", notes)
    return rec


# ============================================================================== strategy comparisons
def _comparison_figure(summary, metric: str, datasets) -> PlotFigure | None:
    rows = [c for c in summary.comparisons if c.metric == metric and c.analysis == "primary_complete_case"]
    if not rows:
        return None
    mock = summary.status == "mock_layout_test"
    per_ds = {ds: [c for c in rows if c.dataset == ds] for ds in datasets}
    height = max(len(v) for v in per_ds.values())
    fig, axes = _new_figure(len(datasets), panel_w=5.8, panel_h=0.24 * height + 1.3)
    axes = axes[0]
    name = f"comparisons_{metric}"
    title = f"Paired strategy differences in {METRIC_SHORT.get(metric, metric)}"
    rec = PlotFigure(name=name, title=title, status=summary.status, banner=summary.banner, fig=fig)
    undrawn = 0
    for ax, ds in zip(axes, datasets):
        cs = per_ds[ds]
        ys = list(range(len(cs)))[::-1]
        for y, c in zip(ys, cs):
            if c.mean_diff is None:
                undrawn += 1
                rec.series.append({"panel": ds, "label": f"{c.strategy_a} vs {c.strategy_b} @{c.budget_pct}%", "x": None,
                                   "lo": None, "hi": None, "significant_holm": c.significant_holm})
                continue
            has_ci = c.ci_low is not None and c.ci_high is not None and not mock
            color = _style(c.strategy_a)[0]
            if has_ci:
                ax.plot([c.ci_low, c.ci_high], [y, y], color=INK2, linewidth=1.1, zorder=2)
            filled = bool(c.significant_holm) and not mock
            ax.plot([c.mean_diff], [y], marker="o", markersize=5.5, color=color if filled else "#fcfcfb",
                    markeredgecolor=color, markeredgewidth=1.3, zorder=3)
            rec.series.append({"panel": ds, "label": f"{c.strategy_a} vs {c.strategy_b} @{c.budget_pct}%", "x": c.mean_diff,
                               "lo": c.ci_low if has_ci else None, "hi": c.ci_high if has_ci else None,
                               "significant_holm": c.significant_holm if not mock else None})
        ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{c.strategy_a} − {c.strategy_b} @ {c.budget_pct}%" for c in cs], fontsize=7)
        ax.set_xlabel(f"Mean paired difference A − B in {METRIC_SHORT.get(metric, metric)} (0–1 scale; > 0 favours A)",
                      fontsize=8.5, color=INK2)
        ax.set_title(_ds(ds), fontsize=9.5, color=INK, loc="left")
    if mock:
        sub = "Differences only: statistical tests are deliberately not run on mock data, so there are no intervals and no significance marks."
        extra = []
    else:
        sub = (f"Dot = mean paired difference; line = {_ci_label(summary)} of the difference. Filled dot = passes Holm at alpha = "
               f"{summary.provenance.settings['alpha']} within its family; open dot = does not. Complete-case analysis.")
        extra = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=INK2, markeredgecolor=INK2, label="passes Holm"),
                 Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#fcfcfb", markeredgecolor=INK2,
                        label="does not pass Holm")]
    if extra:
        _, hh = fig.get_size_inches()
        fig.legend(handles=extra, loc="center right", fontsize=8.5, frameon=False, ncol=2, bbox_to_anchor=(0.995, 1 - 1.5 / hh))
    notes = ([f"{undrawn} comparisons have no mean difference (skipped: missing condition or no paired questions) and are not drawn."]
             if undrawn else [])
    rec.notes = notes
    _decorate(fig, summary, title, sub, notes)
    return rec


# ============================================================================== Random shuffle variability
def _shuffle_figure(summary, metric: str, datasets) -> PlotFigure | None:
    rows = [v for v in summary.shuffle_variability if v.metric == metric]
    if not rows:
        return None
    fig, axes = _new_figure(len(datasets))
    name = f"random_variability_{metric}"
    title = f"Random strategy: variability across shuffles ({METRIC_SHORT.get(metric, metric)})"
    rec = PlotFigure(name=name, title=title, status=summary.status, banner=summary.banner, fig=fig)
    color = _style("random")[0]
    for ax, ds in zip(axes[0], datasets):
        vs = sorted((v for v in rows if v.dataset == ds), key=lambda v: v.budget_pct)
        budgets = [v.budget_pct for v in vs]
        for v in vs:
            pts = [p["mean"] for p in v.per_shuffle if p["mean"] is not None]
            ax.plot([v.budget_pct] * len(pts), pts, linestyle="", marker="o", markersize=5, color=color, alpha=0.75,
                    markeredgecolor="#fcfcfb", zorder=3)
        ax.plot(budgets, [_nan(v.mean) for v in vs], color=color, linewidth=1.4, zorder=2)
        ok = [v.min is not None and v.max is not None for v in vs]
        ax.fill_between(budgets, [_nan(v.min) for v in vs], [_nan(v.max) for v in vs], where=ok, color=color, alpha=0.14,
                        linewidth=0, zorder=1)
        rec.series.append({"panel": ds, "label": "random", "strategy": "random", "x": [float(b) for b in budgets],
                           "y": [v.mean for v in vs], "lo": [v.min for v in vs], "hi": [v.max for v in vs],
                           "per_shuffle": [v.per_shuffle for v in vs], "n_shuffles": [v.n_shuffles for v in vs]})
        _fmt_budget_axis(ax, budgets)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=8.5, color=INK2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(_ds(ds), fontsize=9.5, color=INK, loc="left")
    nsh = sorted({v.n_shuffles for v in rows})
    notes = [f"Shuffles per budget: {', '.join(map(str, nsh))}. With so few shuffles the spread is a crude estimate."]
    rec.notes = notes
    _decorate(fig, summary, title, "Dots = one Random shuffle's mean over questions; line = mean of shuffles; band = min–max of shuffles. "
              "This is Random's selection luck on the same questions, not a confidence interval.", notes)
    return rec


# ============================================================================== abstention rate
def _abstention_figure(summary, datasets) -> PlotFigure | None:
    """Abstention rate versus budget, straight from `summary.abstention` (counts and rates computed there)."""
    if not summary.abstention:
        return None
    fig, axes = _new_figure(len(datasets))
    name, title = "abstention_rate", "Abstention rate versus extraction budget"
    rec = PlotFigure(name=name, title=title, status=summary.status, banner=summary.banner, fig=fig)
    strategies: set[str] = set()
    n_undrawn = 0
    for ax, ds in zip(axes[0], datasets):
        by: dict[str, dict[int, object]] = {}
        for r in summary.abstention:
            if r.dataset == ds:
                by.setdefault(r.strategy, {})[r.budget_pct] = r
        budgets = sorted({b for s in by.values() for b in s})
        for strat in _order(by):
            strategies.add(strat)
            cells = [by[strat].get(b) for b in budgets]
            n_undrawn += sum(1 for c in cells if c is not None and c.rate is None)
            multi = any(c is not None and c.n_runs > 1 for c in cells)
            _draw_line(ax, rec, ds, strat, strat, budgets, [c.rate if c else None for c in cells],
                       [c.rate_min if c else None for c in cells] if multi else None,
                       [c.rate_max if c else None for c in cells] if multi else None,
                       n_abstained=[c.n_abstained if c else None for c in cells],
                       n_generated=[c.n_generated if c else None for c in cells],
                       n_failed=[c.n_failed if c else None for c in cells],
                       n_no_prediction=[c.n_no_prediction if c else None for c in cells],
                       n_runs=[c.n_runs if c else None for c in cells])
        _fmt_budget_axis(ax, budgets)
        ax.set_ylabel("Abstention rate (fraction of generated answers that were “not found”, 0–1)", fontsize=8, color=INK2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(_ds(ds), fontsize=9.5, color=INK, loc="left")
    _legend(fig, _order(strategies))
    notes = ["Failed generations and questions without a prediction are not in the denominator (they have no answer); "
             "an abstention scores 0 on EM, F1 and judge accuracy but is a measured answer, not missing data."
             + (f" {n_undrawn} cells had no generated answer and are not drawn." if n_undrawn else "")]
    rec.notes = notes
    _decorate(fig, summary, title, "Line = share of generated answers that were an abstention (counted, no interval). "
              "Random: mean over its shuffle runs; band = min–max over the shuffles.", notes)
    return rec


# ============================================================================== evidence
def _evidence_figure(summary, dataset: str) -> PlotFigure | None:
    metrics = summary.metrics.get("evidence") or []
    if not summary.evidence or not metrics:
        return None
    fig, axes = _new_figure(3, 2, panel_h=3.3)
    name = f"evidence_{dataset}"
    title = f"Evidence metrics — {_ds(dataset)}"
    rec = PlotFigure(name=name, title=title, status=summary.status, banner=summary.banner, fig=fig)
    strategies: set[str] = set()
    for ax, metric in zip(axes.ravel(), metrics):
        arms = _arms(summary, summary.evidence, dataset, metric)
        budgets = sorted({b for s in arms.values() for b in s})
        for strat in _order(arms):
            strategies.add(strat)
            cells = [arms[strat].get(b) for b in budgets]
            _draw_line(ax, rec, f"{dataset}:{metric}", strat, strat, budgets, [c.arm.mean if c else None for c in cells],
                       [c.arm.ci.low if c and c.arm.ci else None for c in cells],
                       [c.arm.ci.high if c and c.arm.ci else None for c in cells],
                       n=[c.arm.n if c else None for c in cells])
        _fmt_budget_axis(ax, budgets or [0])
        ax.set_xlabel("Budget (%)", fontsize=8, color=INK2)
        ax.set_ylabel("")
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=8.5, color=INK, loc="left")
        ax.set_ylim(-0.02, 1.02)
    _legend(fig, _order(strategies))
    n_undef = sum(r.arm.n_missing for r in summary.evidence if r.arm.dataset == dataset)
    notes = [f"Questions where a metric is undefined (no gold chunks, retrieval did not happen, condition not met) are left out of its mean: "
             f"{n_undef} undefined question-values in this dataset; none is drawn as 0."]
    rec.notes = notes
    _decorate(fig, summary, title, f"Gold chunks are known only after the run (post hoc). Line = mean over questions where the metric is "
              f"defined; band = {_ci_label(summary)}.", notes)
    return rec


# ============================================================================== public API
def build_figures(summary: AnalysisSummary) -> FigureSet:
    """Every figure the summary supports, plus the reasons for the ones it does not. Reads only the summary."""
    datasets = sorted(summary.design["datasets"])
    metrics = summary.metrics["analysed"]
    figs: list[PlotFigure | None] = []
    skipped: dict[str, str] = {}

    for m in ("f1", "judge_accuracy", "em"):
        if m in metrics:
            figs.append(_quality_figure(summary, m, summary.arms, f"quality_{m}", f"{METRIC_SHORT[m]} versus extraction budget",
                                        datasets))
        else:
            skipped[f"quality_{m}"] = summary.skipped_metrics.get(m, "metric not analysed")
    for m in metrics:
        f = _cost_quality_figure(summary, m, datasets)
        if f is None:
            skipped[f"cost_vs_quality_{m}"] = "no cost data in the summary"
        figs.append(f)
    figs.append(_measure_figure(summary, "cost_vs_budget", "Total cost versus extraction budget",
                                [("cost_total_usd", "Mean total cost per condition (USD)")], datasets))
    figs.append(_measure_figure(summary, "runtime_vs_budget", "Total runtime versus extraction budget",
                                [("runtime_total_s", "Mean total runtime per condition (seconds)")], datasets))
    figs.append(_measure_figure(summary, "tokens_vs_budget", "LLM tokens versus extraction budget",
                                [("tokens_total_input", "Mean input tokens per condition"),
                                 ("tokens_total_output", "Mean output tokens per condition")], datasets))
    for n in ("cost_vs_budget", "runtime_vs_budget", "tokens_vs_budget"):
        if not summary.cost:
            skipped[n] = "no cost / runtime / token data in the summary"
    for m in metrics:
        f = _comparison_figure(summary, m, datasets)
        if f is None:
            skipped[f"comparisons_{m}"] = "no paired comparisons in the summary"
        figs.append(f)
    for m in metrics:
        f = _shuffle_figure(summary, m, datasets)
        if f is None:
            skipped[f"random_variability_{m}"] = "the matrix has no Random shuffles (no random_seeds)"
        figs.append(f)
    abst = _abstention_figure(summary, datasets)
    if abst is not None:
        figs.append(abst)
    if summary.evidence:
        for ds in datasets:
            figs.append(_evidence_figure(summary, ds))
    else:
        skipped["evidence"] = "no evidence report was supplied (gold chunks are needed and are not loaded by default)"
    if abst is None:
        skipped["abstention_rate"] = ABSTENTION_REASON

    return FigureSet(status=summary.status, figures=[f for f in figs if f is not None], skipped=skipped)


def output_subdir(status: str) -> str:
    return {"mock_layout_test": MOCK_DIR, "real_partial": PARTIAL_DIR}.get(status, REAL_DIR)


def figure_dir(out_root: str | Path, status: str) -> Path:
    """Where figures of this status go. Mock, partial and complete real figures never share a folder."""
    return Path(out_root) / output_subdir(status) / "figures"


def file_stem(name: str, status: str) -> str:
    return {"mock_layout_test": "MOCK_", "real_partial": "PARTIAL_"}.get(status, "") + name


def write_figures(figset: FigureSet, out_root: str | Path, formats: Sequence[str] = FORMATS, *, dpi: int = DPI) -> list[Path]:
    """Save every figure (PNG at `dpi`, PDF) under figure_dir(out_root, status). Returns the paths written.
    Metadata that varies from run to run (creation date, producer) is left out so repeated runs give identical bytes."""
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise ValueError(f"unknown figure format(s) {bad}; expected {list(FORMATS)}")
    out = figure_dir(out_root, figset.status)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in figset.figures:
        if f.status != figset.status:
            raise ValueError(f"figure {f.name} has status {f.status} but the set is {figset.status}")
        for fmt in formats:
            p = out / f"{file_stem(f.name, f.status)}.{fmt}"
            if fmt == "png":
                f.fig.savefig(p, dpi=dpi, format="png", metadata={"Software": None})
            else:
                f.fig.savefig(p, format="pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
            paths.append(p)
    return paths
