"""Phase 12, substep 4B: analysis/plots.py. The plotted data is checked against the summary AND against the matplotlib
artists that were actually drawn; missing values, status marking (mock / partial), file formats and output-folder
separation each have their own tests. SYNTHETIC data; nothing here is a research result. No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import math

import numpy as np
import pytest
from PIL import Image

from analysis import stats as st
from analysis.evidence import EvidenceReport, question_evidence
from analysis.plots import (ABSTENTION_REASON, DPI, FigureSet, PlotFigure, build_figures, figure_dir, file_stem,
                            output_subdir, write_figures)
from analysis.summary import MOCK_BANNER, PARTIAL_BANNER, build_summary
from tests.helpers_summary import DS, QIDS, f1_table, synthetic_analysis

FAST = dict(n_resamples=200, n_permutations=500)
EXPECTED = ["quality_f1", "quality_judge_accuracy", "quality_em", "cost_vs_quality_f1", "cost_vs_quality_judge_accuracy",
            "cost_vs_quality_em", "cost_vs_budget", "runtime_vs_budget", "tokens_vs_budget", "comparisons_f1",
            "comparisons_judge_accuracy", "comparisons_em", "random_variability_f1", "random_variability_judge_accuracy",
            "random_variability_em", "abstention_rate", "evidence_hotpotqa"]


def _evidence():
    rows = []
    for (strat, b, sh) in f1_table():
        for q in QIDS:
            sel = ["c1", "c3"] if q == "q12" else ["c1", "c2", "c3", "c4"]
            rows.append(question_evidence(["c1", "c2"], sel, ["c1"]).model_copy(
                update=dict(dataset=DS, strategy=strat, budget_pct=b, shuffle_seed=sh, question_id=q)))
    return EvidenceReport(status="x", gold_origin={}, rows=rows, conditions=[], issues=[], reconciled=True)


def _summary(root, *, status=None, partial=False, failed=(), evidence=True, **kw):
    a, _ = synthetic_analysis(root, status=status, failed=failed, **kw)
    if partial:
        a = a.model_copy(update={"status": "real_partial"})
    return build_summary(a, _evidence() if evidence else None, **FAST)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    return _summary(tmp_path_factory.mktemp("p_real"), failed=[("ketrag", 25, None, "q12")])


@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    return _summary(tmp_path_factory.mktemp("p_mock"), status="mock_layout_test")


@pytest.fixture(scope="module")
def partial(tmp_path_factory):
    return _summary(tmp_path_factory.mktemp("p_part"), partial=True)


@pytest.fixture(scope="module")
def real_figs(real):
    return build_figures(real)


@pytest.fixture(scope="module")
def mock_figs(mock):
    return build_figures(mock)


@pytest.fixture(scope="module")
def partial_figs(partial):
    return build_figures(partial)


def texts(fig):
    return [t.get_text() for t in fig.texts]


def arm(s, strat, b, metric):
    (r,) = [r for r in s.arms if (r.arm.strategy, r.arm.budget_pct, r.arm.metric) == (strat, b, metric)]
    return r.arm


# ------------------------------------------------------------------------------------------ what is built
def test_every_supported_figure_is_built_in_order_and_nothing_is_skipped(real_figs):
    # 4D: the abstention rate is now supported by the summary, so it is drawn (4B listed it as skipped)
    assert [f.name for f in real_figs.figures] == EXPECTED
    assert real_figs.skipped == {}
    assert "new statistic" in ABSTENTION_REASON and all(isinstance(f, PlotFigure) for f in real_figs.figures)


def test_evidence_figures_need_an_evidence_report(tmp_path):
    s = _summary(tmp_path, evidence=False)
    fs = build_figures(s)
    assert "evidence_hotpotqa" not in fs.by_name() and "gold chunks are needed" in fs.skipped["evidence"]


def test_judge_disabled_skips_judge_figures_with_the_reason(tmp_path):
    s = _summary(tmp_path, judge_enabled=False)
    fs = build_figures(s)
    assert not [n for n in fs.by_name() if "judge_accuracy" in n]
    assert fs.skipped["quality_judge_accuracy"] == "the judge was disabled in the matrix"


def test_no_random_shuffles_skips_the_variability_figures(real):
    s = real.model_copy(update={"shuffle_variability": []})
    fs = build_figures(s)
    assert not [n for n in fs.by_name() if n.startswith("random_variability")]
    assert all("no Random shuffles" in fs.skipped[f"random_variability_{m}"] for m in ("f1", "judge_accuracy", "em"))


# ------------------------------------------------------------------------------------------ plotted data == summary
def test_quality_series_equal_the_summary_and_the_drawn_lines(real, real_figs):
    f = real_figs.by_name()["quality_f1"]
    by = {x["strategy"]: x for x in f.series}
    assert set(by) == {"random", "ketrag"}
    for strat, ser in by.items():
        assert ser["x"] == [25.0, 50.0, 100.0]
        for i, b in enumerate((25, 50, 100)):
            a = arm(real, strat, b, "f1")
            assert ser["y"][i] == a.mean and ser["lo"][i] == a.ci.low and ser["hi"][i] == a.ci.high and ser["n"][i] == a.n
    assert by["random"]["n_replicates"] == 2 and by["ketrag"]["n_replicates"] == 1
    # the matplotlib artists carry the same numbers
    ax = f.fig.axes[0]
    drawn = {tuple(np.round(l.get_ydata(), 9)) for l in ax.lines}
    for ser in by.values():
        assert tuple(np.round(ser["y"], 9)) in drawn
    assert len(ax.collections) == 2                                          # one CI band per strategy
    assert ax.get_ylim() == (-0.02, 1.02)


def test_hand_calculated_values_appear_on_the_figure(real_figs):
    ser = {x["strategy"]: x for x in real_figs.by_name()["quality_f1"].series}
    assert ser["random"]["y"][0] == pytest.approx(5 / 12, abs=1e-9)           # hand calculation in tests/helpers_summary.py
    assert ser["ketrag"]["y"][0] == pytest.approx(8 / 11, abs=1e-9)           # q12 failed: dropped, not scored 0


def test_axes_titles_units_and_legend_are_labelled(real, real_figs):
    f = real_figs.by_name()["quality_f1"]
    ax = f.fig.axes[0]
    assert "Extraction budget" in ax.get_xlabel() and "%" in ax.get_xlabel()
    assert "F1" in ax.get_ylabel() and "0–1" in ax.get_ylabel()
    assert "HotpotQA" in ax.get_title(loc="left") and "n = 12 questions" in ax.get_title(loc="left")
    assert [t.get_text() for t in f.fig.legends[0].get_texts()] == ["random", "ketrag"]
    assert f.title in texts(f.fig) and any("95% bootstrap CI" in t for t in texts(f.fig))
    assert "Judge accuracy" in real_figs.by_name()["quality_judge_accuracy"].fig.axes[0].get_ylabel()
    assert "Exact match" in real_figs.by_name()["quality_em"].fig.axes[0].get_ylabel()


def test_cost_runtime_and_token_figures_use_summary_means_and_label_units(real, real_figs):
    cost = {(c.strategy, c.budget_pct, c.measure): c for c in real.cost}
    for name, measure, unit in (("cost_vs_budget", "cost_total_usd", "USD"), ("runtime_vs_budget", "runtime_total_s", "seconds")):
        f = real_figs.by_name()[name]
        assert unit in f.fig.axes[0].get_ylabel()
        for ser in f.series:
            want = [cost[(ser["strategy"], int(b), measure)].mean for b in ser["x"]]
            assert ser["y"] == want
    t = real_figs.by_name()["tokens_vs_budget"]
    assert len(t.fig.axes) == 2 and "input tokens" in t.fig.axes[0].get_ylabel() and "output tokens" in t.fig.axes[1].get_ylabel()
    assert {s["panel"] for s in t.series} == {f"{DS}:tokens_total_input", f"{DS}:tokens_total_output"}


def test_cost_vs_quality_pairs_each_cost_with_its_own_quality(real, real_figs):
    f = real_figs.by_name()["cost_vs_quality_f1"]
    cost = {(c.strategy, c.budget_pct): c.mean for c in real.cost if c.measure == "cost_total_usd"}
    assert "USD" in f.fig.axes[0].get_xlabel()
    for ser in f.series:
        for x, y, b in zip(ser["x"], ser["y"], ser["budget_pct"]):
            assert x == cost[(ser["strategy"], b)] and y == arm(real, ser["strategy"], b, "f1").mean


def test_comparison_figure_plots_summary_differences_and_marks_holm_passes(real):
    rows = [c.model_copy(update={"significant_holm": True}) if (c.budget_pct == 25 and c.metric == "f1"
            and c.analysis == "primary_complete_case") else c for c in real.comparisons]
    f = build_figures(real.model_copy(update={"comparisons": rows})).by_name()["comparisons_f1"]
    prim = [c for c in rows if c.metric == "f1" and c.analysis == "primary_complete_case"]
    assert [s["x"] for s in f.series] == [c.mean_diff for c in prim]
    assert [s["lo"] for s in f.series] == [c.ci_low for c in prim]
    assert [s["significant_holm"] for s in f.series] == [True, False]
    dots = [l for l in f.fig.axes[0].lines if l.get_marker() == "o"]
    assert dots[0].get_markerfacecolor() != "#fcfcfb" and dots[1].get_markerfacecolor() == "#fcfcfb"   # filled vs open
    assert "Holm" in " ".join(texts(f.fig)) and "A − B" in f.fig.axes[0].get_xlabel()
    assert all("sensitivity" not in s["label"] for s in f.series)                # only the primary analysis is plotted


def test_random_variability_is_a_separate_figure_from_the_confidence_bands(real, real_figs):
    f = real_figs.by_name()["random_variability_f1"]
    v = [v for v in real.shuffle_variability if v.metric == "f1" and v.budget_pct == 25][0]
    ser = f.series[0]
    i = ser["x"].index(25.0)
    assert ser["y"][i] == v.mean and ser["lo"][i] == v.min and ser["hi"][i] == v.max
    assert [p["mean"] for p in ser["per_shuffle"][i]] == [p["mean"] for p in v.per_shuffle]
    assert ser["n_shuffles"][i] == 2
    ax = f.fig.axes[0]
    dots = [l for l in ax.lines if l.get_linestyle() == "None"]
    assert sorted(y for l in dots if list(l.get_xdata())[0] == 25 for y in l.get_ydata()) == pytest.approx(
        sorted(p["mean"] for p in v.per_shuffle))
    assert "not a confidence interval" in " ".join(texts(f.fig)) and "crude" in " ".join(texts(f.fig))
    assert not any(s["strategy"] != "random" for s in f.series)


def test_evidence_figure_draws_every_evidence_metric_with_its_summary_values(real, real_figs):
    f = real_figs.by_name()["evidence_hotpotqa"]
    assert len(f.fig.axes) == 6
    want = [r for r in real.evidence if r.arm.metric == "selection_recall" and r.arm.strategy == "ketrag"]
    got = [s for s in f.series if s["panel"].endswith("selection_recall") and s["strategy"] == "ketrag"][0]
    assert got["y"] == [r.arm.mean for r in sorted(want, key=lambda r: r.arm.budget_pct)]
    assert "post hoc" in " ".join(texts(f.fig))


# ------------------------------------------------------------------------------------------ missing data
def test_a_missing_mean_is_not_drawn_and_never_becomes_zero(real):
    rows = [r.model_copy(update={"arm": r.arm.model_copy(update={"mean": None, "ci": None})})
            if (r.arm.strategy, r.arm.budget_pct, r.arm.metric) == ("ketrag", 50, "f1") else r for r in real.arms]
    f = build_figures(real.model_copy(update={"arms": rows})).by_name()["quality_f1"]
    ser = [s for s in f.series if s["strategy"] == "ketrag"][0]
    assert ser["y"][1] is None and ser["lo"][1] is None and ser["hi"][1] is None
    line = [l for l in f.fig.axes[0].lines if l.get_color() == "#eb6834"][0]
    y = np.asarray(line.get_ydata(), dtype=float)
    assert math.isnan(y[1]) and not (y == 0).any()                         # a gap, not a 0
    assert y[0] == pytest.approx(8 / 11, abs=1e-9)


def test_cost_vs_quality_omits_points_without_both_values_and_says_so(real):
    cost = [c.model_copy(update={"mean": None, "n_known": 0, "n_unknown": c.n_records})
            if (c.strategy, c.budget_pct, c.measure) == ("ketrag", 50, "cost_total_usd") else c for c in real.cost]
    f = build_figures(real.model_copy(update={"cost": cost})).by_name()["cost_vs_quality_f1"]
    ser = [s for s in f.series if s["strategy"] == "ketrag"][0]
    assert 50 not in ser["budget_pct"] and None not in ser["x"] and None not in ser["y"]
    assert any("1 (strategy, budget) cells without both" in n for n in f.notes)


def test_measure_figures_leave_unmeasured_cells_blank(real):
    cost = [c.model_copy(update={"mean": None}) if (c.strategy, c.budget_pct, c.measure) == ("random", 25, "runtime_total_s")
            else c for c in real.cost]
    f = build_figures(real.model_copy(update={"cost": cost})).by_name()["runtime_vs_budget"]
    ser = [s for s in f.series if s["strategy"] == "random"][0]
    assert ser["y"][0] is None and "1 unmeasured cells left blank" in f.notes[0]


def test_attrition_is_announced_on_the_quality_figure(real_figs, real):
    f = real_figs.by_name()["quality_f1"]
    assert any("Missing data" in n and "No missing value is drawn as 0" in n for n in f.notes)
    assert any("Missing data" in t for t in texts(f.fig))
    assert not any("Missing data" in n for n in build_figures(real.model_copy(update={"arms": [
        r.model_copy(update={"arm": r.arm.model_copy(update={"n_missing": 0})}) for r in real.arms]})).by_name()["quality_f1"].notes)


def test_skipped_comparisons_are_not_drawn(real):
    rows = [c.model_copy(update={"mean_diff": None, "ci_low": None, "ci_high": None, "skipped_reason": "x"})
            if (c.budget_pct == 50 and c.metric == "f1") else c for c in real.comparisons]
    f = build_figures(real.model_copy(update={"comparisons": rows})).by_name()["comparisons_f1"]
    assert [s["x"] is None for s in f.series] == [False, True] and any("not drawn" in n for n in f.notes)
    assert len([l for l in f.fig.axes[0].lines if l.get_marker() == "o"]) == 1


# ------------------------------------------------------------------------------------------ status marking
def test_mock_figures_carry_the_watermark_banner_and_footer(mock_figs):
    assert len(mock_figs.figures) == 17
    for f in mock_figs.figures:
        ts = [t for t in f.fig.texts if MOCK_BANNER in t.get_text()]
        assert len(ts) >= 3, f.name                                         # banner, diagonal watermark, footer
        wm = [t for t in ts if t.get_text() == MOCK_BANNER][0]
        assert wm.get_fontsize() >= 30 and wm.get_rotation() != 0 and 0 < wm.get_alpha() < 1
        assert f.banner == MOCK_BANNER and f.status == "mock_layout_test"


def _red_pixels(fig, dpi=60):
    fig.savefig(buf := __import__("io").BytesIO(), format="png", dpi=dpi)
    buf.seek(0)
    a = np.asarray(Image.open(buf).convert("RGB")).astype(int)
    return int(((a[..., 0] > 170) & (a[..., 1] < 70) & (a[..., 2] < 70)).sum())


def test_mock_watermark_is_visible_in_the_pixels_and_absent_on_real(mock_figs, real_figs):
    assert _red_pixels(mock_figs.by_name()["quality_f1"].fig) > 500
    assert _red_pixels(real_figs.by_name()["quality_f1"].fig) == 0


def test_mock_comparison_figure_shows_no_intervals_and_no_significance(mock, mock_figs):
    for m in ("f1", "judge_accuracy", "em"):
        f = mock_figs.by_name()[f"comparisons_{m}"]
        assert all(s["lo"] is None and s["hi"] is None and s["significant_holm"] is None for s in f.series)
        assert not any(l.get_markerfacecolor() not in ("#fcfcfb",) for l in f.fig.axes[0].lines if l.get_marker() == "o")
        txt = " ".join(texts(f.fig)) + " ".join(t.get_text() for t in f.fig.legends and f.fig.legends[0].get_texts() or [])
        assert "deliberately not run" in txt and "passes Holm" not in txt and "significan" not in txt.replace("no significance marks", "")
        assert [s["x"] for s in f.series] == [c.mean_diff for c in mock.comparisons if c.metric == m
                                                and c.analysis == "primary_complete_case"]
    assert not any("significan" in t.lower().replace("no significance marks", "") for f in mock_figs.figures for t in texts(f.fig))


def test_partial_runs_are_marked_and_real_complete_is_not(partial_figs, real_figs):
    for f in partial_figs.figures:
        assert PARTIAL_BANNER in texts(f.fig) and "PARTIAL RUN" in texts(f.fig) and f.banner == PARTIAL_BANNER
        assert any(t.startswith("PARTIAL RUN · status: real_partial") for t in texts(f.fig))
    for f in real_figs.figures:
        assert f.banner is None and not any("MOCK" in t or "PARTIAL" in t for t in texts(f.fig))
        assert any(t.startswith("status: real_complete · input digest ") for t in texts(f.fig))


# ------------------------------------------------------------------------------------------ files and folders
def test_png_is_300_dpi_and_pdf_is_written(real_figs, tmp_path):
    assert DPI == 300
    sub = FigureSet(real_figs.status, real_figs.figures[:2], {})
    paths = write_figures(sub, tmp_path)
    assert sorted(p.suffix for p in paths) == [".pdf", ".pdf", ".png", ".png"]
    for p in paths:
        if p.suffix == ".png":
            im = Image.open(p)
            w, h = [round(x) for x in real_figs.figures[0].fig.get_size_inches()] if p.stem == real_figs.figures[0].name else (None, None)
            assert tuple(round(d) for d in im.info["dpi"]) == (300, 300)
            fig = [f for f in sub.figures if f.name == p.stem][0].fig
            assert im.size == (round(fig.get_figwidth() * 300), round(fig.get_figheight() * 300))
        else:
            assert p.read_bytes()[:5] == b"%PDF-" and b"%%EOF" in p.read_bytes()[-64:]


def test_formats_can_be_restricted_and_unknown_formats_are_rejected(real_figs, tmp_path):
    sub = FigureSet(real_figs.status, real_figs.figures[:1], {})
    assert [p.suffix for p in write_figures(sub, tmp_path, formats=["pdf"])] == [".pdf"]
    with pytest.raises(ValueError, match="unknown figure format"):
        write_figures(sub, tmp_path, formats=["svg"])


def test_files_are_byte_identical_across_runs(real_figs, tmp_path):
    sub = FigureSet(real_figs.status, real_figs.figures[:3], {})
    a, b = write_figures(sub, tmp_path / "a"), write_figures(sub, tmp_path / "b")
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(a, b))


def test_mock_figures_go_only_to_the_mock_directory(mock_figs, tmp_path):
    paths = write_figures(mock_figs, tmp_path, dpi=40)
    mock_dir = tmp_path / "mock_layout_test" / "figures"
    assert len(paths) == 34 and all(p.parent == mock_dir for p in paths)
    assert all(p.name.startswith("MOCK_") for p in paths)
    assert [p.name for p in tmp_path.iterdir()] == ["mock_layout_test"]       # nothing in a real/ or any other folder
    assert not (tmp_path / "real").exists() and not (tmp_path / "figures").exists()


def test_real_partial_and_mock_never_share_a_folder(real_figs, partial_figs, mock_figs, tmp_path):
    for fs in (real_figs, partial_figs, mock_figs):
        write_figures(FigureSet(fs.status, fs.figures[:1], {}), tmp_path, formats=["pdf"], dpi=40)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.pdf")) == [
        "mock_layout_test/figures/MOCK_quality_f1.pdf", "real/figures/quality_f1.pdf",
        "real_partial/figures/PARTIAL_quality_f1.pdf"]
    assert len({output_subdir(s) for s in ("real_complete", "real_partial", "mock_layout_test")}) == 3
    assert figure_dir("out", "mock_layout_test") == Path("out/mock_layout_test/figures")
    assert figure_dir("out", "real_complete") == Path("out/real/figures")
    assert file_stem("x", "mock_layout_test") == "MOCK_x" and file_stem("x", "real_complete") == "x"


def test_a_figure_of_another_status_cannot_be_smuggled_into_a_set(mock_figs, real_figs, tmp_path):
    bad = FigureSet("real_complete", [mock_figs.figures[0]], {})
    with pytest.raises(ValueError, match="has status mock_layout_test"):
        write_figures(bad, tmp_path)
    assert not list(tmp_path.rglob("*.png"))
    bad2 = FigureSet("mock_layout_test", [real_figs.figures[0]], {})
    with pytest.raises(ValueError):
        write_figures(bad2, tmp_path)
    assert not list((tmp_path / "mock_layout_test").rglob("*.png")) if (tmp_path / "mock_layout_test").exists() else True


# ------------------------------------------------------------------------------------------ no new statistics
def test_plots_imports_no_statistics_module_and_no_pyplot():
    src = Path(__file__).resolve().parents[1] / "analysis" / "plots.py"
    mods = set()
    for n in ast.walk(ast.parse(src.read_text())):
        if isinstance(n, ast.ImportFrom) and n.level == 0:
            mods.add(n.module)
        elif isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
    assert mods <= {"__future__", "textwrap", "dataclasses", "pathlib", "typing", "matplotlib", "numpy",
                    "matplotlib.backends.backend_agg", "matplotlib.figure", "matplotlib.lines", "analysis.summary"}, mods
    assert not any("pyplot" in m for m in mods)


def test_building_figures_runs_no_statistical_test(real, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("statistics were computed in the plotting module")
    for fn in ("bootstrap_mean_ci", "sign_flip_test", "compare_paired", "describe_arm", "holm_adjust", "paired_effect"):
        monkeypatch.setattr(st, fn, boom)
    assert len(build_figures(real).figures) == 17


def test_building_figures_does_not_change_the_summary(real):
    before = real.model_dump_json()
    build_figures(real)
    assert real.model_dump_json() == before


def test_a_genuinely_loaded_mock_matrix_gives_watermarked_figures(tmp_path):
    from analysis.loader import load_analysis_input
    from tests.helpers_analysis import mock_matrix
    a = load_analysis_input(mock_matrix(tmp_path / "m", random_seeds=[0, 1]), allow_mock_layout_test=True)
    fs = build_figures(build_summary(a, **FAST))
    assert fs.status == "mock_layout_test" and fs.figures
    assert all(MOCK_BANNER in texts(f.fig) for f in fs.figures)


def test_mock_plots_never_draw_inference_even_if_a_summary_wrongly_contains_it(real):
    """Defence in depth: the summary already suppresses inference for mock data, but if a hand-edited or buggy summary
    carried intervals and significance flags under a mock status, the figure must still not show them."""
    rows = [c.model_copy(update={"significant_holm": True}) for c in real.comparisons]
    s = real.model_copy(update={"status": "mock_layout_test", "banner": MOCK_BANNER, "comparisons": rows})
    f = build_figures(s).by_name()["comparisons_f1"]
    assert all(x["lo"] is None and x["hi"] is None and x["significant_holm"] is None for x in f.series)
    dots = [l for l in f.fig.axes[0].lines if l.get_marker() == "o"]
    assert dots and all(l.get_markerfacecolor() == "#fcfcfb" for l in dots)               # none filled
    assert len([l for l in f.fig.axes[0].lines if l.get_marker() in (None, "None", "")]) == 1   # only the zero line, no CI bars
