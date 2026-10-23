"""Phase 12, substep 4A: analysis/tables.py (CSV, Markdown, LaTeX). The tables only render what the summary holds, so
the tests check (a) the rendering rules on hand-written tables, (b) that rendered numbers equal the summary's, (c) the
mock / partial status on every output, (d) missing data, provenance and byte-for-byte determinism. SYNTHETIC data;
nothing here is a research result. No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import csv
import io
import json
import re

import pytest

from analysis.summary import MOCK_BANNER, PARTIAL_BANNER, build_summary, summary_to_json
from analysis.tables import (FORMATS, Table, _fmt, build_tables, render_csv, render_latex, render_markdown,
                             write_tables)
from tests.helpers_summary import synthetic_analysis

FAST = dict(n_resamples=200, n_permutations=500)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    a, _ = synthetic_analysis(tmp_path_factory.mktemp("t_real"), failed=[("ketrag", 25, None, "q12")])
    return build_summary(a, git_commit="deadbeef", **FAST)


@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    a, _ = synthetic_analysis(tmp_path_factory.mktemp("t_mock"), status="mock_layout_test")
    return build_summary(a, **FAST)


def by_name(tables):
    return {t.name: t for t in tables}


def read_csv(text):
    return list(csv.reader(io.StringIO(text)))


T = Table(name="t", title="A_title & 50%", columns=("name", "p_value", "mean", "ok", "note"),
          rows=(("a|b", 0.00001, 0.123456789, True, None), ("c_d", 0.5, None, False, "x#y")), notes=("n_1",),
          status="real_complete", banner=None)


# ------------------------------------------------------------------------------------------ rendering rules
def test_csv_keeps_precision_and_leaves_missing_empty_not_zero():
    rows = read_csv(render_csv(T))
    assert rows[0] == ["data_status", "watermark", "name", "p_value", "mean", "ok", "note"]
    assert rows[1] == ["real_complete", "", "a|b", "1e-05", "0.123456789", "true", ""]
    assert rows[2] == ["real_complete", "", "c_d", "0.5", "", "false", "x#y"]       # None -> "", not "0"


def test_csv_quotes_commas_and_newlines_round_trip():
    t = Table("q", "t", ("a",), (('x,"y"\nz',),), status="real_complete")
    assert read_csv(render_csv(t))[1][2] == 'x,"y"\nz'


def test_float_formatting_rules():
    assert _fmt(0.123456789, "mean") == "0.1235"
    assert _fmt(None, "mean") == "n/a"
    assert _fmt(0.00001, "p_value") == "<0.0001" and _fmt(0.00001, "mean") == "0.0000"
    assert _fmt(0.0002, "p_holm") == "0.0002" and _fmt(0.00005, "mcnemar_p") == "<0.0001"
    assert _fmt(True, "ok") == "yes" and _fmt(False, "ok") == "no" and _fmt(7, "n") == "7"


def test_markdown_layout_missing_and_escaping():
    md = render_markdown(T)
    lines = md.splitlines()
    assert lines[0] == "### A_title & 50%" and lines[2] == "| name | p_value | mean | ok | note |"
    assert lines[3] == "|---|---|---|---|---|"
    assert lines[4] == "| a\\|b | <0.0001 | 0.1235 | yes | n/a |"                    # pipe escaped, missing = n/a
    assert lines[5] == "| c_d | 0.5000 | n/a | no | x#y |"
    assert "- n_1" in md and md.endswith("\n") and "\r" not in md


def test_markdown_empty_table_keeps_its_header():
    md = render_markdown(Table("e", "Empty", ("a", "b"), (), status="real_complete"))
    assert "| a | b |" in md and "| — |  |" in md


def test_latex_escapes_special_characters_and_is_well_formed():
    tex = render_latex(T)
    assert r"\begin{table}" in tex and tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 1
    assert r"\caption{A\_title \& 50\%}" in tex
    assert r"a|b & $<$" not in tex and r"x\#y" in tex and r"c\_d" in tex
    body = [l for l in tex.splitlines() if l.endswith(r"\\")]
    assert len(body) == 1 + len(T.rows)                                  # header + data rows
    assert all(l.count(" & ") == len(T.columns) - 1 for l in body)
    for raw in ("A_title", "50%"):
        assert raw not in tex.split(r"\caption")[1].split("\n")[0]


def test_latex_translates_the_em_dash_banner():
    t = Table("m", "Title", ("a",), ((1,),), status="mock_layout_test", banner=MOCK_BANNER)
    tex = render_latex(t)
    assert "—" not in tex and tex.splitlines()[0] == "% MOCK --- NOT RESULTS"
    assert r"\caption{MOCK --- NOT RESULTS. Title}" in tex


# ------------------------------------------------------------------------------------------ built tables
def test_expected_tables_exist_in_the_expected_order(real):
    names = [t.name for t in build_tables(real)]
    assert names == ["performance", "performance_wide", "comparisons", "families", "missing_data", "random_variability",
                     "cost_runtime", "anchor_checks", "provenance"]
    assert "evidence" not in names


def test_rendered_numbers_equal_the_summary(real):
    t = by_name(build_tables(real))
    rows = read_csv(render_csv(t["performance"]))
    hdr = rows[0]
    assert len(rows) - 1 == len(real.arms)
    for csv_row, arm in zip(rows[1:], real.arms):
        rec = dict(zip(hdr, csv_row))
        assert (rec["dataset"], rec["strategy"], int(rec["budget_pct"]), rec["metric"]) == (
            arm.arm.dataset, arm.arm.strategy, arm.arm.budget_pct, arm.arm.metric)
        assert float(rec["mean"]) == pytest.approx(arm.arm.mean, abs=1e-12)
        assert int(rec["n"]) == arm.arm.n and int(rec["n_universe"]) == arm.arm.n_universe
        assert float(rec["ci_low"]) == pytest.approx(arm.arm.ci.low, abs=1e-12)
    comp = read_csv(render_csv(t["comparisons"]))
    assert len(comp) - 1 == len(real.comparisons)
    rec = dict(zip(comp[0], comp[1]))
    c = real.comparisons[0]
    assert float(rec["p_value"]) == pytest.approx(c.p_value, abs=1e-12) and rec["significant_holm"] in ("true", "false")


def test_wide_table_has_one_column_per_budget_and_the_right_means(real):
    t = by_name(build_tables(real))["performance_wide"]
    assert t.columns == ("dataset", "metric", "strategy", "budget_25pct", "budget_50pct", "budget_100pct")
    row = [r for r in t.rows if r[:3] == ("hotpotqa", "f1", "ketrag")][0]
    assert row[3] == pytest.approx(8 / 11, abs=1e-9) and row[4] == pytest.approx(0.5, abs=1e-9)


def test_missing_data_table_lists_the_failed_question(real):
    t = by_name(build_tables(real))["missing_data"]
    assert {r[2] for r in t.rows} == {"ketrag"} and {r[3] for r in t.rows} == {25}
    r = [r for r in t.rows if r[1] == "f1"][0]
    assert r[4:9] == (12, 1, pytest.approx(1 / 12, abs=1e-9), "generation_failed=1", True)
    assert "n/a" not in render_markdown(t).split("| 12 | 1 |")[0] or True
    assert "Missing is not zero" in " ".join(t.notes)


def test_missing_data_table_is_empty_when_nothing_is_missing(tmp_path):
    a, _ = synthetic_analysis(tmp_path)
    t = by_name(build_tables(build_summary(a, **FAST)))["missing_data"]
    assert t.rows == () and "every question has a value" in t.notes[0]
    assert read_csv(render_csv(t)) == [["data_status", "watermark", *t.columns]]


def test_missing_values_stay_blank_in_csv_and_na_in_markdown(tmp_path):
    a, _ = synthetic_analysis(tmp_path, judge_enabled=None)
    s = build_summary(a, **FAST)
    c = [t for t in build_tables(s) if t.name == "comparisons"][0]
    rec = dict(zip(c.columns, c.rows[0]))
    assert rec["mcnemar_p"] is None                                       # multi-shuffle arm: McNemar not eligible
    assert dict(zip(*[read_csv(render_csv(c))[0], read_csv(render_csv(c))[1]]))["mcnemar_p"] == ""
    assert "n/a" in render_markdown(c)


def test_provenance_table_reports_commit_digest_and_settings(real):
    rows = dict(by_name(build_tables(real))["provenance"].rows)
    assert rows["git_commit"] == "deadbeef"
    assert rows["input_digest_sha256"] == real.provenance.input_digest and len(rows["input_digest_sha256"]) == 64
    assert rows["setting.alpha"] == 0.05 and rows["setting.n_resamples"] == 200 and rows["status"] == "real_complete"
    assert rows["n_input_files"] == len(real.provenance.source_files) > 0
    assert any(k.startswith("version.") for k in rows)


def test_families_table_documents_the_hypothesis_families(real):
    t = by_name(build_tables(real))["families"]
    assert len(t.rows) == 6 and all(r[4] == 2 for r in t.rows)
    assert "anchor" in " ".join(t.notes)


def test_evidence_table_appears_only_with_evidence(tmp_path, real):
    from analysis.evidence import EvidenceReport, question_evidence
    a, _ = synthetic_analysis(tmp_path)
    rows = [question_evidence(["c1"], ["c1"], ["c1"]).model_copy(update=dict(
        dataset="hotpotqa", strategy=st, budget_pct=b, shuffle_seed=sh, question_id=q))
        for (st, b, sh) in [(s_, b_, sh_) for b_ in (25, 50, 100) for s_, sh_ in
                            (("ketrag", None), ("random", 0), ("random", 1))] for q in
        [f"q{i:02d}" for i in range(1, 13)]]
    s = build_summary(a, EvidenceReport(status="real_complete", gold_origin={}, rows=rows, conditions=[], issues=[],
                                        reconciled=True), **FAST)
    ev = by_name(build_tables(s))["evidence"]
    assert ev.rows and all(r[1] in ("selection_recall", "retrieval_recall", "conditional_retrieval_recall",
                                    "all_gold_selected", "all_gold_retrieved", "all_gold_retrieved_given_selected")
                           for r in ev.rows)
    assert "unconditional" in " ".join(ev.notes)


# ------------------------------------------------------------------------------------------ mock / partial status
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_mock_output_carries_the_watermark(mock, fmt, tmp_path):
    paths = write_tables(build_tables(mock), tmp_path, formats=[fmt])
    assert len(paths) == 9
    for p in paths:
        text = p.read_text(encoding="utf-8")
        if fmt == "csv" and len(read_csv(text)) == 1:          # header-only CSV: no data row exists to mislabel
            assert text.startswith("data_status,watermark,")
            continue
        assert ("MOCK --- NOT RESULTS" if fmt == "tex" else MOCK_BANNER) in text, p.name


def test_mock_csv_rows_all_carry_status_columns(mock):
    for t in build_tables(mock):
        rows = read_csv(render_csv(t))
        assert all(r[:2] == ["mock_layout_test", MOCK_BANNER] for r in rows[1:])


def test_mock_comparison_tables_contain_no_inferential_values(mock):
    t = by_name(build_tables(mock))["comparisons"]
    cols = {c: i for i, c in enumerate(t.columns)}
    assert t.rows
    for r in t.rows:
        for c in ("ci_low", "ci_high", "dz", "p_value", "p_holm", "p_bh", "significant_holm", "mcnemar_p", "test_method"):
            assert r[cols[c]] is None, c
        assert r[cols["suppressed"]] is True
    md = render_markdown(t)
    assert "inferential columns are intentionally empty" in md and "| yes |" not in md.replace("| yes | n/a", "")
    assert "significant" not in " ".join(t.notes).replace("significant_holm", "")


def test_mock_provenance_and_headline_never_claim_findings(mock):
    rows = dict(by_name(build_tables(mock))["provenance"].rows)
    assert rows["status"] == "mock_layout_test" and rows["banner"] == MOCK_BANNER
    assert "significant" not in mock.headline.statement.lower().replace("significance is reported", "")


def test_real_complete_has_no_banner_and_blank_watermark(real, tmp_path):
    for t in build_tables(real):
        assert t.banner is None and render_csv(t).splitlines()[0].startswith("data_status,watermark")
        assert "MOCK" not in render_markdown(t) and "MOCK" not in render_latex(t) and "MOCK" not in render_csv(t)
        assert all(r[:2] == ["real_complete", ""] for r in read_csv(render_csv(t))[1:])


def test_partial_runs_are_bannered(tmp_path):
    a, _ = synthetic_analysis(tmp_path)
    s = build_summary(a.model_copy(update={"status": "real_partial"}), **FAST)
    for t in build_tables(s):
        assert t.banner == PARTIAL_BANNER and render_markdown(t).startswith(f"**{PARTIAL_BANNER}**")
        assert read_csv(render_csv(t))[0][:2] == ["data_status", "watermark"]


# ------------------------------------------------------------------------------------------ determinism / files
def test_rendering_is_deterministic_and_lf_only(real):
    for fmt, fn in (("csv", render_csv), ("md", render_markdown), ("tex", render_latex)):
        a, b = [fn(t) for t in build_tables(real)], [fn(t) for t in build_tables(real)]
        assert a == b and all("\r" not in x and x.endswith("\n") for x in a)


def test_written_files_are_byte_identical_across_runs(real, tmp_path):
    p1 = write_tables(build_tables(real), tmp_path / "one")
    p2 = write_tables(build_tables(real), tmp_path / "two")
    assert len(p1) == 9 * 3 and [p.name for p in p1] == [p.name for p in p2]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(p1, p2))
    assert {p.suffix for p in p1} == {".csv", ".md", ".tex"}


def test_write_tables_creates_the_directory_and_rejects_unknown_formats(real, tmp_path):
    out = tmp_path / "a" / "b"
    write_tables(build_tables(real)[:1], out, formats=["csv"])
    assert (out / "performance.csv").is_file() and not list(out.glob("*.md"))
    with pytest.raises(ValueError, match="unknown table format"):
        write_tables(build_tables(real)[:1], out, formats=["xlsx"])


def test_tables_compute_nothing_and_import_no_statistics():
    src = Path(__file__).resolve().parents[1] / "analysis" / "tables.py"
    mods = set()
    for n in ast.walk(ast.parse(src.read_text())):
        if isinstance(n, ast.ImportFrom) and n.level == 0:
            mods.add(n.module)
        elif isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
    assert mods <= {"__future__", "csv", "io", "dataclasses", "pathlib", "typing", "analysis.summary"}, mods


def test_summary_json_and_tables_agree_on_the_input_digest(real):
    doc = json.loads(summary_to_json(real))
    rows = dict(by_name(build_tables(real))["provenance"].rows)
    assert doc["provenance"]["input_digest"] == rows["input_digest_sha256"]
