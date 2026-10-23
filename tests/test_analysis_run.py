"""Phase 12, substep 4C: the command line (python -m analysis.run) and the real-benchmark config TEMPLATE.
Every matrix here is produced by the free Phase 10 runner with test doubles ("real-looking" ones are SYNTHETIC, see
tests/helpers_analysis.py) or with the mock backends. A socket guard makes any network attempt fail the test; nothing
here calls an API, downloads data or runs the real configuration. Results say nothing about the research question."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import hashlib
import json
import shutil
import socket
import subprocess

import pytest
import yaml

from analysis import run as run_mod
from analysis.loader import load_analysis_input
from analysis.plots import output_subdir
from analysis.run import (EXIT_OK, EXIT_STOP, EXIT_WARN, Validation, main, parse_args, select_metrics, UsageError,
                          validate_input)
from analysis.summary import MOCK_BANNER, PARTIAL_BANNER, build_summary
from src.corpus import build_chunk_manifest
from tests.helpers_analysis import (Q1_TEXT, cond_dir, cond_ids, copy_matrix, edit_json, file_hashes, mock_matrix,
                                    priced_components, real_matrix)
from tests.helpers_summary import synthetic_analysis

PROJECT = Path(__file__).resolve().parents[1]
FAST = ["--n-resamples", "200", "--n-permutations", "500", "--dpi", "50"]
MOCK_CHUNKS, _q, _d = build_chunk_manifest("hotpotqa", "mock", 3, 42, 250, 40)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_a, **_k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture(scope="module")
def real_m(tmp_path_factory):
    return real_matrix(tmp_path_factory.mktemp("cli_real"), strategies=("random", "ketrag"), budgets=(0.25, 0.5, 1.0),
                       random_seeds=[0, 1])


@pytest.fixture(scope="module")
def mock_m(tmp_path_factory):
    return mock_matrix(tmp_path_factory.mktemp("cli_mock"))


@pytest.fixture
def real_copy(real_m, tmp_path):
    return copy_matrix(real_m, tmp_path / "rm")


@pytest.fixture
def mock_copy(mock_m, tmp_path):
    return copy_matrix(mock_m, tmp_path / "mm")


def go(matrix, out, *extra, fast=True):
    return main(["--matrix-dir", str(matrix), "--out-dir", str(out), *(FAST if fast else []), *extra])


def tree(root):
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file())


def summary_of(out, sub="real"):
    return json.loads((Path(out) / sub / "summary.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ arguments / exit codes
@pytest.mark.parametrize("extra", [
    ["--alpha", "0"], ["--alpha", "1"], ["--alpha", "-0.1"], ["--alpha", "1.5"],
    ["--n-resamples", "10"], ["--n-permutations", "99"], ["--seed", "-1"], ["--dpi", "10"], ["--dpi", "5000"],
    ["--metrics", "bleu"], ["--metrics", "f1,f1"], ["--metrics", ""], ["--formats", "xlsx"], ["--formats", ""],
    ["--figure-formats", "svg"], ["--figure-formats", "png,gif"],
])
def test_invalid_values_stop_with_exit_2_and_write_nothing(real_m, tmp_path, capsys, extra):
    out = tmp_path / "out"
    assert go(real_m, out, *extra, fast=False) == EXIT_STOP
    assert "STOPPED" in capsys.readouterr().err and not out.exists()


def test_exactly_one_input_is_required(real_m, tmp_path, capsys):
    assert main([]) == EXIT_STOP and "exactly one of" in capsys.readouterr().err
    assert main(["--matrix-dir", str(real_m), "--config", "x.yaml"]) == EXIT_STOP


@pytest.mark.parametrize("argv", [["--bogus"], ["--matrix-dir"], ["--evidence", "maybe", "--matrix-dir", "x"],
                                  ["--alpha", "abc", "--matrix-dir", "x"]])
def test_argparse_errors_become_exit_2(argv, capsys):
    assert main(argv) == EXIT_STOP


def test_help_exits_zero_and_lists_every_option(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--validate-only", "--allow-incomplete", "--allow-mock-layout-test", "--no-figures", "--metrics", "--alpha",
                 "--n-resamples", "--n-permutations", "--seed", "--evidence", "--out-dir"):
        assert flag in text


def test_a_missing_matrix_directory_stops_cleanly(tmp_path, capsys):
    assert main(["--matrix-dir", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "o")]) == EXIT_STOP
    assert "STOPPED" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_module_entry_point_and_exit_codes_via_subprocess(tmp_path):
    env = {"PATH": "/usr/bin:/usr/local/bin", "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "analysis.run"], cwd=PROJECT, env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "exactly one of" in r.stderr
    r = subprocess.run([sys.executable, "-m", "analysis.run", "--help"], cwd=PROJECT, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "--validate-only" in r.stdout


# ------------------------------------------------------------------------------------------ a clean real run
def test_clean_real_run_writes_the_expected_files_and_exits_zero(real_m, tmp_path, capsys):
    out = tmp_path / "out"
    assert go(real_m, out) == EXIT_OK
    files = tree(out)
    assert [f for f in files if "/" not in f.replace("real/", "", 1)] == ["real/run_manifest.json", "real/summary.json"]
    assert all(f.startswith("real/") for f in files)
    assert any(f.startswith("real/tables/performance.") for f in files) and any(f.startswith("real/figures/quality_f1.") for f in files)
    assert not any(n in files for n in ("real/MOCK_NOT_RESULTS.txt", "real/PARTIAL_RUN.txt"))
    assert not any("MOCK_" in f or "PARTIAL_" in f for f in files)
    s = summary_of(out)
    assert s["status"] == "real_complete" and s["banner"] is None and s["headline"]["n_primary_tests"] == 4
    text = capsys.readouterr().out
    assert "[PASS] matrix completeness" in text and "mechanical count" in text


def test_manifest_hashes_match_files_and_carry_no_timestamps(real_m, tmp_path):
    out = tmp_path / "out"
    go(real_m, out)
    man = json.loads((out / "real" / "run_manifest.json").read_text())
    assert man["status"] == "real_complete" and man["checks"] and man["options"]["alpha"] == 0.05
    assert man["options"]["n_resamples"] == 200 and man["options"]["dpi"] == 50
    for f in man["files"]:
        p = out / "real" / f["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"] and p.stat().st_size == f["bytes"]
    assert {f["path"] for f in man["files"]} | {"run_manifest.json"} == {p.relative_to(out / "real").as_posix()
                                                                          for p in (out / "real").rglob("*") if p.is_file()}
    assert "time" not in json.dumps(man).lower().replace("runtime", "")


def test_reruns_are_byte_identical_and_the_input_is_never_modified(real_copy, tmp_path):
    before = file_hashes(real_copy)
    a, b = tmp_path / "a", tmp_path / "b"
    assert go(real_copy, a) == EXIT_OK and go(real_copy, b) == EXIT_OK
    assert file_hashes(a) == file_hashes(b)
    assert file_hashes(real_copy) == before                                          # read-only on the matrix


def test_default_output_is_inside_the_matrix_and_its_digest_does_not_see_its_own_output(real_copy):
    assert main(["--matrix-dir", str(real_copy), *FAST, "--no-figures"]) == EXIT_OK
    d1 = json.loads((real_copy / "analysis" / "real" / "summary.json").read_text())["provenance"]["input_digest"]
    assert main(["--matrix-dir", str(real_copy), *FAST, "--no-figures"]) == EXIT_OK        # second run, output already there
    d2 = json.loads((real_copy / "analysis" / "real" / "summary.json").read_text())["provenance"]["input_digest"]
    assert d1 == d2


def test_a_rerun_with_other_options_leaves_no_stale_files(real_m, tmp_path):
    out = tmp_path / "out"
    go(real_m, out)
    assert (out / "real" / "figures").is_dir()
    go(real_m, out, "--no-figures", "--formats", "csv")
    files = tree(out)
    assert not any("/figures/" in f for f in files) and {f.rsplit(".", 1)[1] for f in files if "/tables/" in f} == {"csv"}
    assert json.loads((out / "real" / "run_manifest.json").read_text())["options"]["figures"] is False


def test_no_figures_and_format_selection(real_m, tmp_path):
    out = tmp_path / "out"
    assert go(real_m, out, "--no-figures", "--formats", "csv,md") == EXIT_OK
    files = tree(out)
    assert not any("/figures/" in f for f in files) and {f.rsplit(".", 1)[1] for f in files if "/tables/" in f} == {"csv", "md"}
    out2 = tmp_path / "out2"
    assert go(real_m, out2, "--figure-formats", "pdf", "--formats", "tex") == EXIT_OK
    assert {f.rsplit(".", 1)[1] for f in tree(out2) if "/figures/" in f} == {"pdf"}


def test_statistical_options_reach_the_summary(tmp_path, monkeypatch):
    """The CLI's matrices are degenerate (every F1 is 0), so the hand-built non-degenerate analysis is fed in instead."""
    a, m = synthetic_analysis(tmp_path / "syn")
    monkeypatch.setattr(run_mod, "load_analysis_input", lambda *_a, **_k: a)
    x, y = tmp_path / "x", tmp_path / "y"
    assert go(m, x, "--alpha", "0.1", "--seed", "7", "--no-figures") == EXIT_OK
    assert go(m, y, "--alpha", "0.1", "--seed", "8", "--no-figures") == EXIT_OK
    sx, sy = summary_of(x), summary_of(y)
    st = sx["provenance"]["settings"]
    assert (st["alpha"], st["base_seed"], st["n_resamples"], st["n_permutations"]) == (0.1, 7, 200, 500)
    assert all(f["alpha"] == 0.1 for f in sx["families"])
    assert [r["arm"]["ci"] for r in sx["arms"]] != [r["arm"]["ci"] for r in sy["arms"]]            # the bootstrap follows the seed
    assert [c["p_value"] for c in sx["comparisons"]] == [c["p_value"] for c in sy["comparisons"]]  # exact tests do not
    z = tmp_path / "z"
    assert go(m, z, "--alpha", "0.1", "--seed", "7", "--no-figures") == EXIT_OK
    assert (z / "real" / "summary.json").read_bytes() == (x / "real" / "summary.json").read_bytes()   # same seed: identical


# ------------------------------------------------------------------------------------------ metric selection
def test_metric_selection_filters_every_output_without_changing_numbers(real_m, tmp_path):
    full, sel = tmp_path / "full", tmp_path / "sel"
    go(real_m, full)
    assert go(real_m, sel, "--metrics", "f1") == EXIT_OK
    sf, ss = summary_of(full), summary_of(sel)
    assert ss["metrics"]["analysed"] == ["f1"] and {r["arm"]["metric"] for r in ss["arms"]} == {"f1"}
    assert ss["comparisons"] == [c for c in sf["comparisons"] if c["metric"] == "f1"]       # identical numbers, just fewer rows
    assert [r for r in ss["arms"]] == [r for r in sf["arms"] if r["arm"]["metric"] == "f1"]
    names = {Path(f).stem for f in tree(sel) if "/figures/" in f}
    assert names == {"quality_f1", "cost_vs_quality_f1", "cost_vs_budget", "runtime_vs_budget", "tokens_vs_budget",
                     "comparisons_f1", "random_variability_f1", "abstention_rate"}
    perf = (sel / "real" / "tables" / "performance.csv").read_text().splitlines()
    assert perf and all(",f1," in l for l in perf[1:])


def test_selecting_every_metric_equals_the_default(real_m, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    go(real_m, a, "--no-figures")
    go(real_m, b, "--no-figures", "--metrics", "em,judge_accuracy,f1")
    assert (a / "real" / "summary.json").read_bytes() == (b / "real" / "summary.json").read_bytes()


def test_an_unavailable_metric_is_refused_with_the_reason(tmp_path, capsys):
    m = real_matrix(tmp_path / "nj", no_judge=True)
    out = tmp_path / "out"
    assert go(m, out, "--metrics", "judge_accuracy", "--no-figures") == EXIT_STOP
    err = capsys.readouterr().err
    assert "judge_accuracy" in err and "judge was disabled" in err and not out.exists()
    assert go(m, out, "--no-figures") == EXIT_OK
    assert summary_of(out)["skipped_metrics"] == {"judge_accuracy": "the judge was disabled in the matrix"}


def test_select_metrics_recounts_the_headline_from_the_kept_rows(tmp_path):
    a, _ = synthetic_analysis(tmp_path)
    s = build_summary(a, n_resamples=200, n_permutations=500)
    assert select_metrics(s, ["f1", "judge_accuracy", "em"]) is s
    rows = [c.model_copy(update={"significant_holm": True}) if (c.metric == "f1" and c.budget_pct == 25
            and c.analysis == "primary_complete_case") else c for c in s.comparisons]
    s2 = s.model_copy(update={"comparisons": rows})
    h = select_metrics(s2, ["f1"]).headline
    assert (h.n_primary_tests, h.significant_primary_tests) == (2, 1) and "selected metrics: f1" in h.statement
    h = select_metrics(s2, ["em"]).headline
    assert (h.n_primary_tests, h.significant_primary_tests) == (0, 0)                  # em is secondary: no primary test
    with pytest.raises(UsageError, match="not available"):
        select_metrics(s.model_copy(update={"skipped_metrics": {"judge_accuracy": "x"}, "metrics": {**s.metrics, "analysed": ["f1"]}}),
                       ["judge_accuracy"])


def test_select_metrics_never_gives_a_mock_summary_a_count(tmp_path):
    a, _ = synthetic_analysis(tmp_path, status="mock_layout_test")
    s = build_summary(a, n_resamples=200, n_permutations=500)
    h = select_metrics(s, ["f1"]).headline
    assert h.significant_primary_tests is None and h.statement.startswith(MOCK_BANNER)


# ------------------------------------------------------------------------------------------ mock data
def test_mock_data_is_refused_by_default_and_nothing_is_written(mock_m, tmp_path, capsys):
    out = tmp_path / "out"
    assert go(mock_m, out) == EXIT_STOP
    err = capsys.readouterr().err
    assert "analysis refused (mock)" in err and "--allow-mock-layout-test" in err and not out.exists()
    assert not (mock_m / "analysis").exists()


def test_allowed_mock_output_goes_only_to_the_mock_directory_and_is_marked_everywhere(mock_m, tmp_path, capsys):
    out = tmp_path / "out"
    assert go(mock_m, out, "--allow-mock-layout-test") in (EXIT_OK, EXIT_WARN)
    files = tree(out)
    assert files and all(f.startswith("mock_layout_test/") for f in files)
    assert [p.name for p in out.iterdir()] == ["mock_layout_test"]
    base = out / "mock_layout_test"
    assert "MOCK — NOT RESULTS" in (base / "MOCK_NOT_RESULTS.txt").read_text(encoding="utf-8")
    s = summary_of(out, "mock_layout_test")
    assert s["status"] == "mock_layout_test" and s["banner"] == MOCK_BANNER
    assert s["headline"]["significant_primary_tests"] is None
    assert all(c["p_value"] is None and c["significant_holm"] is None and c["suppressed"] for c in s["comparisons"])
    for f in (base / "tables").glob("*.md"):
        assert MOCK_BANNER in f.read_text(encoding="utf-8")
    for f in (base / "tables").glob("*.tex"):
        assert "MOCK --- NOT RESULTS" in f.read_text(encoding="utf-8")
    for f in (base / "tables").glob("*.csv"):
        rows = f.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1 or all(MOCK_BANNER in r for r in rows[1:])
    figs = sorted((base / "figures").iterdir())
    assert figs and all(f.name.startswith("MOCK_") for f in figs)
    assert json.loads((base / "run_manifest.json").read_text())["banner"] == MOCK_BANNER
    assert "MOCK — NOT RESULTS" in capsys.readouterr().out


def test_mock_and_real_outputs_in_one_root_never_mix(mock_m, real_m, tmp_path):
    out = tmp_path / "out"
    go(mock_m, out, "--allow-mock-layout-test")
    before = file_hashes(out / "mock_layout_test")
    assert go(real_m, out) == EXIT_OK
    assert file_hashes(out / "mock_layout_test") == before                         # the real run did not touch the mock folder
    assert sorted(p.name for p in out.iterdir()) == ["mock_layout_test", "real"]
    assert not any("MOCK" in f.read_text(encoding="utf-8", errors="ignore") for f in (out / "real" / "tables").glob("*.csv"))
    assert not any(p.name.startswith("MOCK_") for p in (out / "real").rglob("*"))


def test_a_matrix_mixing_mock_and_real_is_refused_even_with_the_opt_in(real_copy, tmp_path, capsys):
    cid = cond_ids(real_copy)[0]
    edit_json(cond_dir(real_copy, cid) / "condition_result.json", lambda d: d.update(is_mock=True))
    out = tmp_path / "out"
    for extra in ([], ["--allow-mock-layout-test"], ["--allow-mock-layout-test", "--allow-incomplete"]):
        assert go(real_copy, out, *extra) == EXIT_STOP
    assert "never combined" in capsys.readouterr().err and not out.exists()


def test_a_doctored_mock_matrix_is_still_refused_by_default(mock_m, tmp_path):
    m = copy_matrix(mock_m, tmp_path / "doctored")
    for p in (m / "conditions").iterdir():
        def f(d):
            d.update(is_mock=False, reportable=True, not_reportable_reasons=[], data_source="huggingface")
            d["config"].update(data_source="huggingface", extraction_backend="openai", generation_backend="openai",
                               judge_backend="openai")
        edit_json(p / "condition_result.json", f)
    assert go(m, tmp_path / "out") == EXIT_STOP and not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------ partial runs
def test_an_incomplete_matrix_is_refused_without_the_opt_in(real_copy, tmp_path, capsys):
    shutil.rmtree(cond_dir(real_copy, cond_ids(real_copy)[0]))
    out = tmp_path / "out"
    assert go(real_copy, out) == EXIT_STOP
    err = capsys.readouterr().err
    assert "matrix incomplete" in err and "--allow-incomplete" in err and not out.exists()


def test_partial_run_with_opt_in_is_marked_and_kept_apart(real_copy, tmp_path, capsys):
    shutil.rmtree(cond_dir(real_copy, cond_ids(real_copy)[0]))
    out = tmp_path / "out"
    assert go(real_copy, out, "--allow-incomplete") == EXIT_WARN
    assert [p.name for p in out.iterdir()] == ["real_partial"]
    base = out / "real_partial"
    assert "PARTIAL RUN" in (base / "PARTIAL_RUN.txt").read_text()
    s = summary_of(out, "real_partial")
    assert s["status"] == "real_partial" and s["banner"] == PARTIAL_BANNER and s["headline"]["statement"].startswith("PARTIAL RUN")
    assert all(f.name.startswith("PARTIAL_") for f in (base / "figures").iterdir())
    assert PARTIAL_BANNER in (base / "tables" / "performance.md").read_text(encoding="utf-8")
    assert "[WARN] matrix completeness" in capsys.readouterr().out
    assert not (out / "real").exists()


def test_allow_incomplete_does_not_waive_other_problems(real_copy, tmp_path, capsys):
    ids = cond_ids(real_copy)
    shutil.rmtree(cond_dir(real_copy, ids[0]))
    edit_json(cond_dir(real_copy, ids[1]) / "condition_result.json", lambda d: d.update(is_mock=True))
    assert go(real_copy, tmp_path / "out", "--allow-incomplete", "--allow-mock-layout-test") == EXIT_STOP
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------ missing data
def test_missing_predictions_are_reported_not_zeroed(tmp_path, capsys):
    m = real_matrix(tmp_path / "md", strategies=("random", "ketrag"), budgets=(0.25, 0.5, 1.0), random_seeds=[0, 1],
                    comps=priced_components(fail_retrieval_on=Q1_TEXT))
    out = tmp_path / "out"
    assert go(m, out, "--no-figures") == EXIT_WARN                                  # attrition above 2% is a warning
    s = summary_of(out)
    assert all(r["arm"]["n_missing"] == 1 and r["arm"]["n"] == 2 for r in s["arms"])
    assert any("attrition above 2%" in w for w in s["warnings"])
    miss = (out / "real" / "tables" / "missing_data.csv").read_text().splitlines()
    assert len(miss) > 1 and all("no_prediction" in r for r in miss[1:])
    assert "warning: attrition above 2%" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------ validate-only
def test_validate_only_reports_and_writes_nothing(real_copy, tmp_path, capsys):
    before = file_hashes(real_copy)
    out = tmp_path / "out"
    assert go(real_copy, out, "--validate-only") == EXIT_OK
    text = capsys.readouterr().out
    for name in ("data status", "matrix completeness", "question alignment", "random shuffles", "provenance", "git commit"):
        assert name in text
    assert "validate-only: nothing was written" in text
    assert not out.exists() and not (real_copy / "analysis").exists() and file_hashes(real_copy) == before


def test_validate_only_still_refuses_what_a_full_run_refuses(mock_m, real_copy, tmp_path, capsys):
    assert go(mock_m, tmp_path / "o", "--validate-only") == EXIT_STOP
    assert go(mock_m, tmp_path / "o", "--validate-only", "--allow-mock-layout-test") == EXIT_OK
    shutil.rmtree(cond_dir(real_copy, cond_ids(real_copy)[0]))
    assert go(real_copy, tmp_path / "o", "--validate-only") == EXIT_STOP
    assert go(real_copy, tmp_path / "o", "--validate-only", "--allow-incomplete") == EXIT_WARN
    assert not (tmp_path / "o").exists()


def test_validation_failures_stop_before_anything_is_written(real_copy, tmp_path, monkeypatch, capsys):
    def failing(analysis, validation, git_commit):
        validation.add("provenance", "fail", "forced for the test")
    monkeypatch.setattr(run_mod, "validate_input", failing)
    out = tmp_path / "out"
    assert go(real_copy, out) == EXIT_STOP and not out.exists()
    assert "validation failed; nothing was written" in capsys.readouterr().err


def test_validate_input_detects_changed_inputs_and_misaligned_questions(tmp_path):
    a, m = synthetic_analysis(tmp_path)
    v = Validation()
    validate_input(a, v, None)
    assert not v.failed and {c.name for c in v.checks} >= {"provenance", "question alignment", "git commit"}
    bad = a.model_copy(update={"source_files": [a.source_files[0].model_copy(update={"sha256": "0" * 64})] + a.source_files[1:]})
    v = Validation()
    validate_input(bad, v, None)
    assert v.failed and any(c.name == "provenance" and c.status == "fail" for c in v.checks)
    conds = [a.conditions[0].model_copy(update={"question_ids": a.conditions[0].question_ids[:-1]})] + a.conditions[1:]
    v = Validation()
    validate_input(a.model_copy(update={"conditions": conds}), v, None)
    assert any(c.name == "question alignment" and c.status == "fail" for c in v.checks)
    drift = [c.model_copy(update={"shuffle_seed": 5}) if (c.strategy == "random" and c.budget_pct == 25 and c.shuffle_seed == 1) else c
             for c in a.conditions]
    v = Validation()
    validate_input(a.model_copy(update={"conditions": drift}), v, None)
    assert any(c.name == "random shuffles" and c.status == "warn" for c in v.checks)


# ------------------------------------------------------------------------------------------ evidence
def _write_gold(root, analysis, chunks=MOCK_CHUNKS):
    d = Path(root) / analysis.conditions[0].corpus_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks]), encoding="utf-8")
    return Path(root)


def test_evidence_modes_off_auto_require_without_gold(real_m, tmp_path, capsys, monkeypatch):
    import src.prepare_data as prep
    monkeypatch.setattr(prep, "build", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("a rebuild was attempted")))
    root = tmp_path / "no_gold"
    off, auto, req = tmp_path / "off", tmp_path / "auto", tmp_path / "req"
    assert go(real_m, off, "--evidence", "off", "--corpus-root", str(root), "--no-figures") == EXIT_OK
    assert go(real_m, auto, "--corpus-root", str(root), "--no-figures") == EXIT_OK
    assert "evidence metrics were skipped" in capsys.readouterr().out
    for o in (off, auto):
        assert not (o / "real" / "tables" / "evidence.csv").exists() and summary_of(o)["evidence"] == []
    assert go(real_m, req, "--evidence", "require", "--corpus-root", str(root), "--no-figures") == EXIT_STOP
    assert "required but unavailable" in capsys.readouterr().err and not req.exists()


def test_evidence_is_computed_when_gold_is_on_disk(real_m, tmp_path, capsys):
    a = load_analysis_input(real_m)
    root = _write_gold(tmp_path / "corpus", a)
    out = tmp_path / "out"
    assert go(real_m, out, "--corpus-root", str(root)) == EXIT_OK
    s = summary_of(out)
    assert s["evidence"] and {r["arm"]["metric"] for r in s["evidence"]} == set(s["metrics"]["evidence"])
    assert (out / "real" / "tables" / "evidence.csv").exists() and (out / "real" / "figures" / "evidence_hotpotqa.png").exists()
    ev_check = [c for c in json.loads((out / "real" / "run_manifest.json").read_text())["checks"] if c["name"] == "evidence"][0]
    assert ev_check["status"] == "pass" and "reconciled with the runner" in ev_check["detail"]
    assert "chunks.json" in ev_check["detail"]
    assert go(real_m, tmp_path / "out_r", "--evidence", "require", "--corpus-root", str(root), "--no-figures") == EXIT_OK


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_gold_from_the_wrong_corpus_stops_the_run(real_m, tmp_path, capsys, mode):
    a = load_analysis_input(real_m)
    root = _write_gold(tmp_path / "corpus", a, MOCK_CHUNKS[:-1])                    # one chunk short = another corpus
    out = tmp_path / "out"
    assert go(real_m, out, "--evidence", mode, "--corpus-root", str(root), "--no-figures") == EXIT_STOP
    assert "do not belong to this matrix" in capsys.readouterr().err and not out.exists()


def test_a_validate_only_run_with_required_evidence_checks_it_too(real_m, tmp_path, capsys):
    assert go(real_m, tmp_path / "o", "--validate-only", "--evidence", "require", "--corpus-root", str(tmp_path / "x")) == EXIT_STOP
    assert not (tmp_path / "o").exists()


# ------------------------------------------------------------------------------------------ config input
def test_config_input_resolves_the_matrix_directory(tmp_path):
    from experiments.run import main as run_experiments
    cfg = dict(experiment_id="bench", seed=42, dataset="hotpotqa", strategy="random", budget=0.1, data_source="mock",
               num_questions=3, embedding_backend="tfidf", ketrag_mode="tfidf", fast_use_spacy=False,
               extraction_backend="mock", generation_backend="mock", judge_backend="mock",
               cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "out"))
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert run_experiments(["--config", str(path), "--datasets", "hotpotqa", "--strategies", "random", "--budgets", "25,100"]) == 0
    assert main(["--config", str(path), "--out-dir", str(tmp_path / "an")]) == EXIT_STOP                 # mock: refused
    assert not (tmp_path / "an").exists()
    assert main(["--config", str(path), "--out-dir", str(tmp_path / "an"), "--allow-mock-layout-test", "--no-figures", *FAST]) in (0, 1)
    assert [p.name for p in (tmp_path / "an").iterdir()] == ["mock_layout_test"]


# ------------------------------------------------------------------------------------------ the real config template
REAL_CFG = PROJECT / "configs" / "benchmark_real.yaml"


def test_real_config_is_a_valid_unexecuted_template():
    from experiments.conditions import generate_conditions, load_matrix
    from experiments.runner import paid_backends
    head = REAL_CFG.read_text(encoding="utf-8").split("experiment_id:")[0]
    assert "NOT EXECUTED" in head and "NEVER RUN" in head.upper() and "COSTS REAL MONEY" in head and "REPLACE_ME" in head
    cfg, spec = load_matrix(REAL_CFG)
    conds = generate_conditions(spec, cfg.seed)
    assert len(conds) == 72 and cfg.seed == 42 and cfg.num_questions == 200 and cfg.data_source == "huggingface"
    assert spec.random_seeds == [0, 1, 2] and sorted(spec.datasets) == ["hotpotqa", "musique"]
    assert [round(b * 100) for b in spec.budgets] == [5, 10, 25, 50, 75, 100]
    assert sum(1 for c in conds if c.strategy == "random") == 2 * 6 * 3 and sum(1 for c in conds if c.strategy != "random") == 2 * 6 * 3
    assert (cfg.extraction_backend, cfg.generation_backend, cfg.judge_backend) == ("openai",) * 3
    assert len(paid_backends(cfg, False)) == 3                                      # the runner will demand --confirm-paid
    assert cfg.embedding_backend == "sentence-transformers" and cfg.ketrag_mode == "faithful"


def test_real_config_forces_human_decisions_before_a_paid_run():
    from experiments.conditions import load_matrix
    cfg, _ = load_matrix(REAL_CFG)
    assert cfg.judge_model.startswith("REPLACE_ME")                                   # no silent default judge
    assert cfg.extraction_max_cost_usd is not None and cfg.extraction_max_cost_usd <= 1.0   # an unreviewed run stops early
    text = REAL_CFG.read_text(encoding="utf-8")
    assert "--confirm-paid" in text and "--dry-run" in text and "REVIEW" in text


def test_the_real_config_was_never_executed_by_these_tests(tmp_path, capsys):
    """Running the analysis CLI against the template's own output folder finds no matrix: nothing was ever run."""
    assert not (PROJECT / "results" / "experiments" / "benchmark_real").exists()
    assert main(["--config", str(REAL_CFG), "--out-dir", str(tmp_path / "o")]) == EXIT_STOP
    assert not (tmp_path / "o").exists() and not (PROJECT / "results" / "experiments" / "benchmark_real").exists()
    assert not (PROJECT / "data" / "corpus").exists() or not any(
        "huggingface" in p.name for p in (PROJECT / "data" / "corpus").iterdir() if "benchmark_real" in p.name)


# ------------------------------------------------------------------------------------------ design guarantees
def test_run_module_computes_nothing_and_imports_no_calculation_or_api_code():
    src = (PROJECT / "analysis" / "run.py").read_text(encoding="utf-8")
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and n.level == 0:
            mods.add(n.module)
            mods |= {f"{n.module}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
    top = {m.split(".")[0] for m in mods}
    assert top <= {"__future__", "argparse", "hashlib", "json", "sys", "dataclasses", "pathlib", "shutil", "analysis",
                   "aggregation", "experiments", "src"}, top
    forbidden = {"analysis.stats", "analysis.random_avg", "numpy", "scipy", "matplotlib", "openai", "requests", "httpx"}
    assert not (forbidden & {m for m in mods if m in forbidden or m.split(".")[0] in forbidden})
    assert "git" not in {m.split(".")[0] for m in mods} and "subprocess" not in top


def test_the_pipeline_still_cannot_import_analysis():
    import re
    pattern = re.compile(r"^\s*(from|import)\s+analysis(\.|\s|$)", re.M)
    for folder in ("aggregation", "evaluation", "experiments", "extraction", "generation", "graph", "native", "retrieval",
                   "src", "strategies"):
        assert not [p for p in (PROJECT / folder).rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]


def test_output_subdirs_are_distinct_for_every_status():
    assert len({output_subdir(s) for s in ("mock_layout_test", "real_partial", "real_complete")}) == 3


def test_dpi_option_reaches_the_png_files(real_m, tmp_path):
    from PIL import Image
    out = tmp_path / "out"
    assert main(["--matrix-dir", str(real_m), "--out-dir", str(out), "--n-resamples", "200", "--n-permutations", "500",
                 "--dpi", "60", "--figure-formats", "png", "--metrics", "f1"]) == EXIT_OK
    pngs = list((out / "real" / "figures").glob("*.png"))
    assert pngs and all(tuple(round(d) for d in Image.open(p).info["dpi"]) == (60, 60) for p in pngs)
