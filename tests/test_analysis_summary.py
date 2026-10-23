"""Phase 12, substep 4A: analysis/summary.py. Every expected number is computed on paper (see
tests/helpers_summary.py for the hand calculations) or by an independent brute-force recomputation. The data is
SYNTHETIC; nothing here is a research result. No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import hashlib
import itertools
import json
import math
import os
import subprocess

import pytest

from analysis import stats as st
from analysis.evidence import EvidenceReport, question_evidence
from analysis.loader import load_analysis_input
from analysis.summary import (ATTRITION_WARNING_FRACTION, MOCK_BANNER, PARTIAL_BANNER, build_summary, read_git_commit,
                              summary_to_json, write_summary)
from tests.helpers_analysis import cond_dir, copy_matrix, file_hashes
from tests.helpers_summary import DS, QIDS, f1_table, synthetic_analysis

FAST = dict(n_resamples=200, n_permutations=500)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    a, m = synthetic_analysis(tmp_path_factory.mktemp("clean"))
    return a, m, build_summary(a, **FAST)


def _arm(s, strat, b, metric):
    (r,) = [r for r in s.arms if (r.arm.strategy, r.arm.budget_pct, r.arm.metric) == (strat, b, metric)]
    return r


def _cmp(s, b, metric, analysis="primary_complete_case"):
    (c,) = [c for c in s.comparisons if (c.budget_pct, c.metric, c.analysis) == (b, metric, analysis)]
    return c


# ------------------------------------------------------------------------------------------ hand calculations
def test_arm_means_and_sample_sizes_count_questions_not_runs(clean):
    s = clean[2]
    k, r = _arm(s, "ketrag", 25, "f1").arm, _arm(s, "random", 25, "f1").arm
    assert (k.n, k.n_universe, k.n_missing, k.n_replicates) == (12, 12, 0, 1)
    assert k.mean == pytest.approx(8 / 12, abs=1e-9)
    assert (r.n, r.n_replicates) == (12, 2)               # 12 questions, NOT 24 shuffle runs
    assert r.mean == pytest.approx(5 / 12, abs=1e-9)      # mean of per-question shuffle means (hand calculation)
    assert _arm(s, "ketrag", 25, "judge_accuracy").arm.mean == pytest.approx(8 / 12, abs=1e-9)


def test_paired_comparison_matches_the_hand_calculation(clean):
    c = _cmp(clean[2], 25, "f1")
    assert (c.strategy_a, c.strategy_b, c.n_pairs, c.n_dropped) == ("random", "ketrag", 12, 0)
    assert c.mean_a == pytest.approx(5 / 12, abs=1e-9) and c.mean_b == pytest.approx(8 / 12, abs=1e-9)
    assert c.mean_diff == pytest.approx(-0.25, abs=1e-9)
    assert (c.n_a_better, c.n_b_better, c.n_tied) == (0, 6, 6)
    assert c.test_method == "exact_binomial"
    assert c.p_value == pytest.approx(2 * 0.5 ** 6, abs=1e-12)


def test_p_value_agrees_with_independent_brute_force_sign_flips(clean):
    d = [-0.5] * 6                                         # the six non-zero differences
    obs = abs(sum(d))
    hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12 for signs in itertools.product((1, -1), repeat=6))
    assert _cmp(clean[2], 25, "f1").p_value == pytest.approx(hits / 2 ** 6, abs=1e-12)


def test_holm_and_bh_inside_the_family_and_significance_flag(clean):
    s = clean[2]
    c25, c50 = _cmp(s, 25, "f1"), _cmp(s, 50, "f1")
    assert c50.test_method == "all_zero" and c50.p_value == 1.0 and "all_differences_zero" in c50.flags
    assert c25.p_holm == pytest.approx(0.0625, abs=1e-12) and c50.p_holm == 1.0       # 2 tests in the family
    assert c25.p_bh == pytest.approx(0.0625, abs=1e-12) and c50.p_bh == 1.0
    assert c25.significant_holm is False                                                # 0.0625 > 0.05


def test_families_are_per_dataset_metric_and_policy_and_exclude_the_anchor(clean):
    s = clean[2]
    assert len(s.families) == 3 * 2                                                      # 3 metrics x 2 policies
    assert all(f.n_tests == 2 and f.n_tests_computed == 2 and f.complete and not f.inference_suppressed for f in s.families)
    assert not [c for c in s.comparisons if c.budget_pct == 100]                         # the anchor is not a test
    roles = {f.metric: f.role for f in s.families}
    assert roles == {"f1": "primary", "judge_accuracy": "primary", "em": "secondary"}
    assert s.headline.n_primary_tests == 4 and s.headline.significant_primary_tests == 0  # 2 metrics x 2 budgets, complete-case
    assert "mechanical count" in s.headline.statement


def test_random_shuffle_variability_is_separate_and_hand_calculated(clean):
    (v,) = [v for v in clean[2].shuffle_variability if v.budget_pct == 25 and v.metric == "f1"]
    assert v.n_shuffles == 2
    assert [p["mean"] for p in v.per_shuffle] == [pytest.approx(4 / 12, abs=1e-9), pytest.approx(6 / 12, abs=1e-9)]
    assert v.mean == pytest.approx(5 / 12, abs=1e-9) and v.min == pytest.approx(4 / 12, abs=1e-9)
    assert v.sd == pytest.approx((2 / 12) / math.sqrt(2), abs=1e-8)
    assert all(v.dataset == DS and v.metric in ("f1", "judge_accuracy", "em") for v in clean[2].shuffle_variability)


def test_cost_means_recomputed_independently_from_the_records(clean):
    a, _m, s = clean
    for row in s.cost:
        vals = [r.measures[row.measure].value for r in a.records
                if r.included and (r.dataset, r.strategy, r.budget_pct) == (row.dataset, row.strategy, row.budget_pct)
                and r.measures[row.measure].status == "measured"]
        assert row.n_known == len(vals)
        assert row.mean == (pytest.approx(sum(vals) / len(vals), abs=1e-8) if vals else None)
    assert {r.measure for r in s.cost} == {"cost_total_usd", "tokens_total_input", "tokens_total_output", "runtime_total_s"}
    rand25 = [r for r in s.cost if r.strategy == "random" and r.budget_pct == 25 and r.measure == "cost_total_usd"][0]
    assert rand25.n_records == 2                                                         # the two shuffle runs


# ------------------------------------------------------------------------------------------ missing data
def test_missing_values_are_reported_never_zero_and_the_policies_differ(tmp_path):
    a, _ = synthetic_analysis(tmp_path, failed=[("ketrag", 25, None, "q12")])
    s = build_summary(a, **FAST)
    k = _arm(s, "ketrag", 25, "f1")
    assert (k.arm.n, k.arm.n_missing, k.arm.missing_reasons) == (11, 1, {"generation_failed": 1})
    assert k.arm.mean == pytest.approx(8 / 11, abs=1e-9)                  # NOT 8/12: the failure is not a zero
    assert k.attrition_fraction == pytest.approx(1 / 12, abs=1e-9) and k.attrition_warning
    cc, aw = _cmp(s, 25, "f1"), _cmp(s, 25, "f1", "sensitivity_missing_as_wrong")
    assert (cc.n_pairs, cc.n_dropped, cc.mean_diff) == (11, 1, pytest.approx(-3 / 11, abs=1e-9))
    assert "1_questions_dropped_for_missing_values" in cc.flags
    assert (aw.n_pairs, aw.n_dropped, aw.mean_diff) == (12, 0, pytest.approx(-0.25, abs=1e-9))   # q12 scored 0 (it was 0)
    assert any("attrition above 2%" in w and "ketrag/25%/f1" in w for w in s.warnings)
    assert ATTRITION_WARNING_FRACTION == 0.02


def test_no_attrition_warning_when_nothing_is_missing(clean):
    s = clean[2]
    assert not any(r.attrition_warning or r.arm.n_missing for r in s.arms)
    assert not [w for w in s.warnings if "attrition" in w]


def test_attrition_threshold_is_strictly_above_two_percent(tmp_path):
    # 1 missing of 12 = 8.3% > 2% warns (above); the boundary itself is exercised through the stored constant
    a, _ = synthetic_analysis(tmp_path, failed=[("ketrag", 25, None, "q12")])
    r = _arm(build_summary(a, **FAST), "ketrag", 25, "f1")
    assert r.attrition_fraction > ATTRITION_WARNING_FRACTION


def test_a_question_missing_in_one_shuffle_is_averaged_over_the_other(tmp_path):
    a, _ = synthetic_analysis(tmp_path, failed=[("random", 25, 1, "q03")])
    s = build_summary(a, **FAST)
    r = _arm(s, "random", 25, "f1").arm
    assert (r.n, r.n_missing, r.n_partial) == (12, 0, 1)                  # q03 rests on shuffle 0 only (value 1.0)
    # per-question means: q03 = 1.0 (shuffle 0 only), q04 = 1.0, q01,q02 = .5, q05..q08 = .5 - 0 (shuffle 1 only: 1)...
    # shuffle0 ones: q01-04; shuffle1 ones: q03-08, q03 dropped from shuffle 1 -> q03 = 1.0 (s0), others unchanged
    assert r.mean == pytest.approx(5 / 12, abs=1e-9)
    c = _cmp(s, 25, "f1")
    assert "some_values_average_fewer_shuffles_than_planned" in c.flags


def test_judge_disabled_skips_judge_accuracy_with_a_reason(tmp_path):
    a, _ = synthetic_analysis(tmp_path, judge_enabled=False)
    s = build_summary(a, **FAST)
    assert s.skipped_metrics == {"judge_accuracy": "the judge was disabled in the matrix"}
    assert "judge_accuracy" not in s.metrics["analysed"]
    assert not [r for r in s.arms if r.arm.metric == "judge_accuracy"]


# ------------------------------------------------------------------------------------------ mock: no findings
def test_mock_summary_has_no_findings_or_significance_claims(tmp_path, monkeypatch):
    a, _ = synthetic_analysis(tmp_path, status="mock_layout_test")

    def boom(*_a, **_k):
        raise AssertionError("a statistical test ran on mock data")
    monkeypatch.setattr(st, "sign_flip_test", boom)
    monkeypatch.setattr(st, "compare_paired", boom)
    monkeypatch.setattr(st, "apply_corrections", boom)
    s = build_summary(a, **FAST)
    assert s.status == "mock_layout_test" and s.banner == MOCK_BANNER == "MOCK — NOT RESULTS"
    assert s.comparisons, "the layout still lists the comparisons"
    for c in s.comparisons:
        assert c.suppressed and "inference_suppressed_mock" in c.flags
        for f in ("ci_low", "ci_high", "dz", "p_value", "p_holm", "p_bh", "significant_holm", "mcnemar_p", "test_method"):
            assert getattr(c, f) is None, f
    assert all(f.inference_suppressed for f in s.families)
    assert s.headline.significant_primary_tests is None and s.headline.n_primary_tests is None
    assert s.headline.statement.startswith(MOCK_BANNER) and "no result" in s.headline.statement


def test_mock_json_has_no_non_null_inferential_field_anywhere(tmp_path):
    a, _ = synthetic_analysis(tmp_path, status="mock_layout_test")
    doc = json.loads(summary_to_json(build_summary(a, **FAST)))
    bad = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("p_value", "p_holm", "p_bh", "significant_holm", "mcnemar_p") and v is not None:
                    bad.append(path + "/" + k)
                walk(v, path + "/" + k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(doc)
    assert bad == []
    assert doc["headline"]["significant_primary_tests"] is None


def test_real_complete_and_partial_banners(tmp_path):
    a, _ = synthetic_analysis(tmp_path)
    assert build_summary(a, **FAST).banner is None
    p = build_summary(a.model_copy(update={"status": "real_partial"}), **FAST)
    assert p.banner == PARTIAL_BANNER and p.headline.statement.startswith("PARTIAL RUN")
    assert p.headline.significant_primary_tests == 0               # real data: inference is computed


def test_real_summary_never_interprets(clean):
    text = summary_to_json(clean[2]).lower()
    for word in ("outperform", "superior", "better than", "we conclude", "proves", "confirms the hypothesis"):
        assert word not in text


# ------------------------------------------------------------------------------------------ 100% anchor
def test_anchor_identical_outcomes_and_identical_selection(clean):
    rows = [x for x in clean[2].anchor_checks]
    assert rows and all(x.budget_pct == 100 and x.strategy_a == "random" and x.strategy_b == "ketrag" for x in rows)
    assert all(x.identical_outcomes is True and x.selected_chunks_identical is True and x.n_pairs == 12 for x in rows)
    assert all("only the selected chunks were verified" in x.note for x in rows)


def test_anchor_mismatch_with_same_selection_is_reported_not_forced(tmp_path):
    a, _ = synthetic_analysis(tmp_path, differ_at_100=True)
    row = [x for x in build_summary(a, **FAST).anchor_checks if x.metric == "f1"][0]
    assert row.identical_outcomes is False and row.selected_chunks_identical is True
    assert "investigate" in row.note


def test_anchor_with_different_selection_does_not_expect_equality(tmp_path):
    a, _ = synthetic_analysis(tmp_path, same_selection_at_100=False, differ_at_100=True)
    row = [x for x in build_summary(a, **FAST).anchor_checks if x.metric == "f1"][0]
    assert row.selected_chunks_identical is False and "do not expect identical" in row.note


def test_anchor_unknown_selection_is_none(tmp_path):
    a, _ = synthetic_analysis(tmp_path)
    conds = [c.model_copy(update={"selected_chunk_ids": None}) if (c.budget_pct == 100 and c.strategy == "ketrag") else c
             for c in a.conditions]
    row = build_summary(a.model_copy(update={"conditions": conds}), **FAST).anchor_checks[0]
    assert row.selected_chunks_identical is None


# ------------------------------------------------------------------------------------------ evidence
def _evidence(rows):
    return EvidenceReport(status="real_complete", gold_origin={DS: "test"}, rows=rows, conditions=[], issues=[],
                          reconciled=True)


def test_evidence_arms_are_summarised_per_question(tmp_path):
    a, _ = synthetic_analysis(tmp_path)
    rows = []
    for (strat, b, shuf), _vals in f1_table().items():
        for q in QIDS:
            # gold = {c1, c2}; selected = {c1..c4} for every question except q12 (selects only c1, c3 -> recall .5)
            sel = ["c1", "c3"] if q == "q12" else ["c1", "c2", "c3", "c4"]
            e = question_evidence(["c1", "c2"], sel, ["c1"])
            rows.append(e.model_copy(update=dict(dataset=DS, strategy=strat, budget_pct=b, shuffle_seed=shuf,
                                                 question_id=q)))
    s = build_summary(a, _evidence(rows), **FAST)
    e = [r for r in s.evidence if r.arm.metric == "selection_recall" and r.arm.strategy == "ketrag"
         and r.arm.budget_pct == 25][0]
    assert e.role == "evidence" and e.arm.n == 12
    assert e.arm.mean == pytest.approx((11 * 1.0 + 0.5) / 12, abs=1e-9)
    ur = [r for r in s.evidence if r.arm.metric == "retrieval_recall" and r.arm.strategy == "random"
          and r.arm.budget_pct == 25][0]
    assert ur.arm.mean == pytest.approx(0.5, abs=1e-9) and ur.arm.n == 12 and ur.arm.n_replicates == 2
    assert s.metrics["evidence"] and build_summary(a, **FAST).evidence == []


# ------------------------------------------------------------------------------------------ provenance
def test_input_digest_and_source_hashes_match_the_files_on_disk(clean):
    a, m, s = clean
    files = s.provenance.source_files
    assert files and {f["path"] for f in files} >= {"matrix.json"}
    for f in files:
        assert hashlib.sha256((m / f["path"]).read_bytes()).hexdigest() == f["sha256"]
    h = hashlib.sha256()
    for f in sorted(files, key=lambda f: f["path"]):
        h.update(f"{f['path']}\0{f['sha256']}\n".encode())
    assert s.provenance.input_digest == h.hexdigest()


def test_digest_changes_when_an_input_file_changes(tmp_path):
    a, m = synthetic_analysis(tmp_path)
    d0 = build_summary(a, **FAST).provenance.input_digest
    cp = copy_matrix(m, tmp_path / "copy")
    p = cond_dir(cp, a.conditions[0].condition_id) / "scored.jsonl"
    p.write_text(p.read_text() + "\n")
    changed = load_analysis_input(cp)
    assert build_summary(a.model_copy(update={"source_files": changed.source_files}), **FAST).provenance.input_digest != d0


def test_provenance_fields_settings_and_no_absolute_paths(clean, tmp_path):
    a, m, s = clean
    p = s.provenance
    assert p.matrix_dir_name == m.name and p.matrix_ids == a.matrix_ids
    assert p.question_seed == {DS: a.question_seed[DS]}
    assert p.settings["alpha"] == 0.05 and p.settings["n_resamples"] == 200 and p.settings["anchor_budget_pct"] == 100
    assert {"python", "numpy", "scipy"} <= set(p.versions)
    assert str(m.parent) not in summary_to_json(s)                       # no machine-specific path in the output
    assert build_summary(a, git_commit="abc123", **FAST).provenance.git_commit == "abc123"
    assert p.git_commit is None


def test_git_commit_is_read_without_git_and_without_writing(tmp_path):
    repo = tmp_path / "r"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / "sub" / "deep").mkdir(parents=True)
    sha = "0123456789abcdef0123456789abcdef01234567"
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / ".git" / "refs" / "heads" / "main").write_text(sha + "\n")
    before = file_hashes(repo)
    assert read_git_commit(repo / "sub" / "deep") == sha
    assert file_hashes(repo) == before                                   # strictly read-only
    (repo / ".git" / "refs" / "heads" / "main").unlink()                 # now only packed-refs knows the branch
    (repo / ".git" / "packed-refs").write_text(f"# pack-refs\n{sha} refs/heads/main\n")
    assert read_git_commit(repo) == sha
    (repo / ".git" / "HEAD").write_text(sha + "\n")                      # detached HEAD
    assert read_git_commit(repo) == sha
    assert read_git_commit(tmp_path / "nowhere") is None
    (tmp_path / "plain").mkdir()
    assert read_git_commit(tmp_path / "plain") in (None, read_git_commit(tmp_path))   # no repo inside tmp_path


# ------------------------------------------------------------------------------------------ determinism
def test_same_input_gives_byte_identical_json(tmp_path, clean):
    a = clean[0]
    j1, j2 = summary_to_json(build_summary(a, **FAST)), summary_to_json(build_summary(a, **FAST))
    assert j1 == j2 and j1.endswith("\n") and "\r" not in j1
    w1, w2 = write_summary(build_summary(a, **FAST), tmp_path / "x" / "a.json"), write_summary(build_summary(a, **FAST), tmp_path / "b.json")
    assert w1.read_bytes() == w2.read_bytes()
    json.loads(w1.read_text(encoding="utf-8"))                           # valid JSON, no NaN/Infinity (allow_nan=False)
    assert "generated_at" not in j1 and "timestamp" not in j1


def test_base_seed_is_recorded_and_changes_resampling_only(clean):
    a = clean[0]
    s0, s1 = build_summary(a, base_seed=0, **FAST), build_summary(a, base_seed=7, **FAST)
    assert s1.provenance.settings["base_seed"] == 7
    assert [r.arm.mean for r in s0.arms] == [r.arm.mean for r in s1.arms]          # data-determined
    assert [c.p_value for c in s0.comparisons] == [c.p_value for c in s1.comparisons]   # exact tests: seed-free
    assert [r.arm.ci for r in s0.arms] != [r.arm.ci for r in s1.arms]               # bootstrap intervals move


def test_json_is_identical_across_python_hash_seeds(tmp_path, clean):
    script = tmp_path / "run.py"
    script.write_text(
        "import sys, hashlib\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from pathlib import Path\n"
        "from tests.helpers_summary import synthetic_analysis\n"
        "from analysis.summary import build_summary, summary_to_json\n"
        "a, _ = synthetic_analysis(Path(sys.argv[1]), matrix_dir=sys.argv[2])\n"
        "s = build_summary(a, n_resamples=200, n_permutations=500)\n"
        "print(hashlib.sha256(summary_to_json(s).encode()).hexdigest())\n")
    out = []
    for i, seed in enumerate(("1", "99")):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
        out.append(subprocess.run([sys.executable, "-I", str(script), str(tmp_path / f"w{i}"), str(clean[1])], env=env, check=True,
                                  capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1])).stdout.strip())
    assert out[0] == out[1] and len(out[0]) == 64


# ------------------------------------------------------------------------------------------ design / safety
def test_summary_imports_only_analysis_and_numeric_libraries():
    src = Path(__file__).resolve().parents[1] / "analysis" / "summary.py"
    mods = set()
    for n in ast.walk(ast.parse(src.read_text())):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            mods.add(n.module.split(".")[0])
    assert mods <= {"__future__", "hashlib", "itertools", "json", "platform", "pathlib", "typing", "numpy", "scipy",
                    "pydantic", "analysis"}, mods


def test_summary_does_not_modify_the_matrix(tmp_path):
    a, m = synthetic_analysis(tmp_path)
    before = file_hashes(m)
    build_summary(a, **FAST)
    assert file_hashes(m) == before


def test_a_genuinely_loaded_mock_matrix_flows_through_with_no_findings(tmp_path):
    from tests.helpers_analysis import mock_matrix
    m = mock_matrix(tmp_path / "mock", random_seeds=[0, 1])
    a = load_analysis_input(m, allow_mock_layout_test=True)
    s = build_summary(a, **FAST)
    assert s.status == "mock_layout_test" and s.data_kind == "mock" and s.banner == MOCK_BANNER
    assert any("MOCK DATA" in c and "not research results" in c for c in s.caveats)     # the loader's own caveat
    assert s.comparisons and all(c.suppressed and c.p_value is None and c.significant_holm is None for c in s.comparisons)
    assert s.headline.significant_primary_tests is None


def test_cost_mean_uses_measured_values_only_and_counts_the_rest(tmp_path):
    a, _ = synthetic_analysis(tmp_path)
    recs = []
    done = False
    for r in a.records:
        if r.included and (r.strategy, r.budget_pct) == ("random", 25) and not done:
            m = dict(r.measures)
            m["cost_total_usd"] = m["cost_total_usd"].model_copy(update={"status": "lower_bound", "value": 999.0,
                                                                         "lower_bound": 999.0})
            r = r.model_copy(update={"measures": m})
            done = True
        recs.append(r)
    row = [c for c in build_summary(a.model_copy(update={"records": recs}), **FAST).cost
           if (c.strategy, c.budget_pct, c.measure) == ("random", 25, "cost_total_usd")][0]
    other = [r.measures["cost_total_usd"].value for r in a.records if r.included and (r.strategy, r.budget_pct) == ("random", 25)][1]
    assert (row.n_records, row.n_known, row.n_lower_bound, row.n_unknown) == (2, 1, 1, 0)
    assert row.mean == pytest.approx(other, abs=1e-9)                     # the 999 lower bound never enters the mean


def test_a_missing_condition_skips_its_test_and_marks_the_family_incomplete(tmp_path):
    a, _ = synthetic_analysis(tmp_path)
    obs = [o for o in a.observations if not (o.strategy == "ketrag" and o.budget_pct == 50)]
    s = build_summary(a.model_copy(update={"observations": obs}), **FAST)
    skipped = [c for c in s.comparisons if c.skipped_reason]
    assert skipped and all(c.budget_pct == 50 and c.p_value is None and "skipped" in c.flags for c in skipped)
    fam = [f for f in s.families if f.metric == "f1" and f.analysis == "primary_complete_case"][0]
    assert (fam.n_tests, fam.n_tests_computed, fam.n_tests_skipped, fam.complete) == (2, 1, 1, False)
    assert any("tests skipped" in w and "f1" in w for w in s.warnings)
    c25 = _cmp(s, 25, "f1")
    assert c25.p_holm == pytest.approx(c25.p_value, abs=1e-12)           # Holm over the single computable test
