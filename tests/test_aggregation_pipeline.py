"""Phase 11: end-to-end aggregation of real Phase 10 output: the 48-condition mock matrix, priced matrices,
failed / aborted / missing / unreadable conditions, duplicates, mixed configs, output files, read-only
behaviour, determinism and the command line. No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import hashlib
import json
import re
import shutil

import pytest
import yaml

from aggregation.aggregate import aggregate_matrices, assert_reportable, main, write_outputs
from aggregation.loader import AggregationError
from aggregation.schemas import AggregateReport, ConditionRecord
from evaluation.score import NotReportableError
from experiments.run import main as run_main
from tests.helpers_aggregation import (AbortingPricedExtractor, PartlyUnknownExtractor, build_matrix, cond_dir, cond_ids,
                                       copy_matrix, edit_json, priced_components)

CID = "hotpotqa__random__b050__seed42"
LATE = "2099-01-01T00:00:00+00:00"


@pytest.fixture(scope="module")
def mock48(tmp_path_factory):
    return build_matrix(tmp_path_factory.mktemp("mock48"), mock=True, datasets=("hotpotqa", "musique"),
                        strategies=("random", "ketrag", "lazygraphrag", "fastgraphrag"),
                        budgets=(0.05, 0.10, 0.25, 0.50, 0.75, 1.0))


@pytest.fixture(scope="module")
def priced(tmp_path_factory):
    return build_matrix(tmp_path_factory.mktemp("priced"))


@pytest.fixture()
def mutable(priced, tmp_path):
    return copy_matrix(priced, tmp_path / "m")


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "aggregate" not in p.relative_to(root).parts}


# ======================================================================== the real 48-condition mock output
def test_the_full_48_condition_mock_matrix_aggregates_and_is_never_reportable(mock48):
    report, records = aggregate_matrices([mock48])
    assert (report.n_conditions_loaded, report.n_included, report.n_excluded, report.n_planned) == (48, 48, 0, 48)
    assert report.matrix_complete is True and report.statuses == {"completed": 48}
    levels = {lv: sum(1 for g in report.groups if g.level == lv) for lv in ("cell", "dataset_strategy", "dataset_budget")}
    assert levels == {"cell": 48, "dataset_strategy": 8, "dataset_budget": 12}
    assert report.is_mock and report.n_mock == 48 and report.n_reportable_conditions == 0 and not report.reportable
    assert "mock dataset fixture (data_source: mock)" in report.not_reportable_reasons
    assert all(r.is_mock and not r.reportable for r in records) and not any(g.reportable for g in report.groups)
    assert (report.n_errors, report.n_warnings, report.issues) == (0, 0, [])          # real Phase 10 output reconciles fully
    assert len({r.condition_id for r in records}) == 48 and all(r.reconciled for r in records)
    with pytest.raises(NotReportableError, match="mock"):
        assert_reportable(report)


def test_mock_costs_are_measured_zeros_and_every_measure_has_a_status(mock48):
    _, records = aggregate_matrices([mock48])
    for r in records:
        for k in ("cost_extraction_usd", "cost_generation_usd", "cost_judge_usd", "cost_total_usd"):
            assert r.measures[k].status == "measured" and r.measures[k].value == 0.0
        assert all(m.status in ("measured", "lower_bound", "unknown", "not_run", "disabled", "not_applicable")
                   for m in r.measures.values())
        assert all((m.value is not None) == (m.status == "measured") for m in r.measures.values())   # value only if measured
    cell = next(g for g in aggregate_matrices([mock48])[0].groups if g.level == "cell")
    assert cell.measures["cost_total_usd"].complete and cell.measures["cost_total_usd"].sum == 0.0


# =========================================================== random_seeds (option B) + evidence (gap 1/C) + CI (F)
@pytest.fixture(scope="module")
def random_repeats(tmp_path_factory):
    """Random repeated on 3 shuffle seeds, same questions; ketrag runs once (not repeated)."""
    return build_matrix(tmp_path_factory.mktemp("rrepeat"), mock=True, datasets=("hotpotqa",),
                        strategies=("random", "ketrag"), budgets=(0.5,), random_seeds=[0, 1, 2])


def test_random_seeds_give_three_conditions_random_and_one_ketrag(random_repeats):
    report, records = aggregate_matrices([random_repeats])
    assert report.n_conditions_loaded == 4                                       # 3 Random + 1 KET-RAG
    random_recs = [r for r in records if r.strategy == "random"]
    ketrag_recs = [r for r in records if r.strategy == "ketrag"]
    assert len(random_recs) == 3 and {r.shuffle_seed for r in random_recs} == {0, 1, 2}
    assert len(ketrag_recs) == 1 and ketrag_recs[0].shuffle_seed is None
    assert all(r.seed == 42 for r in records)                                     # same questions/corpus throughout


def test_cell_grouping_collapses_randoms_three_shuffles_into_one_cell(random_repeats):
    report, _ = aggregate_matrices([random_repeats])
    cells = [g for g in report.groups if g.level == "cell"]
    random_cell = next(g for g in cells if g.key["strategy"] == "random")
    ketrag_cell = next(g for g in cells if g.key["strategy"] == "ketrag")
    assert random_cell.n_included == 3 and sorted(random_cell.shuffle_seeds) == [0, 1, 2]
    assert ketrag_cell.n_included == 1 and ketrag_cell.shuffle_seeds == []


def test_duplicate_detection_treats_different_shuffle_seeds_as_distinct_not_duplicates(random_repeats):
    # Three Random conditions share (dataset, strategy, budget_pct, seed) but differ by shuffle_seed:
    # they must NOT be flagged as duplicates of one another.
    report, _ = aggregate_matrices([random_repeats])
    assert report.n_errors == 0 and not report.duplicates


def test_evidence_recall_fields_reach_conditions_jsonl(random_repeats, tmp_path):
    report, records = aggregate_matrices([random_repeats])
    out = write_outputs(tmp_path / "agg", report, records)
    rows = [json.loads(l) for l in (out / "conditions.jsonl").read_text().splitlines()]
    for row in rows:
        assert "selection_recall" in row["evidence"] and "expected_random_selection_recall" in row["evidence"]
        if row["strategy"] == "random":
            # Random's measured recall should be close to its exact expectation (small mock corpus; loose bound).
            assert abs(row["evidence"]["selection_recall"] - row["evidence"]["expected_random_selection_recall"]) <= 1.0


def test_ci95_fields_reach_conditions_jsonl_and_csv(random_repeats, tmp_path):
    report, records = aggregate_matrices([random_repeats])
    out = write_outputs(tmp_path / "agg", report, records)
    rows = [json.loads(l) for l in (out / "conditions.jsonl").read_text().splitlines()]
    assert all("em" in row["ci95"] for row in rows)
    with open(out / "conditions.csv") as f:
        header = next(csv.reader(f))
    assert "em_ci95_low" in header and "em_ci95_high" in header
    assert "evidence_selection_recall" in header


def test_group_statistics_average_evidence_recall_across_randoms_shuffles(random_repeats):
    report, _ = aggregate_matrices([random_repeats])
    random_cell = next(g for g in report.groups if g.level == "cell" and g.key["strategy"] == "random")
    assert random_cell.metrics is not None and "selection_recall" in random_cell.metrics
    assert random_cell.metrics["selection_recall"].n_known == 3                        # averaged over the 3 shuffles
    assert random_cell.metrics["expected_random_selection_recall"].mean == pytest.approx(0.5)  # budget = 0.5


def test_every_cell_of_the_matrix_appears_exactly_once(mock48):
    report, _ = aggregate_matrices([mock48])
    cells = [(g.key["dataset"], g.key["strategy"], g.key["budget_pct"]) for g in report.groups if g.level == "cell"]
    assert len(cells) == len(set(cells)) == 48 and {c[2] for c in cells} == {5, 10, 25, 50, 75, 100}


# ======================================================================== priced matrix: exact group figures
def test_a_real_complete_matrix_is_reportable_and_the_group_sums_are_exact(priced):
    report, recs = aggregate_matrices([priced])
    assert report.reportable and report.matrix_complete and not report.is_mock and report.n_reportable_conditions == 4
    assert_reportable(report)                                                           # does not raise
    g = lambda lv, **k: next(x for x in report.groups if x.level == lv and x.key == k)
    assert g("dataset_strategy", dataset="hotpotqa", strategy="random").measures["cost_extraction_usd"].sum == pytest.approx(0.02 + 0.04)
    assert g("dataset_budget", dataset="hotpotqa", budget_pct=50).measures["cost_extraction_usd"].sum == pytest.approx(0.08)
    cell = g("cell", dataset="hotpotqa", strategy="random", budget_pct=50)
    assert cell.measures["cost_total_usd"].n_known == 1 and cell.measures["tokens_total_input"].sum == 4 * 100 + 600
    assert cell.reportable and cell.counts["n_selected"] == 4 and cell.metrics["em"].n_known == 1
    # this-run money is smaller than the comparable uncached cost because chunks were shared through the cache
    spent = sum(r.measures["spent_extraction_usd"].value for r in recs)
    uncached = sum(r.measures["cost_extraction_usd"].value for r in recs)
    assert spent < uncached and uncached == pytest.approx(0.02 + 0.04 + 0.02 + 0.04)


def test_rollup_groups_carry_no_quality_metrics_but_cells_do(priced):
    report, _ = aggregate_matrices([priced])
    for gr in report.groups:
        assert (gr.metrics is not None) == (gr.level == "cell")
        assert (gr.comparison == "matched_budget_cell") == (gr.level == "cell")


# ======================================================================== failed / aborted / missing stay visible
def test_a_failed_condition_is_listed_with_its_reason_and_the_aggregate_is_not_reportable(tmp_path):
    m = build_matrix(tmp_path, strategies=("random", "ketrag"), budgets=(0.5,), ketrag_mode="faithful")
    report, records = aggregate_matrices([m])
    by_id = {r.condition_id: r for r in records}
    assert by_id[CID].included and not by_id["hotpotqa__ketrag__b050__seed42"].included
    assert [(e.kind, e.condition_id) for e in report.excluded] == [("failed", "hotpotqa__ketrag__b050__seed42")]
    assert "invalid configuration" in report.excluded[0].reason
    assert (report.n_included, report.n_excluded, report.matrix_complete) == (1, 1, False)
    assert any("matrix incomplete: 1 of 2" in r for r in report.not_reportable_reasons) and not report.reportable
    cell = next(g for g in report.groups if g.level == "cell" and g.key["strategy"] == "ketrag")
    assert (cell.n_conditions, cell.n_included, cell.n_excluded, cell.measures) == (1, 0, 1, {})
    ok = next(g for g in report.groups if g.level == "cell" and g.key["strategy"] == "random")
    assert ok.n_included == 1 and ok.measures["cost_total_usd"].complete
    with pytest.raises(NotReportableError, match="matrix incomplete"):
        assert_reportable(report)


def test_an_aborted_condition_and_the_conditions_that_never_ran_are_both_visible(tmp_path):
    m = build_matrix(tmp_path, strategies=("random",), budgets=(0.25, 0.5),
                     comps=priced_components(extractor=AbortingPricedExtractor))
    report, records = aggregate_matrices([m])
    assert sorted(e.kind for e in report.excluded) == ["aborted", "missing_result"]
    missing = next(e for e in report.excluded if e.kind == "missing_result")
    assert missing.condition_id == "hotpotqa__random__b050__seed42" and "never run" in missing.reason
    assert (report.n_planned, report.n_included, report.matrix_complete) == (2, 0, False)
    assert "no included conditions" in report.not_reportable_reasons
    assert [r.condition_id for r in records] == ["hotpotqa__random__b025__seed42"] and not records[0].included


def test_an_unreadable_result_is_excluded_once_with_an_error(mutable):
    (cond_dir(mutable, CID) / "condition_result.json").write_text("{ not json")
    report, _ = aggregate_matrices([mutable])
    assert [e.kind for e in report.excluded] == ["unreadable_result"]                   # not ALSO "missing_result"
    assert any(i.code == "unreadable_result" and i.severity == "error" for i in report.issues)
    assert report.n_included == 3 and report.matrix_complete is False and report.n_errors == 1
    with pytest.raises(NotReportableError):
        assert_reportable(report)


def test_a_condition_folder_without_a_result_is_a_warning_and_a_missing_condition(mutable):
    (cond_dir(mutable, CID) / "condition_result.json").unlink()
    report, _ = aggregate_matrices([mutable])
    assert any(i.code == "no_result_file" and i.severity == "warning" for i in report.issues)
    assert [e.kind for e in report.excluded] == ["missing_result"] and report.n_included == 3


def test_a_missing_manifest_means_completeness_cannot_be_claimed(mutable):
    (mutable / "matrix.json").unlink()
    report, _ = aggregate_matrices([mutable])
    assert report.n_planned is None and report.matrix_complete is None and not report.reportable
    assert any("no_matrix_manifest" == i.code for i in report.issues)
    assert any("planned conditions are unknown" in r for r in report.not_reportable_reasons)


def test_nothing_to_aggregate_is_an_error(tmp_path):
    with pytest.raises(AggregationError, match="not a matrix directory"):
        aggregate_matrices([tmp_path])
    (tmp_path / "conditions").mkdir()
    with pytest.raises(AggregationError, match="no condition results"):
        aggregate_matrices([tmp_path])


# ======================================================================== duplicates
def _duplicate(matrix, cid=CID, finished=None):
    dup = matrix / "conditions" / (cid + "_copy")
    shutil.copytree(cond_dir(matrix, cid), dup)
    if finished:
        edit_json(dup / "condition_result.json", lambda d: d.update(finished_at_utc=finished))
    return dup


def test_duplicate_conditions_are_an_error_by_default_and_name_both_copies(mutable):
    _duplicate(mutable)
    with pytest.raises(AggregationError) as e:
        aggregate_matrices([mutable])
    msg = str(e.value)
    assert "refusing to choose silently" in msg and f"{CID}_copy" in msg and f"m:{CID}" in msg.replace(str(mutable), "m")
    assert "--on-duplicate latest" in msg


def test_latest_resolves_a_duplicate_visibly_and_makes_the_aggregate_not_reportable(mutable):
    _duplicate(mutable, finished=LATE)
    report, records = aggregate_matrices([mutable], on_duplicate="latest")
    assert len(records) == 4 and report.n_included == 4
    (d,) = report.duplicates
    assert d.cell == {"dataset": "hotpotqa", "strategy": "random", "budget_pct": 50, "seed": 42, "shuffle_seed": None}
    assert d.kept.endswith(f"{CID}_copy") and d.dropped[0].endswith(CID) and d.policy == "latest"
    assert [e.kind for e in report.excluded] == ["duplicate_dropped"]
    assert any("--on-duplicate latest" in r for r in report.not_reportable_reasons) and not report.reportable
    assert any(i.code == "directory_name_mismatch" for i in report.issues)


def test_two_duplicates_finished_at_the_same_time_cannot_be_ordered(mutable):
    same = json.loads((cond_dir(mutable, CID) / "condition_result.json").read_text())["finished_at_utc"]
    _duplicate(mutable, finished=same)
    with pytest.raises(AggregationError, match="finished at the same time"):
        aggregate_matrices([mutable], on_duplicate="latest")


def test_the_same_cell_in_two_matrix_directories_is_a_duplicate(priced, tmp_path):
    a, b = copy_matrix(priced, tmp_path / "a"), copy_matrix(priced, tmp_path / "b")
    with pytest.raises(AggregationError, match="4 cell\\(s\\) have more than one result"):
        aggregate_matrices([a, b])
    for cid in cond_ids(b):
        edit_json(cond_dir(b, cid) / "condition_result.json", lambda d: d.update(finished_at_utc=LATE))
    report, records = aggregate_matrices([a, b], on_duplicate="latest")
    assert len(records) == 4 and len(report.duplicates) == 4 and all(d.kept.startswith(str(b)) for d in report.duplicates)


def test_unknown_duplicate_policy_is_rejected(mutable):
    with pytest.raises(AggregationError, match="unknown duplicate policy"):
        aggregate_matrices([mutable], on_duplicate="newest")


# ======================================================================== mixed configurations
def test_conditions_with_different_configs_are_an_error_naming_the_difference(mutable):
    edit_json(cond_dir(mutable, CID) / "condition_result.json", lambda d: d["config"].update(chunk_size_words=300))
    with pytest.raises(AggregationError) as e:
        aggregate_matrices([mutable])
    assert "different configurations" in str(e.value) and "chunk_size_words" in str(e.value) and "--allow-mixed-config" in str(e.value)


def test_a_different_model_is_a_different_configuration(mutable):
    edit_json(cond_dir(mutable, "hotpotqa__ketrag__b025__seed42") / "condition_result.json",
              lambda d: d["config"].update(generation_model="gpt-4o"))
    with pytest.raises(AggregationError, match="generation_model"):
        aggregate_matrices([mutable])


def test_allowing_mixed_configs_is_explicit_visible_and_not_reportable(mutable):
    edit_json(cond_dir(mutable, CID) / "condition_result.json", lambda d: d["config"].update(chunk_size_words=300))
    report, records = aggregate_matrices([mutable], allow_mixed_config=True)
    assert report.mixed_config and "chunk_size_words" in report.config_differences and len(report.shared_config_fingerprints) == 2
    assert not report.reportable and any("different configurations were combined" in r for r in report.not_reportable_reasons)
    assert len({r.shared_config_fingerprint for r in records}) == 2
    cell = next(g for g in report.groups if g.level == "cell" and g.key["budget_pct"] == 50 and g.key["strategy"] == "random")
    assert len(cell.config_fingerprints) == 1                                           # per group: which config stands behind it


def test_judge_on_in_some_conditions_and_off_in_others_is_a_mixed_run(mutable):
    edit_json(cond_dir(mutable, CID) / "condition_result.json", lambda d: d["metrics"].update(judge_enabled=False))
    with pytest.raises(AggregationError, match="judge_enabled"):
        aggregate_matrices([mutable])


def test_coordinates_and_paths_may_differ_between_conditions(mutable):
    edit_json(cond_dir(mutable, CID) / "condition_result.json", lambda d: d["config"].update(cache_dir="/elsewhere", output_dir="/x"))
    report, _ = aggregate_matrices([mutable])
    assert not report.mixed_config and len(report.shared_config_fingerprints) == 1


# ======================================================================== outputs
def test_outputs_are_written_and_validate_against_the_schemas(priced, tmp_path):
    report, records = aggregate_matrices([priced])
    out = write_outputs(tmp_path / "agg", report, records)
    assert sorted(p.name for p in out.iterdir()) == ["aggregate.json", "conditions.csv", "conditions.jsonl"]
    back = AggregateReport.model_validate_json((out / "aggregate.json").read_text())
    assert back.model_dump(mode="json") == report.model_dump(mode="json")
    rows = [ConditionRecord.model_validate_json(l) for l in (out / "conditions.jsonl").read_text().splitlines()]
    assert [r.condition_id for r in rows] == [r.condition_id for r in records] and rows[0].config
    assert not list(out.glob("*.tmp"))


def test_csv_leaves_missing_values_empty_and_always_states_the_status(tmp_path):
    m = build_matrix(tmp_path, strategies=("random",), budgets=(0.5,), comps=priced_components(extractor=PartlyUnknownExtractor))
    report, records = aggregate_matrices([m])
    out = write_outputs(tmp_path / "agg", report, records)
    (row,) = list(csv.DictReader(open(out / "conditions.csv", encoding="utf-8")))
    assert row["cost_extraction_usd"] == "" and row["cost_extraction_usd__status"] == "lower_bound"       # not "0"
    assert row["tokens_extraction_input"] == "" and row["tokens_extraction_input__status"] == "lower_bound"
    assert row["cost_generation_usd"] != "" and row["cost_generation_usd__status"] == "measured"
    assert row["cost_total_usd"] == "" and row["is_mock"] == "false" and row["reportable"] == "true"
    assert row["budget_pct"] == "50" and row["seed"] == "42" and row["strategy"] == "random" and row["metric_em"] != ""
    assert len(csv.reader(open(out / "conditions.csv", encoding="utf-8")).__next__()) == len(row)


def test_csv_distinguishes_a_measured_zero_from_a_missing_value(mock48, tmp_path):
    report, records = aggregate_matrices([mock48])
    out = write_outputs(tmp_path / "agg", report, records)
    row = next(csv.DictReader(open(out / "conditions.csv", encoding="utf-8")))
    assert row["cost_total_usd"] == "0.0" and row["cost_total_usd__status"] == "measured"      # zero is written as 0.0
    assert row["is_mock"] == "true" and row["reportable"] == "false"


def test_aggregation_never_modifies_the_results_it_reads(priced, tmp_path):
    m = copy_matrix(priced, tmp_path / "m")
    before = snapshot(m)
    assert main(["--matrix-dir", str(m)]) == 0
    assert snapshot(m) == before and (m / "aggregate" / "aggregate.json").exists()


def test_aggregation_is_deterministic(priced, tmp_path):
    report1, rec1 = aggregate_matrices([priced])
    report2, rec2 = aggregate_matrices([priced])
    write_outputs(tmp_path / "a", report1, rec1)
    write_outputs(tmp_path / "b", report2, rec2)
    for name in ("conditions.jsonl", "conditions.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    strip = lambda d: {k: v for k, v in d.items() if k != "generated_at_utc"}
    assert strip(json.loads((tmp_path / "a" / "aggregate.json").read_text())) == strip(json.loads((tmp_path / "b" / "aggregate.json").read_text()))


# ======================================================================== the reportability guard
def test_assert_reportable_accepts_a_dict_and_refuses_unresolved_errors(priced):
    report, _ = aggregate_matrices([priced])
    d = report.model_dump(mode="json")
    assert_reportable(d)
    d["n_errors"] = 2
    with pytest.raises(NotReportableError, match="2 unresolved error"):
        assert_reportable(d)
    assert_reportable(d, require_reconciled=False)                                     # explicit opt-out
    d["reportable"], d["not_reportable_reasons"] = False, ["because"]
    with pytest.raises(NotReportableError, match="because"):
        assert_reportable(d, require_reconciled=False)


def test_unreconciled_data_makes_the_aggregate_not_reportable(mutable):
    edit_json(cond_dir(mutable, CID) / "condition_result.json",
              lambda d: d["cost"]["generation"].update(cost_if_uncached_usd=0.5))
    report, records = aggregate_matrices([mutable])
    assert not report.reportable and report.n_errors >= 1 and any("cross-checks failed" in r for r in report.not_reportable_reasons)
    with pytest.raises(NotReportableError):
        assert_reportable(report)


# ======================================================================== command line
def test_cli_reports_a_clean_real_matrix_with_exit_0(priced, tmp_path, capsys):
    m = copy_matrix(priced, tmp_path / "m")
    assert main(["--matrix-dir", str(m)]) == 0
    out = capsys.readouterr()
    assert "aggregated 4 condition(s): included 4, excluded 0" in out.out and "MOCK" not in out.out
    assert "cost_total_usd per included condition: 4 measured, 0 not measured" in out.out and out.err == ""


def test_cli_marks_mock_output_and_still_exits_0_when_everything_was_aggregated(mock48, tmp_path, capsys):
    m = copy_matrix(mock48, tmp_path / "m")
    assert main(["--matrix-dir", str(m)]) == 0
    assert "MOCK RESULTS - testing only, NOT reportable." in capsys.readouterr().out
    assert (m / "aggregate" / "conditions.csv").exists()


def test_cli_exit_1_when_conditions_are_excluded_or_missing(tmp_path, capsys):
    m = build_matrix(tmp_path, strategies=("random", "ketrag"), budgets=(0.5,), ketrag_mode="faithful")
    assert main(["--matrix-dir", str(m)]) == 1
    err = capsys.readouterr().err
    assert "EXCLUDED hotpotqa__ketrag__b050__seed42: failed" in err and "NOT REPORTABLE" in err


def test_cli_exit_2_and_no_output_for_duplicates_and_mixed_configs(mutable, capsys):
    _duplicate(mutable)
    assert main(["--matrix-dir", str(mutable)]) == 2
    assert "refusing to choose silently" in capsys.readouterr().err and not (mutable / "aggregate").exists()
    shutil.rmtree(mutable / "conditions" / f"{CID}_copy")
    edit_json(cond_dir(mutable, CID) / "condition_result.json", lambda d: d["config"].update(chunk_size_words=300))
    assert main(["--matrix-dir", str(mutable)]) == 2 and not (mutable / "aggregate").exists()
    assert main(["--matrix-dir", str(mutable), "--allow-mixed-config"]) == 0              # opted in: aggregated, flagged below
    assert "NOT REPORTABLE" in capsys.readouterr().err
    assert (mutable / "aggregate" / "aggregate.json").exists()
    assert json.loads((mutable / "aggregate" / "aggregate.json").read_text())["mixed_config"] is True


def test_cli_opt_in_flags_work_end_to_end(mutable, tmp_path, capsys):
    _duplicate(mutable, finished=LATE)
    assert main(["--matrix-dir", str(mutable), "--on-duplicate", "latest", "--out-dir", str(tmp_path / "o")]) == 1   # warning issue
    rep = json.loads((tmp_path / "o" / "aggregate.json").read_text())
    assert len(rep["duplicates"]) == 1 and rep["reportable"] is False


def test_cli_bad_arguments_stop_with_exit_2(priced, tmp_path, capsys):
    assert main([]) == 2 and "give --matrix-dir" in capsys.readouterr().err
    assert main(["--matrix-dir", str(tmp_path / "nope")]) == 2
    assert main(["--matrix-dir", str(priced), "--group-by", "colour"]) == 2
    a, b = copy_matrix(priced, tmp_path / "a"), copy_matrix(priced, tmp_path / "b")
    assert main(["--matrix-dir", str(a), "--matrix-dir", str(b)]) == 2
    assert "--out-dir is required" in capsys.readouterr().err


def test_cli_group_by_adds_a_custom_level(priced, tmp_path, capsys):
    m = copy_matrix(priced, tmp_path / "m")
    assert main(["--matrix-dir", str(m), "--group-by", "dataset,strategy,budget_pct,seed", "--group-by", "strategy"]) == 0
    rep = json.loads((m / "aggregate" / "aggregate.json").read_text())
    levels = {g["level"] for g in rep["groups"]}
    assert {"cell", "dataset_strategy", "dataset_budget", "dataset+strategy+budget_pct+seed", "strategy"} <= levels
    per_seed = [g for g in rep["groups"] if g["level"] == "dataset+strategy+budget_pct+seed"]
    assert len(per_seed) == 4 and all(g["metrics"] is not None for g in per_seed)
    assert all(g["metrics"] is None for g in rep["groups"] if g["level"] == "strategy")


def test_cli_finds_the_matrix_through_the_config_exactly_like_the_runner_does(tmp_path, capsys):
    cfg = dict(experiment_id="bench", seed=42, dataset="hotpotqa", strategy="random", budget=0.1, data_source="mock",
               num_questions=3, embedding_backend="tfidf", ketrag_mode="tfidf", fast_use_spacy=False,
               extraction_backend="mock", generation_backend="mock", judge_backend="mock",
               cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "out"),
               matrix={"datasets": ["hotpotqa"], "strategies": ["random"], "budgets": [0.25, 1.0]})
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert run_main(["--config", str(path)]) == 0                                       # the real Phase 10 CLI
    capsys.readouterr()
    assert main(["--config", str(path)]) == 0
    out = tmp_path / "out" / "experiments" / "bench" / "aggregate"
    assert (out / "aggregate.json").exists() and "MOCK RESULTS" in capsys.readouterr().out
    rep = json.loads((out / "aggregate.json").read_text())
    assert rep["n_planned"] == 2 and rep["matrix_complete"] is True and rep["is_mock"] is True


# ======================================================================== scope
def test_no_earlier_phase_depends_on_aggregation():
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"^\s*(from|import)\s+aggregation\b", re.M)
    offenders = [str(p.relative_to(root)) for pkg in ("src", "extraction", "graph", "retrieval", "generation", "strategies",
                                                      "native", "evaluation", "experiments")
                 for p in (root / pkg).rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_aggregation_imports_no_network_or_llm_client():
    root = Path(__file__).resolve().parents[1] / "aggregation"
    pattern = re.compile(r"^\s*(from|import)\s+(openai|requests|httpx|urllib|socket|aiohttp)\b", re.M)
    assert [p.name for p in root.glob("*.py") if pattern.search(p.read_text(encoding="utf-8"))] == []
