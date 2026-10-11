"""Phase 11: group statistics. Hand-built records, so every expected number is computed by hand."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

import pytest

from aggregation.schemas import ConditionRecord, Measure
from aggregation.summarize import DEFAULT_LEVELS, parse_group_by, stat_of, summarize_groups


def M(value=None, status="measured", lb=None, kind="cost"):
    return Measure(kind=kind, status=status, value=value, lower_bound=lb)


def rec(dataset="hotpotqa", strategy="random", pct=25, seed=42, *, cost=1.0, status="completed", em=0.5, mock=False,
        reconciled=True, measures=None, counts=None):
    ms = {"cost_total_usd": cost if isinstance(cost, Measure) else M(cost),
          "tokens_total_input": M(100, kind="tokens"), "cost_total_per_question_usd": M(0.1, kind="normalized")}
    ms.update(measures or {})
    return ConditionRecord(
        condition_id=f"{dataset}__{strategy}__b{pct:03d}__seed{seed}", matrix_id="m", dataset=dataset, data_source="huggingface",
        strategy=strategy, budget=pct / 100, budget_pct=pct, seed=seed, status=status, clean=status == "completed",
        included=status in ("completed", "completed_with_failures"), is_mock=mock, reportable=not mock,
        not_reportable_reasons=["mock extraction backend"] if mock else [], reconciled=reconciled,
        counts={"n_selected": 4, "n_questions": 3, "n_predictions": 3, "extraction_n_failed": 0, **(counts or {})},
        metrics={"em": em, "f1": em, "em_failures_as_wrong": em, "f1_failures_as_wrong": em, "judge_accuracy": None},
        measures=ms)


def group(groups, level, **key):
    return next(g for g in groups if g.level == level and g.key == key)


# ============================================================================ the Stat of one measure
def test_basic_statistics_with_sample_counts():
    s = stat_of([M(1.0), M(2.0), M(3.0)], additive=True)
    assert (s.n_records, s.n_known, s.n_lower_bound, s.n_unknown, s.n_not_applicable) == (3, 3, 0, 0, 0)
    assert s.complete and s.sum == 6.0 and s.sum_lower_bound == 6.0 and s.mean == 2.0
    assert (s.min, s.max, s.std) == (1.0, 3.0, 1.0)


def test_a_group_of_true_zeros_has_a_sum_of_zero_not_a_missing_sum():
    s = stat_of([M(0.0), M(0.0)], additive=True)
    assert s.complete and s.sum == 0.0 and s.mean == 0.0 and s.n_known == 2
    assert s.sum is not None


def test_a_group_with_no_measurements_has_no_numbers_at_all():
    s = stat_of([M(status="unknown"), M(status="not_run")], additive=True)
    assert (s.n_known, s.n_unknown) == (0, 2) and not s.complete
    assert s.sum is None and s.sum_lower_bound is None and s.mean is None and s.min is None and s.max is None and s.std is None


def test_unknown_records_are_counted_not_averaged_and_block_the_sum():
    s = stat_of([M(4.0), M(status="unknown")], additive=True)
    assert (s.n_known, s.n_unknown, s.complete) == (1, 1, False)
    assert s.mean == 4.0 and s.sum is None and s.sum_lower_bound == 4.0           # the mean cannot hide the gap: n_known says 1 of 2


def test_lower_bounds_never_enter_the_mean_but_raise_the_lower_bound_of_the_sum():
    s = stat_of([M(1.0), M(2.0), M(status="lower_bound", lb=5.0)], additive=True)
    assert (s.n_known, s.n_lower_bound, s.complete) == (2, 1, False)
    assert s.mean == 1.5 and (s.min, s.max) == (1.0, 2.0)
    assert s.sum is None and s.sum_lower_bound == 8.0


def test_disabled_records_are_not_applicable_not_zero():
    s = stat_of([M(status="disabled"), M(status="disabled")], additive=True)
    assert s.n_not_applicable == 2 and s.n_known == 0 and not s.complete and s.sum is None and s.mean is None
    mixed = stat_of([M(3.0), M(status="disabled")], additive=True)               # applicable records all measured
    assert mixed.n_known == 1 and mixed.n_not_applicable == 1 and mixed.complete and mixed.sum == 3.0


def test_standard_deviation_needs_two_values_and_is_never_zero_by_default():
    assert stat_of([M(5.0)], additive=True).std is None
    assert stat_of([M(2.0), M(4.0)], additive=True).std == pytest.approx(1.4142135623730951, abs=1e-8)
    assert stat_of([M(2.0), M(2.0)], additive=True).std == 0.0                    # two equal values: a real zero spread


def test_ratios_have_no_sum_and_integers_stay_integers():
    r = stat_of([M(0.1, kind="normalized"), M(0.3, kind="normalized")], additive=False)
    assert r.sum is None and r.sum_lower_bound is None and r.mean == pytest.approx(0.2)
    t = stat_of([M(100, kind="tokens"), M(250, kind="tokens")], additive=True)
    assert t.sum == 350 and isinstance(t.sum, int) and t.min == 100 and t.max == 250


# ============================================================================ groups
def test_default_levels_are_the_cell_and_the_two_descriptive_rollups():
    assert [(n, f) for n, f in DEFAULT_LEVELS] == [
        ("cell", ("dataset", "strategy", "budget_pct")), ("dataset_strategy", ("dataset", "strategy")),
        ("dataset_budget", ("dataset", "budget_pct"))]


def test_cells_collapse_seeds_and_report_the_spread():
    groups = summarize_groups([rec(seed=41, cost=1.0, em=0.4), rec(seed=42, cost=3.0, em=0.6)])
    g = group(groups, "cell", dataset="hotpotqa", strategy="random", budget_pct=25)
    assert g.comparison == "matched_budget_cell" and g.seeds == [41, 42]
    assert (g.n_conditions, g.n_included, g.n_excluded) == (2, 2, 0)
    c = g.measures["cost_total_usd"]
    assert (c.n_known, c.sum, c.mean, c.min, c.max) == (2, 4.0, 2.0, 1.0, 3.0) and c.std == pytest.approx(1.414213562)
    assert g.metrics["em"].mean == pytest.approx(0.5) and g.metrics["em"].n_known == 2


def test_excluded_conditions_stay_visible_but_never_enter_the_statistics():
    groups = summarize_groups([rec(seed=41, cost=2.0), rec(seed=42, status="aborted", cost=M(status="lower_bound", lb=0.5))])
    g = group(groups, "cell", dataset="hotpotqa", strategy="random", budget_pct=25)
    assert (g.n_conditions, g.n_included, g.n_excluded) == (2, 1, 1)
    assert g.excluded_reasons == {"aborted": 1} and g.statuses == {"aborted": 1, "completed": 1}
    c = g.measures["cost_total_usd"]
    assert (c.n_records, c.n_known, c.sum, c.mean) == (1, 1, 2.0, 2.0)            # the aborted 0.5 is not averaged in
    assert not g.reportable and any("1 condition(s) in this group are excluded" in r for r in g.not_reportable_reasons)
    assert g.condition_ids == sorted(g.condition_ids) and len(g.condition_ids) == 2


def test_a_group_with_only_excluded_conditions_says_so_instead_of_inventing_numbers():
    g = summarize_groups([rec(status="failed")])[0]
    assert g.n_included == 0 and g.n_excluded == 1 and g.measures == {} and not g.reportable
    assert "no included conditions" in g.not_reportable_reasons
    assert g.metrics["em"].n_known == 0 and g.metrics["em"].mean is None            # no included record: nothing to average


def test_failed_but_completed_conditions_are_included_with_their_failures_counted():
    groups = summarize_groups([rec(seed=41, status="completed_with_failures", counts={"extraction_n_failed": 2}),
                               rec(seed=42, counts={"extraction_n_failed": 0})])
    g = group(groups, "cell", dataset="hotpotqa", strategy="random", budget_pct=25)
    assert g.n_included == 2 and g.n_conditions_with_failures == 1 and g.counts["extraction_n_failed"] == 2


def test_counts_are_summed_over_included_conditions_only():
    groups = summarize_groups([rec(seed=41), rec(seed=42, status="aborted")])
    g = group(groups, "cell", dataset="hotpotqa", strategy="random", budget_pct=25)
    assert g.counts["n_selected"] == 4 and g.counts["n_questions"] == 3 and g.counts["n_predictions"] == 3


# ============================================================================ matched cells vs descriptive rollups
def test_rollups_are_labelled_descriptive_and_carry_no_quality_metrics():
    records = [rec(strategy="random", pct=25, cost=1.0, em=0.2), rec(strategy="random", pct=50, cost=2.0, em=0.9)]
    groups = summarize_groups(records)
    roll = group(groups, "dataset_strategy", dataset="hotpotqa", strategy="random")
    assert roll.comparison == "descriptive_rollup" and "budgets" in roll.comparison_note
    assert roll.metrics is None and "not a matched comparison" in roll.metrics_omitted_reason
    c = roll.measures["cost_total_usd"]
    assert c.sum == 3.0 and c.n_known == 2                                         # costs add up across budgets: meaningful
    for cell in [g for g in groups if g.level == "cell"]:                          # the cells keep their own metrics
        assert cell.metrics is not None and cell.metrics_omitted_reason is None


def test_the_budget_rollup_collapses_strategies_and_says_so():
    groups = summarize_groups([rec(strategy="random", pct=25), rec(strategy="ketrag", pct=25)])
    roll = group(groups, "dataset_budget", dataset="hotpotqa", budget_pct=25)
    assert roll.comparison == "descriptive_rollup" and "strategies" in roll.comparison_note and roll.metrics is None
    assert roll.n_included == 2 and roll.measures["cost_total_usd"].sum == 2.0


def test_strategies_are_never_pooled_inside_a_cell():
    groups = summarize_groups([rec(strategy="random", pct=25, em=0.1), rec(strategy="ketrag", pct=25, em=0.9)])
    cells = [g for g in groups if g.level == "cell"]
    assert len(cells) == 2 and {g.key["strategy"] for g in cells} == {"random", "ketrag"}
    assert {g.key["strategy"]: g.metrics["em"].mean for g in cells} == {"random": 0.1, "ketrag": 0.9}


def test_custom_group_by_keeps_metrics_only_when_the_matched_keys_are_all_present():
    name, fields = parse_group_by("dataset,strategy,budget_pct,seed")
    assert name == "dataset+strategy+budget_pct+seed" and fields == ("dataset", "strategy", "budget_pct", "seed")
    per_seed = [g for g in summarize_groups([rec(seed=1), rec(seed=2)], levels=[(name, fields)])]
    assert len(per_seed) == 2 and all(g.comparison == "matched_budget_cell" and g.metrics is not None for g in per_seed)
    only_strategy = summarize_groups([rec(), rec(strategy="ketrag")], levels=[parse_group_by("strategy")])
    assert all(g.comparison == "descriptive_rollup" and g.metrics is None for g in only_strategy)
    assert "datasets" in only_strategy[0].comparison_note
    for bad in ("", "colour", "dataset,dataset"):
        with pytest.raises(ValueError):
            parse_group_by(bad)


def test_metrics_with_missing_values_show_n_missing():
    groups = summarize_groups([rec(seed=1, em=0.5), rec(seed=2, em=None)])
    g = group(groups, "cell", dataset="hotpotqa", strategy="random", budget_pct=25)
    assert (g.metrics["em"].n_records, g.metrics["em"].n_known, g.metrics["em"].n_missing) == (2, 1, 1)
    assert g.metrics["em"].mean == 0.5 and g.metrics["judge_accuracy"].n_known == 0 and g.metrics["judge_accuracy"].mean is None


# ============================================================================ reportability of groups
def test_mock_groups_are_never_reportable_and_say_why():
    groups = summarize_groups([rec(seed=1, mock=True), rec(seed=2)])
    g = group(groups, "cell", dataset="hotpotqa", strategy="random", budget_pct=25)
    assert g.n_mock == 1 and g.n_reportable == 1 and not g.reportable and "mock extraction backend" in g.not_reportable_reasons


def test_a_clean_real_group_is_reportable():
    g = group(summarize_groups([rec(seed=1), rec(seed=2)]), "cell", dataset="hotpotqa", strategy="random", budget_pct=25)
    assert g.reportable and g.not_reportable_reasons == [] and g.n_reportable == 2


def test_unreconciled_conditions_taint_their_groups():
    g = group(summarize_groups([rec(reconciled=False)]), "cell", dataset="hotpotqa", strategy="random", budget_pct=25)
    assert g.n_unreconciled == 1 and not g.reportable and any("cross-checks" in r for r in g.not_reportable_reasons)


# ============================================================================ determinism
def test_the_order_of_the_input_does_not_change_the_result():
    records = [rec(strategy=s, pct=p, seed=sd, cost=p / 10 + sd / 100) for s in ("random", "ketrag") for p in (5, 25) for sd in (41, 42)]
    base = [g.model_dump() for g in summarize_groups(records)]
    for seed in range(3):
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        assert [g.model_dump() for g in summarize_groups(shuffled)] == base


def test_float_sums_are_exact_to_nine_decimals():
    s = stat_of([M(0.1), M(0.2), M(0.3)], additive=True)
    assert s.sum == 0.6 and s.mean == 0.2
