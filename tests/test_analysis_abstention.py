"""Phase 12, substep 4D: the abstention rate (analysis/summary.py `abstention`, analysis/plots.py `abstention_rate`).
Definition (the runner's own, evaluation/score.py `abstentions.rate_of_scored`): abstained / questions whose generation
did not fail. Checked against hand calculations and against the runner's saved eval_summary.json of a real Phase 10
matrix built with a test answerer that says "not found" for one question. SYNTHETIC data; no API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math

import numpy as np
import pytest

from analysis.loader import load_analysis_input
from analysis.plots import ABSTENTION_REASON, build_figures
from analysis.run import EXIT_OK, main
from analysis.summary import MOCK_BANNER, build_summary, summary_to_json
from tests.helpers_aggregation import PricedExtractor, TimedAnswerer, TimedJudge, mock_corpus
from tests.helpers_analysis import Q1_TEXT, cond_dir, real_matrix
from tests.helpers_experiments import answer_reply, fake_answerer, fake_judge, judge_verdict, spy_components
from tests.helpers_summary import DS, synthetic_analysis

FAST = dict(n_resamples=200, n_permutations=500)


def _abstain(o):
    return o.model_copy(update=dict(outcome="abstained", generation_status="ok", em=0.0, f1=0.0, judge_correct=False,
                                    judge_status="skipped_abstained"))


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    """Hand-built: ketrag@25 abstains on q01,q02 and q12's generation failed; random@25 shuffle 0 abstains on q01,
    shuffle 1 on q01,q02,q03; ketrag@50 has one question without a prediction; ketrag@100 failed everywhere."""
    a, _ = synthetic_analysis(tmp_path_factory.mktemp("abst"), failed=[("ketrag", 25, None, "q12")] +
                              [("ketrag", 100, None, f"q{i:02d}") for i in range(1, 13)])
    obs = []
    for o in a.observations:
        if (o.strategy, o.budget_pct, o.shuffle_seed) == ("ketrag", 25, None) and o.question_id in ("q01", "q02"):
            o = _abstain(o)
        elif (o.strategy, o.budget_pct, o.shuffle_seed) == ("random", 25, 0) and o.question_id == "q01":
            o = _abstain(o)
        elif (o.strategy, o.budget_pct, o.shuffle_seed) == ("random", 25, 1) and o.question_id in ("q01", "q02", "q03"):
            o = _abstain(o)
        elif (o.strategy, o.budget_pct) == ("ketrag", 50) and o.question_id == "q05":
            o = o.model_copy(update=dict(outcome="no_prediction", generation_status=None, em=None, f1=None,
                                         judge_correct=None, judge_status=None))
        obs.append(o)
    a = a.model_copy(update={"observations": obs})
    return a, build_summary(a, **FAST)


def row(s, strat, b, ds=DS):
    (r,) = [r for r in s.abstention if (r.dataset, r.strategy, r.budget_pct) == (ds, strat, b)]
    return r


# ------------------------------------------------------------------------------------------ summary: hand calculation
def test_deterministic_cell_rate_excludes_failed_generations_from_the_denominator(hand):
    r = row(hand[1], "ketrag", 25)
    assert (r.n_runs, r.n_questions, r.n_generated, r.n_abstained, r.n_failed, r.n_no_prediction) == (1, 12, 11, 2, 1, 0)
    assert r.rate == pytest.approx(2 / 11, abs=1e-9) and r.rate_min == r.rate_max == r.rate and r.n_runs_with_rate == 1


def test_random_cell_is_the_mean_of_the_per_shuffle_rates_with_their_range(hand):
    r = row(hand[1], "random", 25)
    assert (r.n_runs, r.n_generated, r.n_abstained, r.n_failed) == (2, 24, 4, 0)             # counts are sums over shuffles
    assert r.rate == pytest.approx((1 / 12 + 3 / 12) / 2, abs=1e-9)                          # mean of 1/12 and 3/12
    assert r.rate_min == pytest.approx(1 / 12, abs=1e-9) and r.rate_max == pytest.approx(3 / 12, abs=1e-9)


def test_a_question_without_a_prediction_is_counted_apart_and_a_real_zero_stays_zero(hand):
    r = row(hand[1], "ketrag", 50)
    assert (r.n_generated, r.n_abstained, r.n_no_prediction, r.n_failed) == (11, 0, 1, 0)
    assert r.rate == 0.0 and r.n_runs_with_rate == 1                                         # zero abstentions among 11 answers


def test_a_cell_with_no_generated_answer_has_no_rate_not_zero(hand):
    r = row(hand[1], "ketrag", 100)
    assert (r.n_generated, r.n_failed, r.n_runs_with_rate) == (0, 12, 0)
    assert r.rate is None and r.rate_min is None and r.rate_max is None


def test_rows_cover_every_cell_in_a_stable_order(hand):
    s = hand[1]
    assert [(r.strategy, r.budget_pct) for r in s.abstention] == [("random", 25), ("random", 50), ("random", 100),
                                                                   ("ketrag", 25), ("ketrag", 50), ("ketrag", 100)]
    assert all(r.n_questions == 12 for r in s.abstention)


def test_abstention_is_counted_not_tested_and_is_in_the_json_deterministically(hand):
    a, s = hand
    doc = json.loads(summary_to_json(s))
    assert [d["rate"] for d in doc["abstention"]] == [r.rate for r in s.abstention]
    assert summary_to_json(build_summary(a, **FAST)) == summary_to_json(s)
    assert set(doc["abstention"][0]) == {"dataset", "strategy", "budget_pct", "n_runs", "n_questions", "n_generated",
                                         "n_abstained", "n_failed", "n_no_prediction", "n_runs_with_rate", "rate",
                                         "rate_min", "rate_max"}                              # no p-value, no interval


def test_abstention_does_not_change_any_existing_summary_number(hand, tmp_path):
    a, s = hand
    base = json.loads(summary_to_json(s))
    base.pop("abstention")
    other, _ = synthetic_analysis(tmp_path)
    assert {"arms", "comparisons", "families", "headline"} <= set(base)
    # the rate is derived from `outcome`; the quality metrics and tests come from the metric values only
    assert [c["p_value"] for c in base["comparisons"] if c["metric"] == "f1"] == [
        c.p_value for c in s.comparisons if c.metric == "f1"]


def test_a_mock_summary_reports_the_rate_but_still_no_inference(tmp_path):
    a, _ = synthetic_analysis(tmp_path, status="mock_layout_test")
    s = build_summary(a, **FAST)
    assert s.abstention and all(c.suppressed for c in s.comparisons) and s.headline.significant_primary_tests is None
    assert s.banner == MOCK_BANNER


# ------------------------------------------------------------------------------------------ reconciliation with the runner
@pytest.fixture(scope="module")
def abstaining_matrix(tmp_path_factory):
    comps = spy_components(
        [], build_extractor=lambda cfg, dry_run=False: PricedExtractor(),
        answerer=lambda cfg: TimedAnswerer(fake_answerer(
            handler=lambda k: answer_reply("not found" if Q1_TEXT in str(k) else "Yes"))),
        judge=lambda cfg: TimedJudge(fake_judge(handler=lambda k: judge_verdict(True))), load_corpus=mock_corpus)
    return real_matrix(tmp_path_factory.mktemp("abst_m"), strategies=("random", "ketrag"), budgets=(0.25, 0.5, 1.0),
                       random_seeds=[0, 1], comps=comps)


def test_rows_equal_the_runners_own_abstention_figures(abstaining_matrix):
    m = abstaining_matrix
    a = load_analysis_input(m)
    s = build_summary(a, **FAST)
    for r in s.abstention:
        runs = [c for c in a.conditions if (c.dataset, c.strategy, c.budget_pct) == (r.dataset, r.strategy, r.budget_pct)]
        assert len(runs) == r.n_runs
        saved = [json.loads((cond_dir(m, c.condition_id) / "eval_summary.json").read_text()) for c in runs]
        rates = [x["abstentions"]["rate_of_scored"] for x in saved]
        assert r.rate == pytest.approx(sum(rates) / len(rates), abs=1e-6)
        assert r.rate_min == pytest.approx(min(rates), abs=1e-6) and r.rate_max == pytest.approx(max(rates), abs=1e-6)
        assert r.n_abstained == sum(x["abstentions"]["n"] for x in saved)
        assert r.n_generated == sum(x["metrics"]["n"] for x in saved)                        # the runner's non-failed count
        assert r.n_failed == sum(x["failures"]["n"] for x in saved)
    assert {round(r.rate, 6) for r in s.abstention} == {round(1 / 3, 6)}                    # exactly one of 3 questions abstains


# ------------------------------------------------------------------------------------------ the figure
def test_figure_plots_the_summary_rows_and_labels_the_definition(hand):
    a, s = hand
    f = build_figures(s).by_name()["abstention_rate"]
    by = {x["strategy"]: x for x in f.series}
    assert set(by) == {"random", "ketrag"} and by["ketrag"]["x"] == [25.0, 50.0, 100.0]
    assert by["ketrag"]["y"][:2] == [pytest.approx(2 / 11, abs=1e-9), 0.0] and by["ketrag"]["y"][2] is None
    assert by["random"]["lo"][0] == pytest.approx(1 / 12, abs=1e-9) and by["random"]["hi"][0] == pytest.approx(3 / 12, abs=1e-9)
    assert by["ketrag"]["lo"] is None                                                       # one run: no band
    assert by["ketrag"]["n_abstained"] == [2, 0, 0] and by["ketrag"]["n_failed"] == [1, 0, 12]
    ax = f.fig.axes[0]
    assert "Abstention rate" in ax.get_ylabel() and "Extraction budget" in ax.get_xlabel() and ax.get_ylim() == (-0.02, 1.02)
    line = [l for l in ax.lines if l.get_color() == "#eb6834"][0]
    y = np.asarray(line.get_ydata(), dtype=float)
    assert math.isnan(y[2]) and y[1] == 0.0                                                 # no answer = a gap; a real 0 is drawn
    txt = " ".join(t.get_text() for t in f.fig.texts)
    assert "not in the denominator" in txt and "no interval" in txt
    assert any("1 cells had no generated answer" in n for n in f.notes)


def test_figure_is_marked_for_mock_data_like_every_other(tmp_path):
    a, _ = synthetic_analysis(tmp_path, status="mock_layout_test")
    f = build_figures(build_summary(a, **FAST)).by_name()["abstention_rate"]
    assert sum(1 for t in f.fig.texts if MOCK_BANNER in t.get_text()) >= 3


def test_a_summary_without_abstention_rows_skips_the_figure_instead_of_computing(hand):
    fs = build_figures(hand[1].model_copy(update={"abstention": []}))
    assert "abstention_rate" not in fs.by_name() and fs.skipped["abstention_rate"] == ABSTENTION_REASON
    assert len(fs.figures) == 15                                    # this summary has no evidence figures either


# ------------------------------------------------------------------------------------------ CLI
def test_cli_writes_the_abstention_rows_and_figure(abstaining_matrix, tmp_path):
    out = tmp_path / "out"
    assert main(["--matrix-dir", str(abstaining_matrix), "--out-dir", str(out), "--n-resamples", "200",
                 "--n-permutations", "500", "--dpi", "50", "--figure-formats", "png", "--metrics", "f1"]) == EXIT_OK
    doc = json.loads((out / "real" / "summary.json").read_text())
    assert doc["abstention"] and all(abs(r["rate"] - 1 / 3) < 1e-6 for r in doc["abstention"])
    assert (out / "real" / "figures" / "abstention_rate.png").is_file()                     # --metrics does not drop it


def test_an_empty_answer_is_not_an_abstention_but_is_a_generated_answer(hand):
    a, _ = hand
    obs = [o.model_copy(update=dict(outcome="empty", em=0.0, f1=0.0, judge_correct=False, judge_status="skipped_empty"))
           if ((o.strategy, o.budget_pct, o.shuffle_seed, o.question_id) == ("ketrag", 25, None, "q03")) else o
           for o in a.observations]
    r = row(build_summary(a.model_copy(update={"observations": obs}), **FAST), "ketrag", 25)
    assert (r.n_generated, r.n_abstained) == (11, 2) and r.rate == pytest.approx(2 / 11, abs=1e-9)


def test_random_rate_is_the_mean_of_run_rates_not_the_pooled_rate(hand):
    a, _ = hand
    obs = [o.model_copy(update=dict(outcome="failed", generation_status="api_error", em=None, f1=None, judge_correct=None,
                                    judge_status="skipped_failed_generation"))
           if ((o.strategy, o.budget_pct, o.shuffle_seed, o.question_id) == ("random", 25, 1, "q12")) else o
           for o in a.observations]
    r = row(build_summary(a.model_copy(update={"observations": obs}), **FAST), "random", 25)
    assert (r.n_generated, r.n_abstained, r.n_failed) == (23, 4, 1)                          # shuffle 1 has 11 generated
    assert r.rate == pytest.approx((1 / 12 + 3 / 11) / 2, abs=1e-9) and r.rate != pytest.approx(4 / 23, abs=1e-6)
