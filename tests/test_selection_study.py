"""Selection-only study: recall arithmetic, Random averaging, and a free end-to-end run on the mock corpus."""
import json

import numpy as np

from analysis.selection_study import main, per_question


def test_per_question_recall_and_all_gold():
    gold = {"q1": {"a", "b"}, "q2": {"c"}}
    pq = per_question(gold, {"a", "c"}, ["q1", "q2"])
    assert pq["selection_recall"].tolist() == [0.5, 1.0]
    assert pq["all_gold"].tolist() == [0.0, 1.0]


def test_end_to_end_on_mock_data(tmp_path):
    out = tmp_path / "study"
    assert main(["--config", "configs/benchmark_mock.yaml", "--datasets", "hotpotqa", "--num-questions", "3",
                 "--budgets", "0.25,0.75", "--random-seeds", "0,1", "--n-resamples", "200",
                 "--n-permutations", "200", "--out-dir", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["meta"]["llm_calls"] == 0
    res = summary["results"][0]
    perf = {(r["metric"], r["strategy"], r["budget_pct"]): r for r in res["performance"]}
    for s in ("random", "ketrag", "lazygraphrag", "fastgraphrag"):
        assert 0.0 <= perf[("selection_recall", s, 25)]["mean"] <= perf[("selection_recall", s, 75)]["mean"] <= 1.0
    # 3 smart strategies x 2 budgets x 2 metrics, each with a Holm-adjusted p-value
    assert len(res["comparisons"]) == 12 and all(r["p_holm"] is not None for r in res["comparisons"])
    for name in ("report.md", "performance.csv", "comparisons.csv", "selection_recall.png"):
        assert (out / name).exists()
