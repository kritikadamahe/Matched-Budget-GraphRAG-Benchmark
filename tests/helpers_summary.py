"""Hand-built analysis inputs for the summary / table tests (Phase 12, substep 4A).

`synthetic_analysis` loads a small matrix through the REAL loader (so provenance, records and the report are
genuine), then swaps in hand-written per-question observations whose every statistic can be computed on paper.
The values are SYNTHETIC and say nothing about the research question. No API call is made.

Universe: 12 questions q01..q12, dataset hotpotqa, strategies random (shuffles 0 and 1) and ketrag, budgets 25/50/100.
  ketrag  @25 : f1 = 1 on q01..q08, 0 on q09..q12                                   mean 8/12
  random  @25 : shuffle 0 = 1 on q01..q04 (4/12); shuffle 1 = 1 on q03..q08 (6/12)   per-question mean: q01,q02 .5;
                q03,q04 1; q05..q08 .5; rest 0  -> arm mean 5/12; shuffle SD = (2/12)/sqrt(2)
  d = random - ketrag @25 = -0.5 on q01,q02,q05..q08 (six questions), 0 elsewhere: mean -0.25, 6 B-better, 6 ties,
  all non-zero |d| equal -> exact binomial two-sided p = 2 * 0.5**6 = 0.03125
  @50 and @100: both strategies identical (f1 = 1 on q01..q06); em = f1; judge_correct = f1 >= 0.5.
"""
from __future__ import annotations

from pathlib import Path

from analysis.loader import load_analysis_input
from analysis.schemas import ConditionInput, QuestionObservation
from tests.helpers_analysis import real_matrix

QIDS = [f"q{i:02d}" for i in range(1, 13)]
DS = "hotpotqa"


def _ones(first: int, last: int) -> set[str]:
    return {f"q{i:02d}" for i in range(first, last + 1)}


def f1_table() -> dict[tuple, dict[str, float]]:
    """(strategy, budget_pct, shuffle_seed) -> {question: f1}"""
    t = {}
    t[("ketrag", 25, None)] = {q: float(q in _ones(1, 8)) for q in QIDS}
    t[("random", 25, 0)] = {q: float(q in _ones(1, 4)) for q in QIDS}
    t[("random", 25, 1)] = {q: float(q in _ones(3, 8)) for q in QIDS}
    for b in (50, 100):
        for key in (("ketrag", b, None), ("random", b, 0), ("random", b, 1)):
            t[key] = {q: float(q in _ones(1, 6)) for q in QIDS}
    return t


def _obs(cid, strat, b, shuf, q, f1, *, failed=False) -> QuestionObservation:
    if failed:
        return QuestionObservation(condition_id=cid, dataset=DS, strategy=strat, budget_pct=b, seed=42, shuffle_seed=shuf,
                                   question_id=q, outcome="failed", generation_status="api_error", em=None, f1=None,
                                   judge_correct=None, judge_status="skipped_failed_generation")
    return QuestionObservation(condition_id=cid, dataset=DS, strategy=strat, budget_pct=b, seed=42, shuffle_seed=shuf,
                               question_id=q, outcome="answered", generation_status="ok", em=f1, f1=f1,
                               judge_correct=f1 >= 0.5, judge_status="ok")


def synthetic_analysis(root, *, status=None, failed=(), same_selection_at_100=True, differ_at_100=False,
                       judge_enabled=None, table=None, matrix_dir=None):
    """The loaded analysis of a tiny matrix with its observations replaced by the hand-written ones above.
    `failed`: (strategy, budget_pct, shuffle_seed, question_id) tuples whose generation failed (value missing).
    `matrix_dir` reuses an already built matrix (so two processes see the same input files).
    `status="mock_layout_test"` relabels the loaded input as mock (the data is the same; only the label differs,
    which is all the summary looks at). `differ_at_100`: ketrag's 100% outcomes differ from random's."""
    root = Path(root)
    m = Path(matrix_dir) if matrix_dir else real_matrix(root / "m", strategies=("random", "ketrag"),
                                                         budgets=(0.25, 0.5, 1.0), random_seeds=[0, 1])
    base = load_analysis_input(m)
    table = table or f1_table()
    if differ_at_100:
        table = {**table, ("ketrag", 100, None): {q: float(q in _ones(1, 7)) for q in QIDS}}
    ids = {(c.strategy, c.budget_pct, c.shuffle_seed): c.condition_id for c in base.conditions}
    obs, conds = [], []
    for (strat, b, shuf), vals in table.items():
        cid = ids[(strat, b, shuf)]
        for q in QIDS:
            obs.append(_obs(cid, strat, b, shuf, q, vals[q], failed=(strat, b, shuf, q) in set(failed)))
    for c in base.conditions:
        sel = [f"c{i}" for i in range(1, 5)]
        if c.budget_pct == 100 and not same_selection_at_100 and c.strategy == "ketrag":
            sel = ["c1", "c2"]
        conds.append(c.model_copy(update={"question_ids": list(QIDS), "n_questions": len(QIDS), "selected_chunk_ids": sel}))
    upd = {"observations": obs, "conditions": conds, "question_ids": {DS: list(QIDS)}}
    if status:
        upd["status"] = status
        upd["data_kind"] = "mock" if status == "mock_layout_test" else "real"
    if judge_enabled is not None:
        upd["judge_enabled"] = judge_enabled
    return base.model_copy(update=upd), m
