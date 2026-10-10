"""
Records the experiment runner writes (Phase 10). One ConditionResult per condition holds the RAW
per-condition measurements the later cost/runtime aggregation (Phase 11) and analysis (Phase 12)
need: counts, per-stage status / time / usage / cost, headline metrics, and mock / reportability
flags. It stores no gold answers and no retrieved text.

Unknown stays unknown: a cost or token total that includes an unknown usage is None (null in the
JSON), never 0, exactly as in the extraction, generation and evaluation modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

RUNNER_SCHEMA_VERSION = "1"

STAGES = ("corpus", "ranking", "selection", "extraction", "graph", "retrieval", "generation", "evaluation")

# ok       the stage ran to the end (individual items may still have failed: see its details)
# failed   the stage hit an unexpected error      aborted  a run-level stop (bad key, quota, spending cap)
# skipped  nothing to do                          not_run  an earlier stage stopped the condition
StageStatus = Literal["ok", "failed", "aborted", "skipped", "not_run"]

# completed                 every stage ran, nothing failed anywhere
# completed_with_failures   every stage ran, but some items failed (chunks, questions, answers, judge calls)
# aborted                   stopped by a run-level error; resume re-runs it
# failed                    a stage hit an unexpected error; resume re-runs it
ConditionStatus = Literal["completed", "completed_with_failures", "aborted", "failed"]

RESULT_FILE = "condition_result.json"


class StageRecord(BaseModel):
    status: StageStatus = "not_run"
    elapsed_seconds: float = 0.0
    error: str | None = None
    details: dict = Field(default_factory=dict)


class QuestionFailure(BaseModel):
    """A question that produced no prediction (so it is absent from predictions.jsonl)."""
    question_id: str
    stage: str
    error: str


class ConditionResult(BaseModel):
    schema_version: str = RUNNER_SCHEMA_VERSION

    # --- identity: what was run ---
    condition_id: str
    matrix_id: str
    run_id: str | None = None                    # extraction.run.make_run_id(cfg)
    dataset: str
    data_source: str
    strategy: str
    budget: float
    budget_pct: int
    seed: int
    corpus_id: str | None = None
    config_hash: str | None = None
    config: dict = Field(default_factory=dict)   # the full ExperimentConfig of this condition

    # --- outcome ---
    status: ConditionStatus = "failed"
    clean: bool = False                          # status == "completed"
    error: str | None = None
    problems: list[str] = Field(default_factory=list)    # failures that make the status "completed_with_failures"
    warnings: list[str] = Field(default_factory=list)    # data-quality notes (e.g. unknown token usage): no status change
    is_mock: bool = False
    reportable: bool = False
    not_reportable_reasons: list[str] = Field(default_factory=list)

    # --- budget enforcement ---
    n_chunks_total: int | None = None
    n_selected_expected: int | None = None       # src.budget.selected_count(N, budget)
    n_selected: int | None = None                # what the pipeline actually selected / extracted
    selected_fingerprint: str | None = None

    # --- questions ---
    n_questions: int | None = None
    n_predictions: int = 0
    question_failures: list[QuestionFailure] = Field(default_factory=list)

    # --- measurements ---
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    cost: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)

    started_at_utc: str = ""
    finished_at_utc: str = ""
    wall_seconds: float = 0.0


def write_condition_result(condition_dir: str | Path, result: ConditionResult) -> Path:
    """Atomic write (temp file, then rename). The file's existence is what 'this condition has a
    result' means, so it is always written last."""
    condition_dir = Path(condition_dir)
    condition_dir.mkdir(parents=True, exist_ok=True)
    path = condition_dir / RESULT_FILE
    tmp = condition_dir / f"{RESULT_FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_condition_result(condition_dir: str | Path) -> ConditionResult | None:
    """The condition's saved result, or None if there is none (or it is unreadable: treated as
    'not done', so the condition is simply run again)."""
    path = Path(condition_dir) / RESULT_FILE
    if not path.exists():
        return None
    try:
        return ConditionResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
