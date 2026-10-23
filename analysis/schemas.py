"""
Record shapes for Phase 12 (analysis). This first increment holds what the loader produces; the
statistics, evidence, table and summary records are added by the later increments.

THE UNIT OF ANALYSIS IS THE QUESTION. One `QuestionObservation` is one question's outcome under one
condition (dataset, strategy, budget, seed, Random shuffle). Everything the later modules do starts
from these rows; a repeated measurement of the same question (Random's shuffles) is never counted as
an extra sample.

MISSING IS NOT ZERO. `metric_value` returns (value, None) for a measured value and (None, reason) for a
missing one. The only 0s are real scores: an abstention scores 0 for EM, F1 and the judge (Phase 8 rule),
a failed generation or judge call is *missing*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aggregation.schemas import AggregateReport, ConditionRecord, Issue

ANALYSIS_SCHEMA_VERSION = "1"

# --- metrics -------------------------------------------------------------------------------------
METRICS = ("f1", "judge_accuracy", "em")
PRIMARY_METRICS = ("f1", "judge_accuracy")        # the confirmatory family (see analysis/README section)
SECONDARY_METRICS = ("em",)                       # descriptive / secondary

# --- what the analysis is allowed to say about its input -------------------------------------------
# real_complete     every planned condition is present, reportable and reconciled
# real_partial      genuine data, but some planned conditions are missing / unfinished (needs --allow-incomplete)
# mock_layout_test  mock data: for checking layouts only; never findings (needs --allow-mock-layout-test)
AnalysisStatus = Literal["real_complete", "real_partial", "mock_layout_test"]
DataKind = Literal["real", "mock"]

# outcome: the Phase 8 outcome, or "no_prediction" for a question the runner could not answer at all
# (retrieval raised): it has no row in predictions.jsonl / scored.jsonl.
Outcome = Literal["answered", "abstained", "empty", "failed", "no_prediction"]

# why a metric value is missing
MissingReason = Literal["generation_failed", "no_prediction", "judge_failed", "judge_disabled", "judge_not_run"]

_JUDGE_FAILURES = ("api_error", "malformed", "refused", "truncated")


class QuestionObservation(BaseModel):
    """One question under one condition. Gold answers are NOT stored here (only the scores computed from them)."""
    model_config = ConfigDict(frozen=True)

    condition_id: str
    dataset: str
    strategy: str
    budget_pct: int
    seed: int
    shuffle_seed: int | None = None
    question_id: str

    outcome: Outcome
    generation_status: str | None = None          # None for "no_prediction"
    em: float | None = None
    f1: float | None = None
    judge_correct: bool | None = None
    judge_status: str | None = None
    retrieved_chunk_ids: tuple[str, ...] = ()     # chunks whose text was in the context (for evidence metrics)
    n_context_words: int | None = None


def metric_value(obs: QuestionObservation, metric: str) -> tuple[float | None, MissingReason | None]:
    """(value, None) when measured, (None, reason) when missing. `judge_accuracy` is 1.0 / 0.0 per question."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    if obs.outcome == "no_prediction":
        return None, "no_prediction"
    if metric in ("em", "f1"):
        value = obs.em if metric == "em" else obs.f1
        return (None, "generation_failed") if value is None else (float(value), None)
    # judge_accuracy
    if obs.judge_correct is not None:
        return (1.0 if obs.judge_correct else 0.0), None
    if obs.outcome == "failed" or obs.judge_status == "skipped_failed_generation":
        return None, "generation_failed"
    if obs.judge_status == "disabled":
        return None, "judge_disabled"
    if obs.judge_status in _JUDGE_FAILURES:
        return None, "judge_failed"
    return None, "judge_not_run"


# --- the loaded input ------------------------------------------------------------------------------
class SourceFile(BaseModel):
    """An input file the analysis read, with its hash (provenance)."""
    path: str                       # relative to the matrix directory
    sha256: str


class ConditionInput(BaseModel):
    """What the analysis knows about one INCLUDED condition besides its per-question rows."""
    condition_id: str
    directory: str                  # relative to the matrix directory
    dataset: str
    strategy: str
    budget_pct: int
    seed: int
    shuffle_seed: int | None = None
    corpus_id: str | None = None
    status: str
    n_questions: int
    question_ids: list[str]         # the question universe of this condition (predicted + failed)
    n_no_prediction: int = 0
    selected_chunk_ids: list[str] | None = None   # from extraction/selection.json (None = file missing)


class AnalysisInput(BaseModel):
    """Everything the later modules need, after the mock / reportability / alignment gate has passed."""
    schema_version: str = ANALYSIS_SCHEMA_VERSION
    matrix_dir: str
    matrix_ids: list[str]
    status: AnalysisStatus
    data_kind: DataKind
    judge_enabled: bool | None             # None = could not be determined
    question_seed: dict[str, int]          # dataset -> the one question seed
    question_ids: dict[str, list[str]]     # dataset -> the common question universe
    conditions: list[ConditionInput]
    observations: list[QuestionObservation]
    records: list[ConditionRecord]         # the Phase 11 rows of ALL loaded conditions (cost / tokens / runtime)
    excluded: list[dict] = Field(default_factory=list)       # aborted / failed / missing conditions, with reasons
    report: AggregateReport                # the Phase 11 report (reportability, methodology, groups)
    issues: list[Issue] = Field(default_factory=list)        # warnings only: errors refuse the input
    allowed_incomplete: bool = False
    source_files: list[SourceFile] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
