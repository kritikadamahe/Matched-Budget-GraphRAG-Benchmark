"""
Record shapes for Phase 11 (cost / runtime aggregation).

THE CENTRAL RULE: a number is only a number when it was measured. Every quantity is a `Measure`:

    status        meaning                                                       value / lower_bound
    ------------  ------------------------------------------------------------  --------------------
    measured      complete: every item that should be counted was counted       value (may be 0.0 = a MEASURED zero)
    lower_bound   partial: some items unknown or the stage stopped early         lower_bound = the known part
    unknown       it exists but cannot be reconstructed (see `reason`)           both None
    not_run       the stage that would produce it never ran                      both None
    disabled      switched off on purpose (e.g. no judge)                        both None
    not_applicable the denominator / coverage needed does not exist              both None

`value` is None unless status == "measured", so a missing measurement can never be read as zero.
A total is "measured" only if every REQUIRED (enabled) component is measured.

`cache_dependent` marks figures that change with cache state (this run's elapsed time, money
spent this run). The comparable figures (cost_if_uncached, tokens and call runtimes rebuilt from the
per-item files) are not cache dependent.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AGGREGATE_SCHEMA_VERSION = "1"

MeasureStatus = Literal["measured", "lower_bound", "unknown", "not_run", "disabled", "not_applicable"]
MeasureKind = Literal["cost", "spent", "tokens", "runtime", "elapsed", "normalized"]


class Measure(BaseModel):
    kind: MeasureKind
    status: MeasureStatus
    value: int | float | None = None
    lower_bound: int | float | None = None
    reason: str | None = None                  # why it is not a complete measurement
    source: str | None = None                  # where it came from
    components: list[str] | None = None        # for totals: the components that were required
    cache_dependent: bool = False


class Issue(BaseModel):
    """A visible problem found while loading / reconciling. Never silently corrected."""
    code: str
    severity: Literal["error", "warning"]
    condition_id: str | None = None
    message: str
    expected: Any = None
    actual: Any = None


class ConditionRecord(BaseModel):
    """One flat, self-describing row per condition: what Phase 12 reads (conditions.jsonl)."""
    schema_version: str = AGGREGATE_SCHEMA_VERSION

    # --- identity / configuration (as recorded by the runner) ---
    condition_id: str
    matrix_id: str
    run_id: str | None = None
    dataset: str
    data_source: str
    strategy: str
    budget: float
    budget_pct: int
    seed: int
    shuffle_seed: int | None = None                   # Random's shuffle seed (matrix random_seeds), else None
    corpus_id: str | None = None
    config_hash: str | None = None
    shared_config_fingerprint: str | None = None      # config minus dataset/strategy/budget/seed/experiment_id/paths
    config: dict = Field(default_factory=dict)

    # --- outcome ---
    status: str
    clean: bool = False
    error: str | None = None
    problems: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    included: bool = False                            # contributes to group statistics
    exclusion_reason: str | None = None
    is_mock: bool = False
    reportable: bool = False
    not_reportable_reasons: list[str] = Field(default_factory=list)
    judge_enabled: bool | None = None                 # None = unknown (evaluation never ran)
    reconciled: bool = True                           # False if any per-item cross-check failed (see issues)
    issues: list[Issue] = Field(default_factory=list)

    # --- counts and failures (plain integers: always known) ---
    counts: dict[str, int | None] = Field(default_factory=dict)
    failures: dict = Field(default_factory=dict)

    # --- quality metrics, copied from the condition result (None = not computed). Evidence recall
    #     (selection_recall, retrieval_recall, ...) and the 95% interval bounds (em_ci95_low ...) are
    #     merged in as flat keys so they reach conditions.csv and the matched-cell statistics. ---
    metrics: dict = Field(default_factory=dict)
    evidence: dict = Field(default_factory=dict)
    ci95: dict = Field(default_factory=dict)

    # --- cost / tokens / runtime ---
    measures: dict[str, Measure] = Field(default_factory=dict)

    started_at_utc: str = ""
    finished_at_utc: str = ""


# ------------------------------------------------------------------------------ groups
class Stat(BaseModel):
    """Statistics of one measure over the INCLUDED conditions of a group. Counts come first on
    purpose: a mean is only meaningful next to how many values stand behind it."""
    n_records: int
    n_known: int                 # status == measured
    n_lower_bound: int           # partial values: never mixed into mean / min / max
    n_unknown: int               # unknown + not_run
    n_not_applicable: int        # disabled + not_applicable
    complete: bool               # every applicable record is measured
    sum: int | float | None = None                 # only when complete (and the measure is additive)
    sum_lower_bound: int | float | None = None     # known values + lower bounds: a true lower bound of the sum
    mean: float | None = None                      # over the n_known measured values only
    min: int | float | None = None
    max: int | float | None = None
    std: float | None = None                       # sample std (n-1); None unless n_known >= 2


class MetricStat(BaseModel):
    n_records: int
    n_known: int
    n_missing: int
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None


class GroupSummary(BaseModel):
    level: str
    group_by: list[str]
    comparison: Literal["matched_budget_cell", "descriptive_rollup"]
    comparison_note: str
    key: dict
    n_conditions: int            # every condition in the group, included or not
    n_included: int
    n_excluded: int
    excluded_reasons: dict[str, int] = Field(default_factory=dict)
    statuses: dict[str, int] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)
    shuffle_seeds: list[int] = Field(default_factory=list)    # Random's repeated shuffles in this group
    n_mock: int = 0
    n_reportable: int = 0
    reportable: bool = False
    not_reportable_reasons: list[str] = Field(default_factory=list)
    n_unreconciled: int = 0
    config_fingerprints: list[str] = Field(default_factory=list)
    condition_ids: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)      # sums over included conditions
    n_conditions_with_failures: int = 0
    measures: dict[str, Stat] = Field(default_factory=dict)
    metrics: dict[str, MetricStat] | None = None              # None for descriptive rollups
    metrics_omitted_reason: str | None = None


class ExcludedCondition(BaseModel):
    kind: Literal["aborted", "failed", "missing_result", "unreadable_result", "duplicate_dropped"]
    condition_id: str
    matrix_id: str | None = None
    dataset: str | None = None
    strategy: str | None = None
    budget_pct: int | None = None
    seed: int | None = None
    shuffle_seed: int | None = None
    status: str | None = None
    reason: str


class DuplicateRecord(BaseModel):
    cell: dict
    kept: str                    # "<matrix_dir>:<condition_id>"
    dropped: list[str]
    policy: str


class AggregateReport(BaseModel):
    schema_version: str = AGGREGATE_SCHEMA_VERSION
    generated_at_utc: str
    matrix_dirs: list[str]
    matrix_ids: list[str]
    options: dict

    n_conditions_loaded: int
    n_included: int
    n_excluded: int
    n_planned: int | None = None                  # from matrix.json; None if unavailable
    matrix_complete: bool | None = None
    statuses: dict[str, int] = Field(default_factory=dict)

    is_mock: bool
    n_mock: int
    n_reportable_conditions: int
    reportable: bool
    not_reportable_reasons: list[str]

    shared_config_fingerprints: list[str] = Field(default_factory=list)
    mixed_config: bool = False
    config_differences: dict = Field(default_factory=dict)

    excluded: list[ExcludedCondition] = Field(default_factory=list)
    duplicates: list[DuplicateRecord] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    n_errors: int = 0
    n_warnings: int = 0

    methodology: dict = Field(default_factory=dict)
    groups: list[GroupSummary] = Field(default_factory=list)
