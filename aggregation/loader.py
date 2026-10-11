"""
Finding and vetting the Phase 10 results to aggregate.

Reads `<matrix_dir>/conditions/<id>/condition_result.json` (the authoritative per-condition record,
which, unlike results.jsonl, still has the full `config`) with the existing
experiments.schemas.ConditionResult, and `<matrix_dir>/matrix.json` (the planned conditions).

Nothing is silently dropped, picked or merged:
  - a result file that cannot be read             -> listed as excluded + an error issue
  - a planned condition with no result            -> listed as excluded ("missing_result")
  - two results for the same (dataset, strategy, budget, seed) cell
                                                  -> AggregationError, unless --on-duplicate latest
                                                     (then the choice is written into the report)
  - conditions whose configs differ beyond the cell coordinates (a different model, chunk size ...)
                                                  -> AggregationError, unless --allow-mixed-config
                                                     (then the report says so and is not reportable)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from aggregation.schemas import DuplicateRecord, ExcludedCondition, Issue
from experiments.schemas import RESULT_FILE, ConditionResult, read_condition_result

# per-condition coordinates and paths: these legitimately differ between conditions of one matrix
COORDINATE_KEYS = ("dataset", "strategy", "budget", "seed", "experiment_id")
PATH_KEYS = ("cache_dir", "output_dir")
INCLUDED_STATUSES = ("completed", "completed_with_failures")


class AggregationError(RuntimeError):
    """The input cannot be aggregated honestly (duplicates, mixed configs, no data ...)."""


@dataclass
class LoadedCondition:
    result: ConditionResult
    directory: Path
    matrix_dir: Path

    @property
    def cell(self) -> tuple:
        r = self.result
        return (r.dataset, r.strategy, r.budget_pct, r.seed)

    @property
    def label(self) -> str:
        return f"{self.matrix_dir}:{self.directory.name}"


@dataclass
class Discovery:
    conditions: list[LoadedCondition] = field(default_factory=list)
    excluded: list[ExcludedCondition] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    planned_cells: set[tuple] | None = field(default=None)       # None if any matrix lacks matrix.json
    matrix_ids: list[str] = field(default_factory=list)


def cell_dict(cell: tuple) -> dict:
    return {"dataset": cell[0], "strategy": cell[1], "budget_pct": cell[2], "seed": cell[3]}


def shared_config_fingerprint(config: dict) -> str | None:
    """Hash of everything in the config EXCEPT the cell coordinates and paths. Two conditions of one
    matrix must share it (same models, chunking, retrieval ...). None for an empty config."""
    if not config:
        return None
    data = {k: v for k, v in config.items() if k not in COORDINATE_KEYS and k not in PATH_KEYS}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------------------- discovery
def _read_planned(matrix_dir: Path, disc: Discovery):
    """(planned entries, matrix_id), or None if matrix.json is missing / unreadable."""
    path = matrix_dir / "matrix.json"
    if not path.exists():
        disc.issues.append(Issue(code="no_matrix_manifest", severity="warning",
                                 message=f"{path} not found: the planned conditions are unknown, so completeness "
                                         f"cannot be verified"))
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return list(doc["conditions"]), doc.get("matrix_id")
    except (ValueError, KeyError, OSError) as e:
        disc.issues.append(Issue(code="unreadable_matrix_manifest", severity="error",
                                 message=f"{path} could not be read: {e}"))
        return None


def discover(matrix_dirs: list[Path]) -> Discovery:
    """Load every condition result under the given matrix directories."""
    disc = Discovery()
    planned_all: set[tuple] | None = set()
    for matrix_dir in matrix_dirs:
        matrix_dir = Path(matrix_dir)
        conditions_dir = matrix_dir / "conditions"
        if not conditions_dir.is_dir():
            raise AggregationError(f"{matrix_dir} is not a matrix directory (no conditions/ folder inside)")
        planned = _read_planned(matrix_dir, disc)
        loaded_here: set[tuple] = set()
        unreadable_ids: set[str] = set()
        for sub in sorted(p for p in conditions_dir.iterdir() if p.is_dir()):
            path = sub / RESULT_FILE
            if not path.exists():
                disc.issues.append(Issue(
                    code="no_result_file", severity="warning", condition_id=sub.name,
                    message=f"{sub.name}: no {RESULT_FILE} (a run that never finished?); not counted"))
                continue
            result = read_condition_result(sub)
            if result is None:
                disc.issues.append(Issue(code="unreadable_result", severity="error", condition_id=sub.name,
                                         message=f"{path} exists but cannot be parsed"))
                disc.excluded.append(ExcludedCondition(kind="unreadable_result", condition_id=sub.name,
                                                       reason=f"{RESULT_FILE} is unreadable"))
                unreadable_ids.add(sub.name)
                continue
            if result.condition_id != sub.name:
                disc.issues.append(Issue(
                    code="directory_name_mismatch", severity="warning", condition_id=result.condition_id,
                    message=f"folder {sub.name!r} holds the result of {result.condition_id!r}",
                    expected=sub.name, actual=result.condition_id))
            lc = LoadedCondition(result=result, directory=sub, matrix_dir=matrix_dir)
            disc.conditions.append(lc)
            loaded_here.add(lc.cell)
            if result.matrix_id not in disc.matrix_ids:
                disc.matrix_ids.append(result.matrix_id)
        if planned is None:
            planned_all = None
            continue
        entries, matrix_id = planned
        if matrix_id and matrix_id not in disc.matrix_ids:
            disc.matrix_ids.append(matrix_id)
        for e in entries:
            cell = (e["dataset"], e["strategy"], e["budget_pct"], e["seed"])
            if planned_all is not None:
                planned_all.add(cell)
            if cell not in loaded_here and e["condition_id"] not in unreadable_ids:
                disc.excluded.append(ExcludedCondition(
                    kind="missing_result", condition_id=e["condition_id"], matrix_id=matrix_id,
                    dataset=cell[0], strategy=cell[1], budget_pct=cell[2], seed=cell[3],
                    reason="planned in matrix.json but has no readable result (never run, or still running)"))
    disc.planned_cells = planned_all
    if not disc.conditions:
        raise AggregationError("no condition results found under " + ", ".join(str(d) for d in matrix_dirs))
    return disc


# ------------------------------------------------------------------------------- duplicates
def resolve_duplicates(conditions: list[LoadedCondition], policy: str):
    """Returns (kept, dropped_as_excluded, duplicate_records). policy: 'error' (default) or 'latest'
    (keep the result with the latest finished_at_utc; a tie is still an error)."""
    if policy not in ("error", "latest"):
        raise AggregationError(f"unknown duplicate policy {policy!r} (use 'error' or 'latest')")
    by_cell: dict[tuple, list[LoadedCondition]] = {}
    for lc in conditions:
        by_cell.setdefault(lc.cell, []).append(lc)
    dups = {c: v for c, v in by_cell.items() if len(v) > 1}
    if dups and policy == "error":
        lines = [f"  {cell_dict(c)}: " + ", ".join(lc.label for lc in v) for c, v in sorted(dups.items())]
        raise AggregationError(
            f"{len(dups)} cell(s) have more than one result; refusing to choose silently:\n" + "\n".join(lines) +
            "\n  Remove the extra result(s), or pass --on-duplicate latest to keep the most recently finished one.")
    kept: list[LoadedCondition] = []
    dropped: list[ExcludedCondition] = []
    records: list[DuplicateRecord] = []
    for cell, group in by_cell.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        ordered = sorted(group, key=lambda lc: lc.result.finished_at_utc, reverse=True)
        if ordered[0].result.finished_at_utc == ordered[1].result.finished_at_utc:
            raise AggregationError(f"cell {cell_dict(cell)} has results finished at the same time "
                                   f"({ordered[0].label}, {ordered[1].label}); cannot pick 'latest'")
        winner, losers = ordered[0], ordered[1:]
        kept.append(winner)
        records.append(DuplicateRecord(cell=cell_dict(cell), kept=winner.label, dropped=[lc.label for lc in losers],
                                       policy=policy))
        for lc in losers:
            r = lc.result
            dropped.append(ExcludedCondition(
                kind="duplicate_dropped", condition_id=r.condition_id, matrix_id=r.matrix_id, dataset=r.dataset,
                strategy=r.strategy, budget_pct=r.budget_pct, seed=r.seed, status=r.status,
                reason=f"older duplicate of {winner.label} (--on-duplicate latest)"))
    return kept, dropped, records


# ------------------------------------------------------------------------------- configs
def judge_enabled_of(result: ConditionResult) -> bool | None:
    """Whether the judge ran for this condition: known only if evaluation produced metrics."""
    if result.metrics and "judge_enabled" in result.metrics:
        return bool(result.metrics["judge_enabled"])
    return None


def check_configs(conditions: list[LoadedCondition], allow_mixed: bool):
    """Verify the conditions are comparable. Returns (fingerprints, mixed, differences). Raises
    AggregationError on a mix unless allow_mixed."""
    prints = sorted({fp for lc in conditions if (fp := shared_config_fingerprint(lc.result.config))})
    differences: dict = {}
    if len(prints) > 1:
        keys = sorted({k for lc in conditions for k in lc.result.config if k not in COORDINATE_KEYS and k not in PATH_KEYS})
        for k in keys:
            values = {json.dumps(lc.result.config.get(k), sort_keys=True, default=str) for lc in conditions if lc.result.config}
            if len(values) > 1:
                differences[k] = sorted(values)[:6]
    judge_flags = {j for lc in conditions if (j := judge_enabled_of(lc.result)) is not None}
    if len(judge_flags) > 1:
        differences["judge_enabled (--no-judge)"] = ["true", "false"]
    mixed = bool(differences)
    if mixed and not allow_mixed:
        shown = "; ".join(f"{k}: {v}" for k, v in list(differences.items())[:8])
        raise AggregationError(
            "the conditions were run with different configurations, so they are not comparable "
            f"({len(prints)} distinct shared configs). Differing settings: {shown}\n"
            "  Re-run them under one config, or pass --allow-mixed-config to aggregate anyway "
            "(the result is then marked not reportable).")
    return prints, mixed, differences
