"""
The experimental matrix (Phase 10): which conditions exist, how they are named, and the exact
ExperimentConfig each one runs with.

THE 48 CONDITIONS = 2 datasets x 4 budgeted strategies x 6 budgets (one seed):

    datasets   hotpotqa, musique
    strategies random, ketrag, lazygraphrag, fastgraphrag   (strategies.STRATEGIES)
    budgets    5%, 10%, 25%, 50%, 75%, 100%

The native LazyGraphRAG reference point (L4) is NOT part of the matrix: it has no pre-extraction
budget, so it is not a (strategy, budget) cell. It is rejected here rather than silently dropped.

CONDITION ID   {dataset}__{strategy}__b{pct:03d}__seed{seed}      e.g. hotpotqa__ketrag__b025__seed42
Stable (a function of the four coordinates only), unique within a matrix, safe as a directory name.
Each condition's ExperimentConfig gets experiment_id = "{matrix_id}__{condition_id}", so the existing
extraction.run.make_run_id() is unique per condition too.

WHERE THE MATRIX IS DEFINED: a `matrix:` block inside an ordinary config YAML. Everything else in
the file is the shared ExperimentConfig (models, backends, chunking, retrieval, caps ...). The
file's own dataset / strategy / budget are placeholders: each condition overrides them. Example:

    matrix:
      datasets: [hotpotqa, musique]
      strategies: [random, ketrag, lazygraphrag, fastgraphrag]
      budgets: [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
      seeds: [42]            # optional; default = the file's `seed`

SEED LIMITATION (known, not changed here): the single `seed` selects the questions - hence the
corpus, N and the budget counts - AND drives Random's shuffle. A `seeds` list therefore varies the
corpus as well as Random's ranking; conditions are only comparable within one seed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import ExperimentConfig
from strategies import STRATEGIES

DEFAULT_DATASETS = ("hotpotqa", "musique")
DEFAULT_STRATEGIES = ("random", "ketrag", "lazygraphrag", "fastgraphrag")
DEFAULT_BUDGETS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
_DATASETS = ("hotpotqa", "musique")


def budget_pct(budget: float) -> int:
    """Whole-percent form of a budget fraction (0.05 -> 5). Budgets must be whole percentages so
    that condition ids are exact."""
    pct = round(budget * 100)
    if not (1 <= pct <= 100) or abs(budget * 100 - pct) > 1e-6:
        raise ValueError(f"budget {budget!r} must be a whole percentage between 1% and 100%")
    return pct


class MatrixSpec(BaseModel):
    """The `matrix:` block. Unknown keys are an error (a typo must not silently shrink the matrix)."""
    model_config = ConfigDict(extra="forbid")

    datasets: list[str] = Field(default_factory=lambda: list(DEFAULT_DATASETS))
    strategies: list[str] = Field(default_factory=lambda: list(DEFAULT_STRATEGIES))
    budgets: list[float] = Field(default_factory=lambda: list(DEFAULT_BUDGETS))
    seeds: list[int] | None = Field(None, description="Default: just the config's `seed`.")

    @field_validator("datasets")
    @classmethod
    def _datasets(cls, v):
        bad = [d for d in v if d not in _DATASETS]
        if bad or not v or len(set(v)) != len(v):
            raise ValueError(f"datasets must be a non-empty list of unique values from {_DATASETS}; got {v}")
        return v

    @field_validator("strategies")
    @classmethod
    def _strategies(cls, v):
        if "lazygraphrag_native" in v:
            raise ValueError("lazygraphrag_native (L4) has no pre-extraction budget, so it is not a matrix "
                             "cell; run it separately as the reference point")
        bad = [s for s in v if s not in STRATEGIES]
        if bad or not v or len(set(v)) != len(v):
            raise ValueError(f"strategies must be a non-empty list of unique values from {tuple(STRATEGIES)}; got {v}")
        return v

    @field_validator("budgets")
    @classmethod
    def _budgets(cls, v):
        pcts = [budget_pct(b) for b in v]
        if not v or len(set(pcts)) != len(pcts):
            raise ValueError(f"budgets must be non-empty and unique as whole percentages; got {v}")
        return v

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, v):
        if v is not None and (not v or len(set(v)) != len(v)):
            raise ValueError("seeds must be a non-empty list of unique integers")
        return v


@dataclass(frozen=True)
class Condition:
    dataset: str
    strategy: str
    budget: float
    seed: int

    @property
    def budget_pct(self) -> int:
        return budget_pct(self.budget)

    @property
    def condition_id(self) -> str:
        return f"{self.dataset}__{self.strategy}__b{self.budget_pct:03d}__seed{self.seed}"


def generate_conditions(spec: MatrixSpec, base_seed: int) -> list[Condition]:
    """Every (dataset, strategy, seed, budget) cell, in a fixed order: dataset-major, then seed,
    strategy, and ascending budget (so a strategy's ranking is reused across its budgets)."""
    seeds = spec.seeds if spec.seeds is not None else [base_seed]
    budgets = sorted(spec.budgets)
    return [Condition(dataset=d, strategy=s, budget=b, seed=seed)
            for d, seed, s, b in itertools.product(spec.datasets, seeds, spec.strategies, budgets)]


def select_conditions(conditions: list[Condition], *, datasets=None, strategies=None, budgets=None,
                      ids=None, limit: int | None = None) -> list[Condition]:
    """A subset of the matrix (for pilots and reruns). Filters must name things that exist."""
    out = list(conditions)
    known = {
        "datasets": {c.dataset for c in conditions}, "strategies": {c.strategy for c in conditions},
        "budgets": {c.budget_pct for c in conditions}, "ids": {c.condition_id for c in conditions},
    }
    wanted_budgets = None if budgets is None else {budget_pct(b) for b in budgets}
    for label, wanted in (("datasets", None if datasets is None else set(datasets)),
                          ("strategies", None if strategies is None else set(strategies)),
                          ("budgets", wanted_budgets), ("ids", None if ids is None else set(ids))):
        if wanted is not None and (wanted - known[label]):
            raise ValueError(f"unknown {label} {sorted(wanted - known[label])}; the matrix has {sorted(known[label])}")
    if datasets is not None:
        out = [c for c in out if c.dataset in set(datasets)]
    if strategies is not None:
        out = [c for c in out if c.strategy in set(strategies)]
    if wanted_budgets is not None:
        out = [c for c in out if c.budget_pct in wanted_budgets]
    if ids is not None:
        out = [c for c in out if c.condition_id in set(ids)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        out = out[:limit]
    return out


def condition_cfg(base: ExperimentConfig, cond: Condition, matrix_id: str) -> ExperimentConfig:
    """The ExperimentConfig one condition runs with. Rebuilt (not model_copy) so every validator
    runs again: e.g. faithful KET-RAG on TF-IDF embeddings is rejected here, per condition."""
    data = base.model_dump()
    data.update(dataset=cond.dataset, strategy=cond.strategy, budget=cond.budget, seed=cond.seed,
                experiment_id=f"{matrix_id}__{cond.condition_id}")
    return ExperimentConfig(**data)


_NOT_PART_OF_RESULT = ("cache_dir", "output_dir")


def config_hash(cfg: ExperimentConfig) -> str:
    """Fingerprint of everything that can change a condition's results (paths excluded). Stored in
    each result; a completed result is only reused if the current config hashes the same."""
    data = {k: v for k, v in cfg.model_dump().items() if k not in _NOT_PART_OF_RESULT}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def load_matrix(path: str | Path) -> tuple[ExperimentConfig, MatrixSpec]:
    """(shared base config, matrix spec) from one YAML file. No `matrix:` block = the full default 48."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    matrix_raw = raw.pop("matrix", None)
    spec = MatrixSpec(**(matrix_raw or {}))
    return ExperimentConfig(**raw), spec
