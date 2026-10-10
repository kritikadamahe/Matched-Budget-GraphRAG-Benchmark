"""Experiment runner (Phase 10): orchestrates the existing pipeline over the matched-budget matrix.

    experiments/conditions.py   the 48-condition matrix: spec, ids, per-condition config
    experiments/schemas.py      ConditionResult / StageRecord: the raw per-condition measurements
    experiments/runner.py       run_condition(), run_matrix(), plan_matrix(), results index
    experiments/run.py          CLI:  python -m experiments.run --config CONFIG [--dry-run]

The runner contains no strategy, extraction, graph, retrieval, generation or evaluation logic of
its own: it calls the existing modules in order and records what they report.
Importing this package loads only the light modules; import experiments.runner for the pipeline.
"""

from experiments.conditions import (DEFAULT_BUDGETS, DEFAULT_DATASETS, DEFAULT_STRATEGIES, Condition, MatrixSpec,
                                    condition_cfg, config_hash, generate_conditions, load_matrix,
                                    select_conditions)
from experiments.schemas import (RUNNER_SCHEMA_VERSION, STAGES, ConditionResult, QuestionFailure, StageRecord,
                                 read_condition_result, write_condition_result)

__all__ = ["Condition", "ConditionResult", "DEFAULT_BUDGETS", "DEFAULT_DATASETS", "DEFAULT_STRATEGIES",
           "MatrixSpec", "QuestionFailure", "RUNNER_SCHEMA_VERSION", "STAGES", "StageRecord", "condition_cfg",
           "config_hash", "generate_conditions", "load_matrix", "read_condition_result", "select_conditions",
           "write_condition_result"]
