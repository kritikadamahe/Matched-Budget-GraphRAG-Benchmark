"""Phase 10 runner: the 48-condition matrix, condition ids, per-condition configs, matrix loading."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools

import pytest
import yaml

from experiments import (DEFAULT_BUDGETS, DEFAULT_DATASETS, DEFAULT_STRATEGIES, Condition, MatrixSpec,
                         condition_cfg, config_hash, generate_conditions, load_matrix, select_conditions)
from extraction.run import make_run_id
from src.config import ExperimentConfig
from tests.helpers_experiments import base_cfg


# ------------------------------------------------------------------- the 48 conditions
def test_default_matrix_is_exactly_the_48_conditions():
    conds = generate_conditions(MatrixSpec(), 42)
    assert len(conds) == 48 == 2 * 4 * 6
    assert {(c.dataset, c.strategy, c.budget_pct) for c in conds} == set(
        itertools.product(("hotpotqa", "musique"), ("random", "ketrag", "lazygraphrag", "fastgraphrag"),
                          (5, 10, 25, 50, 75, 100)))
    assert {c.seed for c in conds} == {42}
    assert DEFAULT_DATASETS == ("hotpotqa", "musique") and DEFAULT_BUDGETS == (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
    assert DEFAULT_STRATEGIES == ("random", "ketrag", "lazygraphrag", "fastgraphrag")


def test_condition_ids_are_unique_stable_and_directory_safe():
    conds = generate_conditions(MatrixSpec(), 42)
    ids = [c.condition_id for c in conds]
    assert len(set(ids)) == 48
    assert ids[0] == "hotpotqa__random__b005__seed42" and ids[-1] == "musique__fastgraphrag__b100__seed42"
    assert Condition("hotpotqa", "ketrag", 0.25, 7).condition_id == "hotpotqa__ketrag__b025__seed7"
    assert ids == [c.condition_id for c in generate_conditions(MatrixSpec(), 42)]          # stable across calls
    assert all(i.replace("_", "").isalnum() for i in ids)
    # the id depends on the four coordinates only: reordering the spec does not rename a condition
    shuffled = MatrixSpec(datasets=["musique", "hotpotqa"], strategies=["fastgraphrag", "random", "ketrag", "lazygraphrag"],
                          budgets=[1.0, 0.05, 0.5, 0.25, 0.1, 0.75])
    assert {c.condition_id for c in generate_conditions(shuffled, 42)} == set(ids)


def test_order_is_dataset_major_with_ascending_budgets():
    conds = generate_conditions(MatrixSpec(), 42)
    first_block = conds[:6]
    assert {c.strategy for c in first_block} == {"random"} and [c.budget_pct for c in first_block] == [5, 10, 25, 50, 75, 100]
    assert {c.dataset for c in conds[:24]} == {"hotpotqa"} and {c.dataset for c in conds[24:]} == {"musique"}


def test_seeds_multiply_the_matrix_and_keep_ids_distinct():
    conds = generate_conditions(MatrixSpec(seeds=[1, 2]), 42)
    assert len(conds) == 96 and len({c.condition_id for c in conds}) == 96 and {c.seed for c in conds} == {1, 2}


@pytest.mark.parametrize("kwargs", [
    {"datasets": ["squad"]}, {"datasets": []}, {"datasets": ["hotpotqa", "hotpotqa"]},
    {"strategies": ["lazygraphrag_native"]}, {"strategies": ["nope"]}, {"strategies": []},
    {"budgets": [0.0]}, {"budgets": [1.5]}, {"budgets": [0.125]}, {"budgets": [0.05, 0.05]}, {"budgets": [0.1, 0.100001]},
    {"seeds": []}, {"seeds": [1, 1]}, {"colour": "red"},
])
def test_invalid_matrix_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MatrixSpec(**kwargs)


def test_the_native_reference_point_is_rejected_with_an_explanation():
    with pytest.raises(ValueError, match="no pre-extraction budget"):
        MatrixSpec(strategies=["random", "lazygraphrag_native"])


# ------------------------------------------------------------------- subsets
def test_select_conditions_filters_and_limits():
    conds = generate_conditions(MatrixSpec(), 42)
    assert len(select_conditions(conds, datasets=["hotpotqa"])) == 24
    assert len(select_conditions(conds, strategies=["random", "ketrag"], budgets=[0.05, 1.0])) == 2 * 2 * 2
    assert [c.condition_id for c in select_conditions(conds, ids=["musique__random__b010__seed42"])] == ["musique__random__b010__seed42"]
    assert len(select_conditions(conds, limit=5)) == 5
    assert select_conditions(conds) == conds


@pytest.mark.parametrize("kwargs", [{"datasets": ["squad"]}, {"strategies": ["nope"]}, {"budgets": [0.07]},
                                    {"ids": ["x__y"]}, {"limit": 0}])
def test_unknown_selections_are_errors_not_silent_empty_runs(kwargs):
    with pytest.raises(ValueError):
        select_conditions(generate_conditions(MatrixSpec(), 42), **kwargs)


# ------------------------------------------------------------------- per-condition configs
def test_condition_cfg_overrides_the_four_coordinates_and_the_experiment_id(tmp_path):
    base = base_cfg(tmp_path, judge_model="gpt-4o")
    cfg = condition_cfg(base, Condition("musique", "ketrag", 0.5, 7), "bench")
    assert (cfg.dataset, cfg.strategy, cfg.budget, cfg.seed) == ("musique", "ketrag", 0.5, 7)
    assert cfg.experiment_id == "bench__musique__ketrag__b050__seed7"
    assert cfg.judge_model == "gpt-4o" and cfg.num_questions == base.num_questions        # the rest is shared
    assert base.strategy == "random" and base.budget == 0.1                                # base untouched


def test_run_ids_are_unique_across_all_48_conditions(tmp_path):
    base = base_cfg(tmp_path)
    run_ids = {make_run_id(condition_cfg(base, c, "m")) for c in generate_conditions(MatrixSpec(), 42)}
    assert len(run_ids) == 48


def test_condition_cfg_revalidates_so_invalid_cells_are_caught(tmp_path):
    # Valid as a base (strategy random), but faithful KET-RAG on TF-IDF embeddings is invalid for the ketrag cells.
    base = base_cfg(tmp_path, ketrag_mode="faithful", embedding_backend="tfidf")
    condition_cfg(base, Condition("hotpotqa", "random", 0.5, 42), "m")                    # fine
    with pytest.raises(ValueError, match="faithful"):
        condition_cfg(base, Condition("hotpotqa", "ketrag", 0.5, 42), "m")


def test_config_hash_tracks_result_relevant_settings_only(tmp_path):
    a = base_cfg(tmp_path)
    assert config_hash(a) == config_hash(base_cfg(tmp_path))
    assert config_hash(a) != config_hash(base_cfg(tmp_path, retrieval_top_m=10))
    assert config_hash(a) != config_hash(base_cfg(tmp_path, generation_model="gpt-4o"))
    assert config_hash(a) == config_hash(base_cfg(tmp_path / "elsewhere"))                  # paths are not part of the result


# ------------------------------------------------------------------- loading
def test_load_matrix_reads_the_matrix_block_and_the_shared_config(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({
        "experiment_id": "bench", "seed": 5, "dataset": "hotpotqa", "strategy": "random", "budget": 0.1,
        "num_questions": 3, "matrix": {"datasets": ["musique"], "budgets": [0.25, 1.0], "seeds": [5, 6]}}))
    cfg, spec = load_matrix(path)
    assert cfg.experiment_id == "bench" and spec.datasets == ["musique"] and spec.strategies == list(DEFAULT_STRATEGIES)
    assert len(generate_conditions(spec, cfg.seed)) == 1 * 4 * 2 * 2


def test_no_matrix_block_means_the_full_default_matrix(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"experiment_id": "b", "seed": 1, "dataset": "hotpotqa", "strategy": "random",
                                    "budget": 0.1, "num_questions": 3}))
    cfg, spec = load_matrix(path)
    assert len(generate_conditions(spec, cfg.seed)) == 48


def test_a_typo_in_the_matrix_block_is_an_error_not_a_smaller_matrix(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"experiment_id": "b", "seed": 1, "dataset": "hotpotqa", "strategy": "random",
                                    "budget": 0.1, "num_questions": 3, "matrix": {"budget": [0.1]}}))
    with pytest.raises(ValueError):
        load_matrix(path)
    with pytest.raises(FileNotFoundError):
        load_matrix(tmp_path / "missing.yaml")


def test_the_shipped_mock_config_defines_the_48_conditions():
    root = Path(__file__).resolve().parents[1]
    cfg, spec = load_matrix(root / "configs" / "benchmark_mock.yaml")
    assert len(generate_conditions(spec, cfg.seed)) == 48
    assert (cfg.extraction_backend, cfg.generation_backend, cfg.judge_backend, cfg.data_source) == ("mock",) * 4
