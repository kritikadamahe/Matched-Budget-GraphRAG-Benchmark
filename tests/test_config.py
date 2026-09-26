import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, ExperimentConfig


def test_dev_config_loads():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "dev.yaml")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.dataset == "hotpotqa"
    assert cfg.strategy == "random"
    assert cfg.budget == 0.10


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(Exception):
        ExperimentConfig(
            experiment_id="bad", seed=1, dataset="hotpotqa", data_source="mock",
            num_questions=5, chunk_size_words=100, chunk_overlap_words=200,
            strategy="random", budget=0.1,
        )


def test_budget_must_be_in_range():
    with pytest.raises(Exception):
        ExperimentConfig(
            experiment_id="bad", seed=1, dataset="hotpotqa", data_source="mock",
            num_questions=5, strategy="random", budget=1.5,
        )
