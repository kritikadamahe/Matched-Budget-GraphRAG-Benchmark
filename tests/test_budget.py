import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

from src.budget import selected_count, select_top_budget


@pytest.mark.parametrize("n,budget,expected", [
    (100, 0.05, 5),
    (100, 0.10, 10),
    (100, 0.25, 25),
    (100, 0.50, 50),
    (100, 0.75, 75),
    (100, 1.00, 100),
    (3, 0.10, 1),      # rounds to 0, but minimum is 1
    (7, 0.50, 4),       # round(3.5) -> 4 (banker's rounding note below)
])
def test_selected_count_matches_formula(n, budget, expected):
    assert selected_count(n, budget) == expected


def test_full_budget_selects_everything():
    ids = [f"c{i}" for i in range(37)]
    assert select_top_budget(ids, 1.0) == ids


def test_select_top_budget_preserves_rank_order():
    ids = ["best", "second", "third", "fourth"]
    assert select_top_budget(ids, 0.5) == ["best", "second"]


def test_rejects_bad_budget():
    with pytest.raises(ValueError):
        selected_count(100, 0.0)
    with pytest.raises(ValueError):
        selected_count(100, 1.5)
