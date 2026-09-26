import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import build_chunk_manifest
from src.budget import select_top_budget
from strategies.random_strategy import RandomStrategy
from strategies.ketrag_strategy import KETRAGStrategy


def _get_test_chunks():
    chunks, _, _ = build_chunk_manifest(
        dataset="hotpotqa", data_source="mock", num_questions=5, seed=42,
        chunk_size_words=20, chunk_overlap_words=4,
    )
    return chunks


def test_random_returns_full_permutation():
    chunks = _get_test_chunks()
    strategy = RandomStrategy(seed=1)
    ranking = strategy.rank(chunks)
    assert sorted(ranking) == sorted(c.chunk_id for c in chunks)
    assert len(ranking) == len(set(ranking))  # no duplicates


def test_random_is_deterministic_given_seed():
    chunks = _get_test_chunks()
    r1 = RandomStrategy(seed=99).rank(chunks)
    r2 = RandomStrategy(seed=99).rank(chunks)
    assert r1 == r2


def test_random_differs_across_seeds():
    chunks = _get_test_chunks()
    r1 = RandomStrategy(seed=1).rank(chunks)
    r2 = RandomStrategy(seed=2).rank(chunks)
    assert r1 != r2, "Different seeds should (almost certainly) give different orderings"


def test_ketrag_returns_full_permutation():
    chunks = _get_test_chunks()
    strategy = KETRAGStrategy(knn_k=3)
    ranking = strategy.rank(chunks)
    assert sorted(ranking) == sorted(c.chunk_id for c in chunks)
    assert len(ranking) == len(set(ranking))


def test_ketrag_is_deterministic():
    chunks = _get_test_chunks()
    r1 = KETRAGStrategy(knn_k=3).rank(chunks)
    r2 = KETRAGStrategy(knn_k=3).rank(chunks)
    assert r1 == r2, "PageRank + tie-break must give identical results on identical input"


def test_random_and_ketrag_rankings_differ():
    # Not a guarantee in general, but with real varied text they should not
    # coincidentally produce the exact same order - if they do, something's
    # probably wrong with one of the implementations.
    chunks = _get_test_chunks()
    r1 = RandomStrategy(seed=1).rank(chunks)
    r2 = KETRAGStrategy(knn_k=3).rank(chunks)
    assert r1 != r2


def test_all_strategies_converge_at_100_percent_budget():
    # Critical sanity check from blueprint §R: at budget=1.0, every strategy
    # must select the SAME set of chunks (all of them) - only the ORDER can differ.
    chunks = _get_test_chunks()
    all_ids = set(c.chunk_id for c in chunks)

    random_selected = set(select_top_budget(RandomStrategy(seed=1).rank(chunks), 1.0))
    ketrag_selected = set(select_top_budget(KETRAGStrategy(knn_k=3).rank(chunks), 1.0))

    assert random_selected == all_ids
    assert ketrag_selected == all_ids
    assert random_selected == ketrag_selected


def test_budget_cutoff_shrinks_selection_as_expected():
    chunks = _get_test_chunks()
    n = len(chunks)
    ranking = KETRAGStrategy(knn_k=3).rank(chunks)

    selected_10 = select_top_budget(ranking, 0.10)
    selected_50 = select_top_budget(ranking, 0.50)
    selected_100 = select_top_budget(ranking, 1.0)

    assert len(selected_10) <= len(selected_50) <= len(selected_100) == n
    # the 10% selection must be a prefix of the 50% selection (same ranking, smaller cutoff)
    assert selected_10 == selected_50[:len(selected_10)]
