import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus import load_raw_questions, pool_and_dedup_contexts, build_chunk_manifest


def test_load_mock_questions():
    questions = load_raw_questions("hotpotqa", "mock", num_questions=5, seed=0)
    assert len(questions) == 5
    assert all("question_id" in q for q in questions)


def test_sampling_is_deterministic_given_seed():
    a = load_raw_questions("hotpotqa", "mock", num_questions=3, seed=7)
    b = load_raw_questions("hotpotqa", "mock", num_questions=3, seed=7)
    assert [q["question_id"] for q in a] == [q["question_id"] for q in b]


def test_dedup_removes_duplicate_titles():
    # Our mock fixture deliberately repeats "Forrest Gump", "Tom Hanks", etc.
    # across multiple questions - this is realistic (popular pages get cited
    # by many multi-hop questions) and must collapse to ONE entry per title.
    questions = load_raw_questions("hotpotqa", "mock", num_questions=5, seed=0)
    documents = pool_and_dedup_contexts(questions)
    titles = list(documents.keys())
    assert len(titles) == len(set(titles)), "No duplicate titles should survive pooling"
    assert "Forrest Gump" in documents  # appears in q1 and q2 - should be deduped, not doubled


def test_missing_huggingface_raises_clear_error():
    import pytest
    with pytest.raises(NotImplementedError):
        load_raw_questions("hotpotqa", "huggingface", num_questions=5, seed=0)


def test_full_pipeline_end_to_end():
    chunks, questions, documents = build_chunk_manifest(
        dataset="hotpotqa", data_source="mock", num_questions=5, seed=42,
        chunk_size_words=20, chunk_overlap_words=4,
    )
    assert len(questions) == 5
    assert len(documents) > 0
    assert len(chunks) > 0
    # every chunk_id must be unique
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
