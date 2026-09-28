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
    titles = [d["title"] for d in documents.values()]
    assert len(titles) == len(set(titles)), "No duplicate titles should survive pooling"
    # "Forrest Gump" appears in q1, q2 and q4 with identical text - kept once, not tripled
    assert titles.count("Forrest Gump") == 1


def test_unknown_data_source_raises_clear_error():
    import pytest
    with pytest.raises(ValueError):
        load_raw_questions("hotpotqa", "not-a-source", num_questions=5, seed=0)


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


def test_gold_titles_come_from_supporting_facts():
    questions = load_raw_questions("hotpotqa", "mock", num_questions=5, seed=0)
    by_id = {q["question_id"]: q for q in questions}
    assert by_id["q2"]["gold_titles"] == ["Alan Silvestri", "Forrest Gump"]
    for q in questions:
        titles = {t for t, _ in q["context"]}
        assert q["gold_titles"] and set(q["gold_titles"]) <= titles


def test_chunks_are_annotated_with_gold_question_ids():
    chunks, questions, _ = build_chunk_manifest(
        dataset="hotpotqa", data_source="mock", num_questions=5, seed=42,
        chunk_size_words=20, chunk_overlap_words=4,
    )
    gump = [c for c in chunks if c.source_title == "Forrest Gump"]
    assert gump and all(c.is_gold_for_question_ids == ["q1", "q2", "q4"] for c in gump)
    film_score = [c for c in chunks if c.source_title == "Film score"]
    assert all(c.is_gold_for_question_ids == [] for c in film_score)   # distractor only


def test_duplicate_titles_have_identical_text_in_fixture():
    # Real HotpotQA repeats a page verbatim under every question that cites it.
    # If the fixture's copies differed, dedup would silently drop facts.
    questions = load_raw_questions("hotpotqa", "mock", num_questions=5, seed=0)
    seen = {}
    for q in questions:
        for title, sentences in q["context"]:
            assert seen.setdefault(title, sentences) == sentences, title
