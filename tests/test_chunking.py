import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document, chunk_corpus


def test_chunk_ids_are_deterministic():
    text = "one two three four five six seven eight nine ten"
    chunks_a = chunk_document("Doc A", text, chunk_size_words=4, overlap_words=1)
    chunks_b = chunk_document("Doc A", text, chunk_size_words=4, overlap_words=1)
    ids_a = [c.chunk_id for c in chunks_a]
    ids_b = [c.chunk_id for c in chunks_b]
    assert ids_a == ids_b, "Same input must always produce the same chunk_ids"


def test_overlap_actually_overlaps():
    text = "one two three four five six seven eight"
    chunks = chunk_document("Doc A", text, chunk_size_words=4, overlap_words=2)
    # chunk 1 = words[0:4], chunk 2 should start at word 2 (4-2 overlap)
    assert chunks[0].word_start == 0 and chunks[0].word_end == 4
    assert chunks[1].word_start == 2

def test_no_words_lost():
    text = " ".join(str(i) for i in range(23))  # 23 words
    chunks = chunk_document("Doc A", text, chunk_size_words=5, overlap_words=1)
    assert chunks[-1].word_end == 23, "Last chunk must reach the end of the document"


def test_chunk_corpus_is_order_stable():
    docs = {"Zeta": "a b c d e f", "Alpha": "g h i j k l"}
    chunks_1 = chunk_corpus(docs, chunk_size_words=3, overlap_words=0)
    chunks_2 = chunk_corpus(docs, chunk_size_words=3, overlap_words=0)
    assert [c.chunk_id for c in chunks_1] == [c.chunk_id for c in chunks_2]
    # Alpha sorts before Zeta - confirms deterministic ordering, not dict insertion order
    assert chunks_1[0].source_title == "Alpha"


def test_every_chunk_starts_with_its_title():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_document("Forrest Gump", text, chunk_size_words=4, overlap_words=1)
    assert all(c.text.startswith("Forrest Gump: ") for c in chunks)
    # offsets and ids refer to the body only, so the prefix does not shift them
    assert chunks[0].word_start == 0 and chunks[0].word_end == 4
    plain = chunk_document("Forrest Gump", text, chunk_size_words=4, overlap_words=1, title_prefix=False)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in plain]


def test_capitalized_phrases_fallback():
    from src.text_utils import capitalized_phrases
    assert capitalized_phrases("Robert Zemeckis\nParamount Pictures made it") == \
        ["Robert Zemeckis", "Paramount Pictures"]          # never joins across a line break
    assert "University of Alabama" in capitalized_phrases("He attended the University of Alabama.")
