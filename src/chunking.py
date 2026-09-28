"""
Chunking.

WHY THIS EXISTS:
Every selection strategy operates on "chunks", not raw documents. For the
budget math (blueprint §G) to mean anything, chunk boundaries must be
IDENTICAL across every strategy and every budget level - the only thing
allowed to differ between conditions is which chunks get selected, never
how the corpus was cut up in the first place.

chunk_id is a deterministic hash of (source_title, char_start, char_end).
Re-running chunking on the same corpus always produces the same IDs - this
is what lets us later prove "strategy X selected chunk_id Y" reproducibly.

NOTE ON UNITS: the blueprint's original recommendation was ~300 GPT-tokens
per chunk (via tiktoken). tiktoken needs to download its vocab file from a
domain this sandbox cannot reach, so we chunk by WHITESPACE WORD COUNT
instead - an operational substitution, not a change to chunk size intent.
Swap in a real tokenizer later if exact token counts matter to you.

TITLE PREFIX: every chunk's text starts with its document title ("Title: ...").
In HotpotQA/MuSiQue the title is often the key entity and is frequently NOT
repeated in the body (e.g. "It won the Academy Award..." under "Forrest Gump"),
so without it the extractor and every text-based strategy would lose that entity.
Word offsets and chunk_ids refer to the body only, so they are unaffected.

SOURCE DOC ID: `source_doc_id` identifies the exact paragraph a chunk came from.
Titles alone are not unique: MuSiQue often has several different paragraphs of
the same Wikipedia article (all titled e.g. "Green") in one question's context.

GOLD LABELS: `is_gold_for_question_ids` lists the questions whose gold supporting
paragraph this chunk came from. It is filled in by src/corpus.py and is for
ANALYSIS ONLY (evidence recall) - no strategy may read it (blueprint §S).
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, asdict, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_title: str
    word_start: int
    word_end: int
    source_doc_id: str = ""                                            # which paragraph it came from
    is_gold_for_question_ids: list[str] = field(default_factory=list)  # analysis only

    def to_dict(self) -> dict:
        return asdict(self)


def _make_chunk_id(source_key: str, word_start: int, word_end: int) -> str:
    """Stable hash - same inputs always produce the same ID, forever."""
    key = f"{source_key}::{word_start}::{word_end}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def chunk_document(
    source_title: str,
    text: str,
    chunk_size_words: int,
    overlap_words: int,
    title_prefix: bool = True,
    source_doc_id: str | None = None,
) -> list[Chunk]:
    """Split one document's text into overlapping, fixed-size word chunks.
    chunk_size_words counts body words only; the title prefix comes on top.

    source_doc_id identifies the paragraph. It defaults to the title, but must be
    given when two different paragraphs share a title (common in MuSiQue), or
    their chunk_ids would collide."""
    words = text.split()
    if not words:
        return []

    doc_key = source_doc_id or source_title
    step = chunk_size_words - overlap_words
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc_key, start, end),
            text=(f"{source_title}: " if title_prefix else "") + " ".join(chunk_words),
            source_title=source_title,
            word_start=start,
            word_end=end,
            source_doc_id=doc_key,
        ))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_corpus(documents: dict, chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    """
    documents: either {doc_id: {"title": ..., "text": ...}} (what src/corpus.py
    produces) or the simple form {title: text}.
    Returns all chunks across all documents, in a stable order (sorted by title,
    then doc_id, then position) so re-running is fully deterministic.
    """
    docs = []
    for key, value in documents.items():
        if isinstance(value, str):
            docs.append((key, key, value))                      # {title: text}
        else:
            docs.append((value["title"], key, value["text"]))   # {doc_id: {...}}

    all_chunks: list[Chunk] = []
    for title, doc_id, text in sorted(docs, key=lambda d: (d[0], d[1])):
        all_chunks.extend(
            chunk_document(title, text, chunk_size_words, overlap_words, source_doc_id=doc_id)
        )
    return all_chunks
