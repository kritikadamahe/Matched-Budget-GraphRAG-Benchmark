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
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_title: str
    word_start: int
    word_end: int

    def to_dict(self) -> dict:
        return asdict(self)


def _make_chunk_id(source_title: str, word_start: int, word_end: int) -> str:
    """Stable hash - same inputs always produce the same ID, forever."""
    key = f"{source_title}::{word_start}::{word_end}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def chunk_document(source_title: str, text: str, chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    """Split one document's text into overlapping, fixed-size word chunks."""
    words = text.split()
    if not words:
        return []

    step = chunk_size_words - overlap_words
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        chunk_id = _make_chunk_id(source_title, start, end)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=" ".join(chunk_words),
            source_title=source_title,
            word_start=start,
            word_end=end,
        ))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_corpus(documents: dict[str, str], chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    """
    documents: {source_title: full_text}
    Returns all chunks across all documents, in a stable order
    (sorted by title then position) so re-running is fully deterministic.
    """
    all_chunks: list[Chunk] = []
    for title in sorted(documents.keys()):
        all_chunks.extend(
            chunk_document(title, documents[title], chunk_size_words, overlap_words)
        )
    return all_chunks
