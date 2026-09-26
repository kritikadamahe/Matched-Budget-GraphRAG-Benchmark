"""
Corpus construction (blueprint §E).

WHY THIS EXISTS:
HotpotQA/MuSiQue give ~10 short context paragraphs PER QUESTION - too small
to make a 5%-vs-100% budget spectrum meaningful. So we pool the contexts of
a fixed subset of N questions into one shared corpus, deduplicating any
paragraph that appears under more than one question (a popular Wikipedia
page can show up as context for several different questions).

This module produces two things per run:
1. A "documents" dict: {title: full_text} - the deduplicated corpus.
2. A question list, so we can later grade answers question-by-question
   against the ONE shared graph built from that corpus.

The chunk manifest (chunking applied on top of this corpus) is produced by
build_chunk_manifest(), which calls src/chunking.py.
"""

from __future__ import annotations
import json
import random
from pathlib import Path

from src.chunking import chunk_corpus, Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_DATA_PATHS = {
    "hotpotqa": PROJECT_ROOT / "data" / "raw" / "mock_hotpotqa.json",
    # musique mock fixture will be added when we actually need MuSiQue runs.
}


def load_raw_questions(dataset: str, data_source: str, num_questions: int, seed: int) -> list[dict]:
    """
    Returns a list of question dicts: {question_id, question, answer, context}
    where context is [[title, [sentences]], ...] - same shape as raw HotpotQA.
    """
    if data_source == "mock":
        path = MOCK_DATA_PATHS.get(dataset)
        if path is None or not path.exists():
            raise FileNotFoundError(f"No mock fixture available for dataset={dataset!r} yet.")
        with open(path) as f:
            all_questions = json.load(f)
    elif data_source == "huggingface":
        # Real dataset loading - requires internet access to huggingface.co,
        # which this sandbox cannot reach. This path is written to run
        # correctly on a machine WITH normal internet access (your laptop,
        # or a Claude Code session with full network).
        raise NotImplementedError(
            "data_source='huggingface' needs `pip install datasets` and internet "
            "access to huggingface.co. Not available in this sandbox - use "
            "data_source='mock' here, and switch to 'huggingface' when you run "
            "this on your own machine."
        )
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    if num_questions > len(all_questions):
        raise ValueError(
            f"Requested {num_questions} questions but only {len(all_questions)} "
            f"are available in this fixture. Reduce num_questions in your config."
        )

    rng = random.Random(seed)
    sampled = rng.sample(all_questions, num_questions)
    # Sort by question_id after sampling so the SET is seed-dependent but the
    # ORDER we process them in is deterministic given that set.
    sampled.sort(key=lambda q: q["question_id"])
    return sampled


def pool_and_dedup_contexts(questions: list[dict]) -> dict[str, str]:
    """
    Pools every question's context paragraphs into one corpus, deduplicating
    by title (the same Wikipedia page appearing under two questions is only
    kept once). Returns {title: full_text}.
    """
    documents: dict[str, str] = {}
    for q in questions:
        for title, sentences in q["context"]:
            text = " ".join(sentences)
            if title in documents:
                # Same title seen before - keep the longer version if they
                # differ (defensive; in real HotpotQA data these should be
                # byte-identical across questions).
                if len(text) > len(documents[title]):
                    documents[title] = text
            else:
                documents[title] = text
    return documents


def build_chunk_manifest(
    dataset: str,
    data_source: str,
    num_questions: int,
    seed: int,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> tuple[list[Chunk], list[dict], dict[str, str]]:
    """
    Full corpus pipeline, end to end:
    load questions -> pool+dedup -> chunk.
    Returns (chunks, questions, documents) so callers/tests can inspect any stage.
    """
    questions = load_raw_questions(dataset, data_source, num_questions, seed)
    documents = pool_and_dedup_contexts(questions)
    chunks = chunk_corpus(documents, chunk_size_words, chunk_overlap_words)
    return chunks, questions, documents


def save_manifest(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)
