"""
Corpus construction (blueprint §E).

WHY THIS EXISTS:
HotpotQA/MuSiQue give ~10-20 short context paragraphs PER QUESTION - too small
to make a 5%-vs-100% budget spectrum meaningful. So we pool the contexts of
a fixed subset of N questions into one shared corpus, deduplicating any
paragraph that appears under more than one question (a popular Wikipedia
page can show up as context for several different questions).

This module produces three things per run:
1. A "documents" dict: {doc_id: {"title", "text"}} - the deduplicated corpus.
2. A question list, so we can later grade answers question-by-question
   against the ONE shared graph built from that corpus. Each question keeps
   "gold_doc_ids" (the exact supporting paragraphs) and "gold_titles". These
   are ANALYSIS-ONLY labels (blueprint §F, §S) used to measure how much gold
   evidence a strategy selected; strategies never see them.
3. The chunk manifest, via build_chunk_manifest() -> src/chunking.py.

WHY DEDUP IS BY (title, text), NOT BY TITLE ALONE:
Checked on the real dev sets. In MuSiQue, 1,293 of 2,417 questions contain
several DIFFERENT paragraphs with the same title (e.g. two "Green" paragraphs),
and in 466 questions title-only dedup would silently drop a gold paragraph.
In HotpotQA, 61 titles appear with different text under different questions.
So a document is identified by doc_id = hash(title, text): identical paragraphs
collapse to one document, different paragraphs with one title stay separate.

DATA SOURCES:
- "mock": small bundled fixtures in data/raw/ (offline, used by the tests).
- "huggingface": the real dev sets, downloaded once into data/raw/<dataset>/
  from a pinned revision (reproducible) and read locally afterwards.
    HotpotQA  distractor dev (7,405 q)  - hotpotqa/hotpot_qa (official)
    MuSiQue   MuSiQue-Ans dev (2,417 q) - bdsaglam/musique, a mirror of the
              official musique_ans_v1.0_dev.jsonl (verified: 2,417 questions,
              1,252 / 760 / 405 two/three/four-hop, as in the MuSiQue paper).
"""

from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

from src.chunking import chunk_corpus, Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

MOCK_DATA_PATHS = {
    "hotpotqa": RAW_DIR / "mock_hotpotqa.json",
    "musique": RAW_DIR / "mock_musique.jsonl",
}

# Pinned Hugging Face sources: same files for every teammate, forever.
HF_SOURCES = {
    "hotpotqa": {
        "repo_id": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "filename": "distractor/validation-00000-of-00001.parquet",
    },
    "musique": {
        "repo_id": "bdsaglam/musique",
        "revision": "22873a405dd809893b22ada0b499299fb612d2df",
        "filename": "musique_ans_v1.0_dev.jsonl",
    },
}


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------
def raw_file_path(dataset: str) -> Path:
    return RAW_DIR / dataset / HF_SOURCES[dataset]["filename"]


def ensure_raw_file(dataset: str) -> Path:
    """Return the local path of the real dev set, downloading it once if missing."""
    path = raw_file_path(dataset)
    if path.exists():
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("Real data needs: pip install huggingface_hub pyarrow") from e
    src = HF_SOURCES[dataset]
    downloaded = hf_hub_download(
        repo_id=src["repo_id"], filename=src["filename"], revision=src["revision"],
        repo_type="dataset", local_dir=RAW_DIR / dataset,
    )
    return Path(downloaded)


# ---------------------------------------------------------------------------
# Normalisation: every dataset -> one question format
# ---------------------------------------------------------------------------
# {question_id, question, answer, answer_aliases,
#  context: [[title, [sentences]], ...],
#  supporting_paragraphs: [indices into context]}

def normalize_hotpotqa(row: dict) -> dict:
    """Accepts both the raw HotpotQA JSON layout (used by the mock fixture) and
    the Hugging Face parquet layout (dicts of parallel arrays)."""
    ctx = row["context"]
    if isinstance(ctx, dict):   # HF: {"title": [...], "sentences": [[...], ...]}
        context = [[str(t), [str(s) for s in sents]] for t, sents in zip(ctx["title"], ctx["sentences"])]
    else:                       # raw JSON: [[title, [sentences]], ...]
        context = [[t, list(sents)] for t, sents in ctx]

    sf = row.get("supporting_facts", [])
    gold_titles = set(sf["title"]) if isinstance(sf, dict) else {t for t, _ in sf}
    return {
        "question_id": str(row.get("question_id", row.get("id"))),
        "question": row["question"],
        "answer": row["answer"],
        "answer_aliases": [],
        "context": context,
        # HotpotQA titles are unique within one question's context (checked on the
        # full dev set), so a title identifies the supporting paragraph.
        "supporting_paragraphs": [i for i, (t, _) in enumerate(context) if t in gold_titles],
    }


def normalize_musique(row: dict) -> dict:
    paragraphs = row["paragraphs"]
    return {
        "question_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "answer_aliases": list(row.get("answer_aliases", [])),
        "context": [[p["title"], [p["paragraph_text"]]] for p in paragraphs],
        # MuSiQue marks gold per paragraph, which matters because titles repeat.
        "supporting_paragraphs": [i for i, p in enumerate(paragraphs) if p["is_supporting"]],
    }


def _read_all(dataset: str, data_source: str) -> list[dict]:
    if data_source == "mock":
        path = MOCK_DATA_PATHS.get(dataset)
        if path is None or not path.exists():
            raise FileNotFoundError(f"No mock fixture available for dataset={dataset!r}.")
    elif data_source == "huggingface":
        path = ensure_raw_file(dataset)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    if path.suffix == ".parquet":
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)

    if dataset == "hotpotqa":
        return [normalize_hotpotqa(r) for r in rows]
    if dataset == "musique":
        return [normalize_musique(r) for r in rows if r.get("answerable", True)]
    raise ValueError(f"Unknown dataset: {dataset}")


def load_raw_questions(dataset: str, data_source: str, num_questions: int, seed: int) -> list[dict]:
    """
    Sample `num_questions` questions (fixed seed) in the normalised format:
        {question_id, question, answer, answer_aliases, context,
         supporting_paragraphs, gold_titles}
    where context is [[title, [sentences]], ...].
    """
    all_questions = _read_all(dataset, data_source)
    if num_questions > len(all_questions):
        raise ValueError(
            f"Requested {num_questions} questions but only {len(all_questions)} "
            f"are available for {dataset}/{data_source}. Reduce num_questions in your config."
        )

    rng = random.Random(seed)
    sampled = rng.sample(all_questions, num_questions)
    # Sort by question_id after sampling so the SET is seed-dependent but the
    # ORDER we process them in is deterministic given that set.
    sampled.sort(key=lambda q: q["question_id"])
    for q in sampled:
        q["gold_titles"] = sorted({q["context"][i][0] for i in q["supporting_paragraphs"]})
    return sampled


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def paragraph_text(sentences: list[str]) -> str:
    """HotpotQA sentences carry their own leading spaces; normalise whitespace."""
    return " ".join(" ".join(s.split()) for s in sentences if s.strip())


def make_doc_id(title: str, text: str) -> str:
    return hashlib.sha1(f"{title}\x1f{text}".encode("utf-8")).hexdigest()[:12]


def pool_and_dedup_contexts(questions: list[dict]) -> dict[str, dict]:
    """
    Pools every question's context paragraphs into one corpus, deduplicating
    identical (title, text) paragraphs. Returns {doc_id: {"title", "text"}}.
    Also sets q["gold_doc_ids"] on every question (analysis only).
    """
    documents: dict[str, dict] = {}
    for q in questions:
        doc_ids = []
        for title, sentences in q["context"]:
            text = paragraph_text(sentences)
            doc_id = make_doc_id(title, text)
            documents.setdefault(doc_id, {"title": title, "text": text})
            doc_ids.append(doc_id)
        q["gold_doc_ids"] = sorted({doc_ids[i] for i in q.get("supporting_paragraphs", [])})
    return documents


def annotate_gold(chunks: list[Chunk], questions: list[dict]) -> None:
    """Fill Chunk.is_gold_for_question_ids from each question's gold_doc_ids.
    Analysis only - called after chunking, never read by any strategy."""
    gold_by_doc: dict[str, list[str]] = {}
    for q in questions:
        for doc_id in q.get("gold_doc_ids", []):
            gold_by_doc.setdefault(doc_id, []).append(q["question_id"])
    for c in chunks:
        c.is_gold_for_question_ids = sorted(gold_by_doc.get(c.source_doc_id, []))


def build_chunk_manifest(
    dataset: str,
    data_source: str,
    num_questions: int,
    seed: int,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> tuple[list[Chunk], list[dict], dict[str, dict]]:
    """
    Full corpus pipeline, end to end:
    load questions -> pool+dedup -> chunk -> annotate gold.
    Returns (chunks, questions, documents) so callers/tests can inspect any stage.
    """
    questions = load_raw_questions(dataset, data_source, num_questions, seed)
    documents = pool_and_dedup_contexts(questions)
    chunks = chunk_corpus(documents, chunk_size_words, chunk_overlap_words)
    annotate_gold(chunks, questions)
    return chunks, questions, documents


def save_manifest(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)
