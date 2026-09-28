"""
Build, save and verify the pooled corpus + chunk manifest for one config
(roadmap Phases 2 and 3; blueprint §E, §F).

    python -m src.prepare_data --config configs/hotpotqa_dev.yaml
    python -m src.prepare_data --config configs/musique_dev.yaml

Writes data/corpus/<corpus_id>/:
    documents.json   {doc_id: {title, text}}      the deduplicated corpus
    questions.json   question list with gold_doc_ids (analysis only)
    chunks.json      the chunk manifest (N chunks, identical for every strategy/budget)
    meta.json        settings, counts, per-budget chunk counts, chunk-id fingerprint

and checks the roadmap checkpoints:
    Phase 2: no duplicate (title, text) documents; every gold paragraph is in the corpus
    Phase 3: rebuilding from scratch gives exactly the same chunk_ids
The first run with data_source=huggingface downloads the dev set (free, ~30 MB).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.budget import selected_count
from src.config import ExperimentConfig, load_config
from src.corpus import PROJECT_ROOT, HF_SOURCES, build_chunk_manifest

BUDGETS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


def corpus_id(cfg: ExperimentConfig) -> str:
    """Name of the corpus - everything that determines N, and nothing else."""
    return (f"{cfg.dataset}_{cfg.data_source}_{cfg.num_questions}q_seed{cfg.seed}"
            f"_w{cfg.chunk_size_words}o{cfg.chunk_overlap_words}")


def fingerprint(chunk_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest()[:16]


def build(cfg: ExperimentConfig):
    return build_chunk_manifest(cfg.dataset, cfg.data_source, cfg.num_questions, cfg.seed,
                                cfg.chunk_size_words, cfg.chunk_overlap_words)


def check(cfg: ExperimentConfig, chunks, questions, documents) -> dict:
    """Roadmap checkpoints. Raises AssertionError if any fails."""
    pairs = [(d["title"], d["text"]) for d in documents.values()]
    assert len(pairs) == len(set(pairs)), "duplicate (title, text) documents in corpus"
    missing = [q["question_id"] for q in questions
               if not q["gold_doc_ids"] or any(d not in documents for d in q["gold_doc_ids"])]
    assert not missing, f"questions with missing gold paragraphs: {missing[:5]}"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id collision"

    rebuilt, _, _ = build(cfg)
    assert [c.chunk_id for c in rebuilt] == ids, "chunk_ids changed on re-run"
    return {"phase2_no_duplicate_docs": True, "phase2_all_gold_present": True,
            "phase3_deterministic_chunk_ids": True}


def prepare(cfg: ExperimentConfig, out_root: Path | None = None) -> dict:
    chunks, questions, documents = build(cfg)
    checks = check(cfg, chunks, questions, documents)

    n = len(chunks)
    gold_chunks = sum(1 for c in chunks if c.is_gold_for_question_ids)
    titles = [d["title"] for d in documents.values()]
    meta = {
        "corpus_id": corpus_id(cfg),
        "dataset": cfg.dataset,
        "data_source": cfg.data_source,
        "source": HF_SOURCES.get(cfg.dataset) if cfg.data_source == "huggingface" else "mock fixture",
        "num_questions": len(questions),
        "seed": cfg.seed,
        "chunk_size_words": cfg.chunk_size_words,
        "chunk_overlap_words": cfg.chunk_overlap_words,
        "num_paragraphs_before_dedup": sum(len(q["context"]) for q in questions),
        "num_documents": len(documents),
        "num_titles_shared_by_several_documents": len(titles) - len(set(titles)),
        "N_chunks": n,
        "chunks_per_budget": {f"{b:.0%}": selected_count(n, b) for b in BUDGETS},
        "avg_gold_docs_per_question": round(sum(len(q["gold_doc_ids"]) for q in questions) / len(questions), 2),
        "gold_chunk_share": round(gold_chunks / n, 3),
        "avg_words_per_chunk": round(sum(c.word_end - c.word_start for c in chunks) / n, 1),
        "chunk_id_fingerprint": fingerprint([c.chunk_id for c in chunks]),
        "checks": checks,
    }

    out = (out_root or PROJECT_ROOT / "data" / "corpus") / meta["corpus_id"]
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in [("documents.json", documents),
                      ("questions.json", questions),
                      ("chunks.json", [c.to_dict() for c in chunks]),
                      ("meta.json", meta)]:
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    meta["output_dir"] = str(out)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    meta = prepare(load_config(args.config))
    print(f"config: {args.config}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
