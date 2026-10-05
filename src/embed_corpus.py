"""
Embed (and cache) a corpus with the shared embedding model - blueprint Phase 4.

    python -m src.embed_corpus --config configs/hotpotqa_dev.yaml

Embeds every chunk and every question of the config's corpus into
cache/embeddings/<model>.sqlite, then reports hits, misses and whether the model
had to be loaded. Run it twice: the second run must show misses=0 and
model_loads=0 (the Phase 4 checkpoint: "cache hit on 2nd run, 0 API calls").

Strategies and retrieval read the same cache, so after this step ranking a corpus
needs no embedding computation at all.
"""

from __future__ import annotations

import argparse
import time

from src.config import ExperimentConfig, load_config
from src.embeddings import CachedEmbedder, embedder_from_config
from src.prepare_data import build, corpus_id


def embed_corpus(cfg: ExperimentConfig, embedder=None) -> dict:
    embedder = embedder or embedder_from_config(cfg)
    if not isinstance(embedder, CachedEmbedder):
        raise ValueError("embed_corpus only makes sense for the cached semantic backend "
                         "(embedding_backend: sentence-transformers); TF-IDF is not cacheable.")
    chunks, questions, _ = build(cfg)
    t = time.perf_counter()
    chunk_vectors = embedder.embed([c.text for c in chunks])
    question_vectors = embedder.embed([q["question"] for q in questions])
    return {
        "corpus_id": corpus_id(cfg),
        "model": embedder.name,
        "n_chunks": len(chunks),
        "n_questions": len(questions),
        "dim": int(chunk_vectors.shape[1]),
        "hits": embedder.stats["hits"],
        "misses": embedder.stats["misses"],
        "model_loads": embedder.stats["model_loads"],
        "seconds": round(time.perf_counter() - t, 2),
        "cache_file": str(embedder.cache.path) if embedder.cache is not None else None,
        "checkpoint_fully_cached": embedder.stats["misses"] == 0 and embedder.stats["model_loads"] == 0,
        "_shapes_ok": chunk_vectors.shape[0] == len(chunks) and question_vectors.shape[0] == len(questions),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    report = embed_corpus(load_config(args.config))
    print(f"config: {args.config}")
    for k, v in report.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
