# Matched-Budget GraphRAG Selection Benchmark

Evaluating retrieval quality vs. LLM extraction cost under fixed budgets,
comparing Random, KET-RAG, LazyGraphRAG, and FastGraphRAG chunk-selection
strategies on HotpotQA and MuSiQue.

**Status: Phases 1-3 + all four selection strategies.** Project skeleton, config
system, corpus/chunking pipeline (with analysis-only gold labels), budget math,
and all four strategies are implemented and tested, plus the native
LazyGraphRAG reference point. No LLM extraction, graph construction, retrieval,
or full benchmark run exists yet.

| Strategy | File | Blueprint category | How it ranks chunks |
|---|---|---|---|
| Random | `strategies/random_strategy.py` | A: faithful | seeded shuffle |
| KET-RAG | `strategies/ketrag_strategy.py` | A/B: faithful mechanism | PageRank on a **chunk** graph: K/2 keyword-overlap + K/2 embedding neighbours, K=2 (as in the paper); `ketrag_mode: tfidf` keeps the Phase 1 TF-IDF graph as an ablation |
| LazyGraphRAG (L3) | `strategies/lazygraphrag_strategy.py` | B: adaptation | k-means topics, most typical chunk per topic, round-robin |
| FastGraphRAG (F1) | `strategies/fastgraphrag_strategy.py` | C: simplified heuristic | sum of degree centrality of its **entities** (spaCy or regex) |
| Native LazyGraphRAG (L4) | `native/lazygraphrag_native.py` | reference point, no budget | no LLM at indexing; per-question LLM relevance checks, capped |

None of the strategies see questions, answers or gold labels (a test enforces
this). spaCy is optional: `pip install spacy && python -m spacy download en_core_web_sm`.
KET-RAG's default (faithful) mode needs a local embedding model:
`pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install sentence-transformers`
(CPU-only torch is enough, ~200 MB; the model all-MiniLM-L6-v2 downloads on first use, ~90 MB).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # not needed until later phases (no API key required yet)
```

## Run the tests

```bash
python3 -m pytest tests/ -v
```

Expected: all tests pass. This phase makes zero paid API calls.

## Build the real corpora (roadmap Phases 2-3)

```bash
python3 -m src.prepare_data --config configs/hotpotqa_dev.yaml   # HotpotQA, 20 questions
python3 -m src.prepare_data --config configs/musique_dev.yaml    # MuSiQue-Ans, 20 questions
```

The first run downloads the dev set (free, ~30 MB each) into `data/raw/<dataset>/`
from a pinned Hugging Face revision, so every teammate gets identical files. It
writes `documents.json`, `questions.json`, `chunks.json` and `meta.json` to
`data/corpus/<corpus_id>/` and checks the roadmap checkpoints: no duplicate
documents, every gold paragraph present, same chunk_ids on re-run. Downloads and
generated corpora are git-ignored.

| Corpus (seed 42, 250/40 words) | Paragraphs | Documents after dedup | N chunks | 5% budget |
|---|---|---|---|---|
| HotpotQA, 20 questions | 200 | 200 | 204 | 10 |
| MuSiQue, 20 questions | 399 | 399 | 402 | 20 |
| HotpotQA, 200 questions | 1,994 | 1,990 | 2,016 | 101 |
| MuSiQue, 200 questions | 3,997 | 3,370 | 3,398 | 170 |

Documents are deduplicated by exact (title, text), not by title: MuSiQue often
has several different paragraphs with the same title, and title-only dedup
would drop a gold paragraph in 466 of its 2,417 dev questions.

## Run the pipeline on the bundled mock dataset

```bash
python3 -c "
from src.corpus import build_chunk_manifest
chunks, questions, documents = build_chunk_manifest(
    dataset='hotpotqa', data_source='mock', num_questions=5, seed=42,
    chunk_size_words=250, chunk_overlap_words=40,
)
print(f'{len(chunks)} chunks from {len(documents)} deduplicated documents')
"
```

## Project structure

```
configs/        experiment YAML configs
data/raw/       mock fixtures (committed) + downloaded dev sets (git-ignored)
data/corpus/    generated corpus/chunk manifests (not committed - see .gitignore)
cache/          embeddings/extraction/judge-call caches (not committed)
src/            config, corpus (loaders), chunking, budget math, prepare_data
strategies/     selection strategies: random, ketrag, lazygraphrag (L3), fastgraphrag (F1)
native/         native LazyGraphRAG reference point (L4)
graph/          knowledge graph construction (Phase 8, not started)
retrieval/      graph retrieval (Phase 9, not started)
evaluation/     EM/F1/LLM-judge (Phase 11, not started)
experiments/    experiment runner (Phase 13, not started)
results/        run outputs (not committed)
notebooks/      analysis notebooks (Phase 15, not started)
tests/          one test file per component
```

## A note on `data_source`

Configs support `data_source: mock` (bundled fixture, works offline, used for
all current tests) and `data_source: huggingface` (real HotpotQA/MuSiQue,
downloaded once from a pinned revision, see "Build the real corpora" above).
