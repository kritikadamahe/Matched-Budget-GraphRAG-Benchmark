# Matched-Budget GraphRAG Selection Benchmark

Evaluating retrieval quality vs. LLM extraction cost under fixed budgets,
comparing Random, KET-RAG, LazyGraphRAG, and FastGraphRAG chunk-selection
strategies on HotpotQA and MuSiQue.

**Status: Phase 1 complete.** Project skeleton, config system, corpus/chunking
pipeline, and two of four strategies (Random, KET-RAG) are implemented and
tested. LazyGraphRAG and FastGraphRAG are not yet implemented. No LLM
extraction, graph construction, retrieval, or full benchmark run exists yet.

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
data/raw/       raw dataset fixtures (mock_hotpotqa.json for now)
data/corpus/    generated corpus/chunk manifests (not committed - see .gitignore)
cache/          embeddings/extraction/judge-call caches (not committed)
src/            config, corpus, chunking, budget-math modules
strategies/     selection strategy implementations (random, ketrag done;
                lazygraphrag, fastgraphrag not yet started)
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
requires internet access to huggingface.co - not yet implemented/tested).
