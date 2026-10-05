# Matched-Budget GraphRAG Selection Benchmark

Evaluating retrieval quality vs. LLM extraction cost under fixed budgets,
comparing Random, KET-RAG, LazyGraphRAG, and FastGraphRAG chunk-selection
strategies on HotpotQA and MuSiQue.

**Status:** project skeleton, config system, corpus/chunking pipeline (with
analysis-only gold labels), budget math, all four selection strategies, the native
LazyGraphRAG reference point, the **cached embedding pipeline** and the **LLM
extraction layer** (budget-selected chunks only), **knowledge-graph construction** and
**personalised-PageRank retrieval** and **answer generation** (plus the native L4 relevance
check). Evaluation and the full benchmark runner do not exist yet.

**Phase numbers** follow the Blueprint's roadmap (§Q). The team's earlier labels differ
from Phase 4 on, so both are listed:

| Blueprint phase | Team label | What | Status |
|---|---|---|---|
| 1 | Phase 1 | environment, config | done |
| 2-3 | Phases 2-3 | real datasets, pooled corpus, chunking | done |
| 4 | - | cached embeddings | done |
| 5-6 | Phase 1 | strategy interface, Random, budget math | done |
| 7 | "Phase 4" | LLM extraction | done |
| 12 | Phases 1-3 | KET-RAG, LazyGraphRAG, FastGraphRAG | done |
| 8 | "Phase 5" | knowledge-graph construction | done |
| 9 | | retrieval (personalised PageRank) | done |
| 10 | | answer generation (+ native L4 relevance check) | done |
| 11 | | evaluation (EM, F1, LLM judge) | next |
| 13 / 14 / 15 | | experiment runner / full benchmark / analysis | to do |

| Strategy | File | Blueprint category | How it ranks chunks |
|---|---|---|---|
| Random | `strategies/random_strategy.py` | A: faithful | seeded shuffle |
| KET-RAG | `strategies/ketrag_strategy.py` | A/B: faithful mechanism | PageRank on a **chunk** graph: K/2 keyword-overlap + K/2 embedding neighbours, K=2 (as in the paper); `ketrag_mode: tfidf` keeps the Phase 1 TF-IDF graph as an ablation |
| LazyGraphRAG (L3) | `strategies/lazygraphrag_strategy.py` | B: adaptation | k-means topics, most typical chunk per topic, round-robin |
| FastGraphRAG (F1) | `strategies/fastgraphrag_strategy.py` | C: simplified heuristic | sum of degree centrality of its **entities** (spaCy or regex) |
| Native LazyGraphRAG (L4) | `native/lazygraphrag_native.py` | reference point, no budget | no LLM at indexing; per-question LLM relevance checks, capped |

None of the strategies see questions, answers or gold labels (a test enforces
this). spaCy is optional: `pip install spacy && python -m spacy download en_core_web_sm`.
KET-RAG (faithful mode), LazyGraphRAG L3 and the native L4 share one local embedding model
(see "Embeddings" below):
`pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install sentence-transformers`
(CPU-only torch is enough, ~200 MB; the model all-MiniLM-L6-v2 downloads on first use, ~90 MB).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # only needed for REAL extraction (extraction_backend: openai); everything else is free
```

## Run the tests

```bash
python3 -m pytest tests/ -v
```

Expected: all tests pass. Tests make zero API calls (network access is blocked inside the
extraction tests and no API key is visible to them).

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

## Embeddings (Blueprint Phase 4)

One shared model, `all-MiniLM-L6-v2` (local, free), gives the vectors for KET-RAG's
semantic neighbours, LazyGraphRAG L3's topics, the native L4's question-chunk relevance
and, later, retrieval. Every text is embedded at most once and cached in
`cache/embeddings/<model>.sqlite` (git-ignored), keyed by the model and the text, so a
fully cached run never even loads the model.

```bash
python3 -m src.embed_corpus --config configs/hotpotqa_dev.yaml   # 1st run: embeds and caches
python3 -m src.embed_corpus --config configs/hotpotqa_dev.yaml   # 2nd run: misses=0, model_loads=0
```

| Config field | Default | Meaning |
|---|---|---|
| `embedding_backend` | `sentence-transformers` | or `tfidf` (tests/ablations; needs no torch, not cached) |
| `embedding_model` | `all-MiniLM-L6-v2` | local sentence-transformers model |
| `embedding_cache_enabled` | `true` | |

`ketrag_mode: faithful` requires the semantic backend: TF-IDF is itself keyword-based, so
it cannot be the "semantic" half (the config rejects that combination).

Measured on the 200-question corpora (laptop CPU):

| | First run | Cached |
|---|---|---|
| KET-RAG ranking, HotpotQA (N=2,016) | 29.1 s | 1.8 s |
| KET-RAG ranking, MuSiQue (N=3,398) | 50.7 s | 5.4 s |
| LazyGraphRAG L3 ranking | 28-65 s with TF-IDF | ~1 s |

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

## Phase 4: LLM extraction

```
corpus -> chunks -> selection strategy -> budget -> budget-selected chunks
       -> LLM entity/relationship extraction -> entities + relationships
       -> (later phase) knowledge graph construction
```

Phase 4 is **only** the extraction layer (`extraction/`). It does not build a graph,
retrieve, answer questions or evaluate, and it is not the benchmark runner.

| Piece | File | Notes |
|---|---|---|
| Schemas | `extraction/schemas.py` | `ExtractionInput`, `Entity`, `Relationship`, `ExtractionResult`; 6 fixed entity types: PERSON, ORGANIZATION, LOCATION, EVENT, WORK, OTHER |
| Interface | `extraction/base_extractor.py` | `extract(ExtractionInput) -> ExtractionResult`, same contract for every extractor |
| Mock extractor | `extraction/mock_extractor.py` | free, deterministic, offline; **testing only** (heuristic, types are arbitrary) |
| OpenAI extractor | `extraction/openai_extractor.py` | GPT-4o-mini, strict structured JSON output, retries, validation |
| Prompt | `extraction/prompts.py` | `PROMPT_VERSION` is part of the cache key |
| Cache | `extraction/cache.py` | `cache/extractions/<extractor>/<model>/<key>.json` |
| Pipeline | `extraction/pipeline.py` | select -> budget gate -> cache -> extract -> save |
| One-run CLI | `extraction/run.py` | one config -> one extraction run (not a benchmark runner) |

**Faithful vs adaptation.** Faithful to the benchmark design: only budget-selected chunks
are extracted, and every strategy is extracted with the identical extractor, model, prompt
and settings, so the strategy is the only variable. **Adaptation (blueprint category B):**
the extraction itself is "GraphRAG-style" with our own prompt and a simplified schema (no
gleaning, no claim extraction); it is not Microsoft GraphRAG's prompt or KET-RAG's code.
**Heuristic:** `MockExtractor`; its output must never appear in a reported result.
Note for the graph phase: real KET-RAG also builds a cheap keyword graph for the chunks it
does *not* extract. Here unselected chunks are sent nowhere, so that choice is deferred.

### Try it for free (mock extractor, no key, no network)

```bash
python3 -m extraction.run --config configs/extraction_dev.yaml --dry-run   # plan only
python3 -m extraction.run --config configs/extraction_dev.yaml             # full run
python3 -m extraction.run --config configs/extraction_dev.yaml             # again: all cache hits
```

Results go to `results/extractions/<run_id>/`: `extractions.jsonl` (one result per selected
chunk), `selection.json` (exactly which chunks were selected, for auditing) and
`run_summary.json` (model, tokens, cost, runtime, status counts, cache hits, validation issues).
Exit codes: 0 = all ok and cost fully known, 1 = finished with problems (failed or truncated
chunks, or token usage missing from the API so the cost is unknown), 2 = stopped (bad config,
missing key, fatal API error, spending cap).

### Real extraction with OpenAI (costs money)

1. `python3 -m src.prepare_data --config configs/hotpotqa_dev.yaml` (free, downloads the data).
2. `python3 -m extraction.run --config configs/extraction_openai_example.yaml --dry-run`
   prints the chunk count and an **estimated** cost. It needs no API key and sends nothing.
3. Put `OPENAI_API_KEY=...` in `.env` (git-ignored), then run the same command without `--dry-run`.

With `extraction_backend: openai` and no key the run **stops with a clear error. It never
falls back to the mock.** Prices are config values (`extraction_price_input_per_1m`,
`extraction_price_output_per_1m`), not hardcoded: check them on OpenAI's pricing page before
a real run. As a rough scale (assuming ~730 input + ~250 output tokens per chunk) extraction
costs a few hundredths of a cent per chunk, so 100% of a 20-question corpus is about five cents.
`--dry-run` shows the estimate for your exact corpus. The optional `extraction_max_cost_usd`
cap refuses to start if the estimate is above it, and stops mid-run (keeping finished work in
the cache) if spending reaches it; it can overshoot by at most the estimation error of one call.

### Config (all optional, in the YAML)

| Field | Default | Meaning |
|---|---|---|
| `extraction_backend` | `mock` | `mock` or `openai` |
| `extraction_model` | `gpt-4o-mini` | model used by the openai backend |
| `extraction_temperature` | `0.0` | |
| `extraction_max_output_tokens` | `8192` | cap on one reply; a reply that hits it is marked `truncated` and is **not** retried |
| `extraction_max_retries` | `3` | retries per chunk (temporary errors, malformed replies) |
| `extraction_request_timeout_s` | `60` | |
| `extraction_cache_enabled` | `true` | |
| `extraction_price_input_per_1m` / `_output_per_1m` | `0.15` / `0.60` | USD per 1M tokens |
| `extraction_max_cost_usd` | none | optional spending cap per run |

### How the pipeline behaves

- **Budget gate.** The pipeline takes (all chunks, the strategy's ranking, the budget), picks
  the chunks with the existing `select_top_budget()`, and wraps the extractor in a gate that
  raises `BudgetViolation` for any other chunk. The extractor only ever receives
  `{chunk_id, text}`: no gold labels, questions or answers.
- **Cache.** Key = SHA-256 of (extractor, model, prompt version, schema version, temperature,
  max output tokens, seed) plus the chunk text. Change any of them and old entries are not
  reused. Only successful results with known usage are cached; failed or truncated chunks are retried next run; a crashed
  run resumes for free. The same text selected by two strategies is one API call. `cache/` and
  `results/` are git-ignored, so API responses are never committed.
- **Cost reporting.** `cost_spent_usd` is what this run actually paid (cache hits cost 0).
  `cost_if_uncached_usd` counts cached entries at their original cost: **use this one for a
  strategy's cost on the budget-vs-quality plot**, otherwise a cache hit makes it look free.
  **Unknown is never free:** if an API response carries no usage information, that chunk's usage
  is recorded as unknown (`null`), no token counts are invented, and every total that includes it
  becomes `null`. `usage_complete` is `false`, `n_usage_unknown` and `usage_unknown_chunk_ids` say
  which chunks, and `usage_known_part_lower_bound` gives what *is* known (a lower bound, not the
  total). The CLI exits with code 1 and warns. Do not use such a run's cost as complete. Such chunks are
  not cached, so a re-run extracts them again and records a real cost.
- **Failures.** Temporary API errors (rate limit, timeout, 5xx) are retried with backoff
  (1s, 2s, 4s...). Bad key, no quota, unknown model, bad request or any unrecognised error
  stops the whole run immediately. Invalid JSON, schema violations and refusals are retried,
  then the chunk is marked `malformed`/`refused` and the run continues.
  **Truncation** (the reply hit the output limit) is different: it is marked `truncated`
  immediately and is **not** retried, because the same input would be cut off again and every
  retry would pay for the same failure. Truncated chunks are counted in `n_truncated` and
  `failed_chunk_ids`, are never cached, and are handled identically for every strategy. The
  fix is a larger `extraction_max_output_tokens` (default 8192, sized from the schema: a dense
  chunk with 15 entities + 15 relationships needs about 1,900 tokens, 40 + 40 about 5,100).
  Tokens of every attempt, failed ones included, are counted.
- **Validation.** A relationship that names an entity not in the extracted list is dropped and
  counted in `validation_issues` (no retry). Self-relationships are dropped the same way.
- **Reproducibility.** Temperature 0 and a fixed seed are sent, but OpenAI only promises
  best-effort determinism. The cache is what guarantees an identical re-run.

## Knowledge graph (Blueprint Phase 8)

```bash
python3 -m extraction.run --config configs/extraction_dev.yaml    # extraction first
python3 -m graph.build   --config configs/extraction_dev.yaml     # -> results/graphs/<run_id>/
```

Built from the extraction results of the budget-selected chunks only, identically for
every strategy (`graph/graph_builder.py`). Writes `graph.json` (networkx node-link JSON)
and `graph_stats.json` (nodes, edges, isolated nodes, components, types).

| Decision | Choice |
|---|---|
| Node = entity | merged across chunks by name after light cleanup: lowercase, `.`/apostrophes removed, other punctuation -> space, leading "the" dropped ("The Beatles" = "Beatles" = "beatles.") |
| Type clashes | one node; majority type (ties alphabetical); `type_counts` keeps all |
| Descriptions | every distinct one kept as a list; no LLM summarisation (indexing cost = extraction only) |
| Edge = relationship | relation text as an edge attribute; MultiDiGraph |
| Provenance | every node lists its chunk_ids, every edge its chunk_id; the build fails if any cites an unselected chunk (Phase 8 checkpoint) |
| Failed chunks | contribute nothing, counted in the stats |
| Unselected chunks | not indexed at all, for every strategy (no KET-RAG keyword graph) |

Sanity check on the real 20-question HotpotQA corpus (mock extractor, so the counts only
show the wiring): at 100% budget all four strategies give the identical graph (1,823
nodes, 1,873 edges); at 5% they differ (82 to 251 nodes).

## Retrieval (Blueprint Phase 9)

```bash
python3 -m retrieval.inspect --config CONFIG --n 3   # after extraction.run and graph.build
```

Identical for every strategy (`retrieval/ppr_retrieval.py`); only the graph differs:

1. embed the question with the shared embedder;
2. seeds = the 5 entities whose card (name + descriptions) is most similar to it;
3. personalised PageRank from the seeds (damping 0.85), edges treated as undirected
   (an LLM's source/target order is arbitrary wording);
4. top 20 entities by PageRank;
5. context = `Facts:` (relationships among them) + `Passages:` (their source chunks, ordered
   by PageRank mass), at most 1,500 words. An empty graph gives an empty context.

Config: `retrieval_k_seeds` (5), `retrieval_damping` (0.85), `retrieval_top_m` (20),
`retrieval_max_context_words` (1500), `retrieval_max_facts` (30).

Check on the real 20-question HotpotQA corpus (KET-RAG, 25% budget, mock extractor): of the
40 gold chunks, KET-RAG selected 8, and retrieval returned 7 of those 8 (88%). Misses at
this budget come from selection, not retrieval - the split the benchmark is designed to show.

## Answer generation (Blueprint Phase 10)

```bash
python3 -m generation.demo --config CONFIG --n 3              # after extraction.run and graph.build
python3 -m generation.demo --config CONFIG --n 20 --dry-run   # cost estimate, nothing sent, no key needed
```

One fixed prompt for every condition (`generation/prompts.py`, version `answer-v1`):
answer **only** from the context, else `"not found"`; strict JSON reply with a one-or-two
sentence `reasoning` and the shortest exact `answer` (`yes`/`no` for yes/no questions).
Only `answer` is scored. GPT-4o-mini at temperature 0 (`generation_backend: openai`), or a
free `mock` for tests that is never reported.

- **Empty context** (possible at tiny budgets): no call, answer `not found`, status
  `skipped_empty_context`, $0.
- **Same safety rules as extraction**, reusing its code: retries for temporary errors, the
  run stops on fatal ones, truncated replies are not retried, a missing `OPENAI_API_KEY`
  stops with an error (never a silent mock), unknown token usage is never counted as $0.
- **Cache** in `cache/answers/` (and `cache/relevance/`) keyed by model, prompt version,
  question and context: at 100% budget all strategies share the same context, so those
  answers are paid once.
- **Native L4**: the same module provides the real yes/no relevance check
  (`RelevanceChecker`, version `relevance-v1`); its passages are answered by the same
  answer generator, within the same 1,500-word limit.

Config: `generation_backend` (`mock`), `generation_model` (`gpt-4o-mini`),
`generation_temperature` (0), `generation_max_output_tokens` (512), `generation_max_retries`
(3), `generation_cache_enabled`, `generation_price_input_per_1m` / `_output_per_1m`
(0.15 / 0.60). Estimated cost: about $0.007 per 20 questions per condition.

## Project structure

```
configs/        experiment YAML configs
data/raw/       mock fixtures (committed) + downloaded dev sets (git-ignored)
data/corpus/    generated corpus/chunk manifests (not committed - see .gitignore)
cache/          embeddings/extraction/judge-call caches (not committed)
src/            config, corpus (loaders), chunking, budget math, prepare_data, embeddings, embed_corpus
strategies/     selection strategies: random, ketrag, lazygraphrag (L3), fastgraphrag (F1)
native/         native LazyGraphRAG reference point (L4)
extraction/     LLM entity/relationship extraction on budget-selected chunks (Blueprint Phase 7)
graph/          knowledge-graph construction from extraction results (Blueprint Phase 8)
retrieval/      personalised-PageRank retrieval + inspection CLI (Blueprint Phase 9)
generation/     answer generation + native L4 relevance check, mock/OpenAI (Blueprint Phase 10)
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
