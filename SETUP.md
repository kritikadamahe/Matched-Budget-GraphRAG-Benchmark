# Setup on a new machine

Tested on Python 3.14 (3.11+ should work). The `cache/`, `results/`, `data/raw` and `data/corpus` folders and `.env` are git-ignored, so they are rebuilt locally.

## 1. Get the code and install

```bash
git clone https://github.com/kritikadamahe/Matched-Budget-GraphRAG-Benchmark
cd Matched-Budget-GraphRAG-Benchmark
git checkout <branch-name>        # the branch holding the free-tier changes

python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# local embedding model for KET-RAG faithful mode (CPU-only is enough)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

## 2. Check everything works (no key needed)

```bash
python -m pytest -q                                   # expect all tests to pass
python -m experiments.run --config configs/pilot_hotpotqa_mock.yaml
```

The mock run uses fake backends, so its scores are never reportable.

## 3. Download the real datasets (free)

```bash
python3 -m src.prepare_data --config configs/hotpotqa_dev.yaml
python3 -m src.prepare_data --config configs/musique_dev.yaml
```

## 4. Add your API key

```bash
cp .env.example .env
```

Edit `.env` (never commit it, never paste it into chat):

**Option A: Gemini free tier** (pipeline checks only, daily caps are too small for the full run)
```
OPENAI_API_KEY=<your Google AI Studio key>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MIN_INTERVAL_S=7
```

**Option B: OpenAI (paid)**
```
OPENAI_API_KEY=sk-...
```
Leave `OPENAI_BASE_URL` and `LLM_MIN_INTERVAL_S` out. Changing the model changes the cache keys, so a cache built with one provider is not reused by another.

**Speed (paid keys):** add `LLM_WORKERS=8` to `.env` to make 8 API calls at once. Results are identical to a sequential run. Leave `LLM_MIN_INTERVAL_S` out when using it. To skip the LLM judge (EM/F1 only, about half the calls), add `--no-judge` to the run command.

## 5. Run the pilot

```bash
python -m experiments.run --config configs/pilot_hotpotqa_free.yaml --dry-run           # free estimate
python -m experiments.run --config configs/pilot_hotpotqa_free.yaml --confirm-paid --max-total-usd 0.50
```

For OpenAI use `configs/pilot_hotpotqa.yaml` instead. Expect 4 conditions. Two checks:
1. Random and KET-RAG score identically at 100% budget.
2. Re-running makes no new API calls (the cache works).

Pilot scores come from 20 questions and are not results to report.

## 6. Full benchmark (later)

Needs a full-run config (model, question count, budgets, seeds), which doesn't exist yet. Always run `--dry-run` first and set `--max-total-usd` as a cap.

## Troubleshooting

- `RateLimitError` / `insufficient_quota`: free-tier daily limit or no credit. Raise `LLM_MIN_INTERVAL_S` or wait for the quota reset.
- `seed` rejected by the provider: the code skips it automatically when `OPENAI_BASE_URL` is set.
- Retired model name: edit `extraction_model`, `generation_model`, `judge_model` in the config.
