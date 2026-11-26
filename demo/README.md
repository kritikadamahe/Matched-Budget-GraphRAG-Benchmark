# Presentation demo

A local web UI that walks through the benchmark using the real pipeline code and the
saved results of the 75-question HotpotQA run. Built for screen-recording a walkthrough.

```bash
cd Matched-Budget-GraphRAG-Benchmark
./demo/run.sh                      # or: .venv/bin/python -m demo.server
# open http://127.0.0.1:8765
```

Startup takes ~10 s (it re-ranks all chunks with the four selectors and loads the embedder).
No API key and no money needed: every benchmark question is answered from `cache/answers`, so
each replay returns exactly the answer logged in the run (the UI confirms this with a green tick).

`--allow-live` also answers **typed-in** questions with a real LLM call (key from `.env`,
about $0.002 per question). Without it, a typed question shows the retrieval steps only.

## Tabs

| Tab | Shows |
|---|---|
| 1 · The idea | The problem, the fixed pipeline with the selector as the only variable, the four selectors, headline numbers |
| 2 · Budget explorer | All 759 chunks as squares, side by side for the four selectors. Drag the budget slider or press *Sweep*; gold evidence is outlined, and the share of gold selected updates live. Pick a question to see only its gold paragraphs |
| 3 · Ask the pipeline | Pick a question, selector and budget, then *Run pipeline*: selection (were the gold paragraphs extracted?), graph size, PageRank seeds and subgraph, the exact context sent to the LLM (gold passages highlighted), and the answer with F1 against gold |
| 4 · Compare | One question across every selector × budget, from the logged run. Click any cell to replay it in tab 3. Below it, mean F1 with 95% CIs over all 75 questions |
| 5 · Results | Selection study (500 + 500 questions, with oracle) and end-to-end F1 curves, plus the takeaways |
| Budget planner | The research turned into a tool: upload your own documents and get a recommended extraction budget (see below) |


## Budget planner

**Who it is for:** a team that wants GraphRAG over its own documents (policies, manuals, contracts,
research notes, a support knowledge base) and has to decide whether indexing pays off, and how much
to index, *before* paying to extract everything.

**The question it answers:** *"What is the smallest share of our corpus we need to send to the LLM
extractor to keep, say, 90% of full-extraction answer quality, and what will that cost?"*

| Step | What happens | Cost |
|---|---|---|
| 1. Documents | Upload PDF / Markdown / text / HTML / CSV / JSON. Chunked exactly like the benchmark. | free |
| 2. Model & cost | Full-corpus indexing cost at 10-100% budget, from a cost model **calibrated on 759 real extraction calls**. The usual rule of thumb (1.33 tokens/word, 250 output tokens) underestimates by about 3.4× | free |
| 3. Pilot | On a sample of the corpus: the LLM writes test questions with known source chunks (single and two-hop), the sample is extracted once, then every selector × budget is answered and scored, plus a **vector-RAG baseline** with no extraction at all. Hard spending cap; everything cached | paid, shown before running |
| 4. Report | Recommended budget, projected cost and savings, quality retained, vector-RAG verdict, budget table with 95% CIs, evidence recall, example questions, method and caveats. Download as Markdown or JSON, or print to PDF | free |

The quality-target slider re-derives the recommendation live. *Use the sample corpus* opens a complete
report built from the logged benchmark (real HotpotQA questions, real costs), so the planner can be
shown with no API key. Its numbers match the paper: F1 0.068 → 0.609, and no budget below 100% keeps 90%.

```bash
./demo/run.sh --allow-live      # enable paid pilots (and typed-in questions); key from .env
./demo/run.sh --planner-mock    # offline wiring test: mock extractor/answerer, results meaningless
```

Code: `planner/` (ingest, costs, questions, pilot, sample, report_md) and `demo/planner_api.py`.
Uploaded corpora and reports are saved under `results/planner/<corpus_id>/`.

Deep link straight into a run: `http://127.0.0.1:8765/#ask/<question_id>/<strategy>/<budget>`, e.g.
`#ask/5ac3a76e554299741d48a2be/ketrag/10`.

## Suggested recording script (~6 min)

Record at 1440×900 or 1920×1080, browser zoom 100-110%, and hide the bookmarks bar.

1. **The idea (45 s).** Read the headline question. Point at the pipeline: only the orange
   selector box changes between conditions. Introduce the four selectors, then note that none of
   them sees the questions or uses an LLM.
2. **Budget explorer (1 min).** At 10%, show that each selector picks a different set of
   squares, yet all four keep about the same share of gold evidence. Press *Sweep 1→100%* and
   let the bars rise together. Pick the "With or Without You" question to show its two gold
   paragraphs.
3. **Ask the pipeline (2 min).** The ★ questions are the interesting ones.
   - *With or Without You*, **KET-RAG, 10%**: one gold paragraph was extracted. The graph links
     the song to U2 and the answer is correct (F1 1.00).
   - Same question, **Random, 10%**: the other gold paragraph made it in, but the model answers
     "Jai McDowall", which is wrong. Which chunks get extracted changes the answer.
   - Same question, **100%**: every selector is identical, and so are the answers (sanity check).
4. **Compare (1 min).** Show the U2 grid: KET-RAG is right from 10%, the others from 25%, and
   Random fails until 100%. Then scroll to the 75-question means: at each budget the strategies
   are close together and the CIs overlap. One question is an anecdote; the table is the evidence.
5. **Results (1 min).** On the selection study, the three selectors hug the Random line while
   the oracle sits far above it. On the F1 curve, budget drives quality (0.07 → 0.61) and the
   selectors don't separate. End on the black takeaway banner.

Other ★ questions: *Roseanne star in Scream 2* (Entity centrality and one of the three Random
shuffles right at 10%, KET-RAG only at 50%), *John André* (no selector at 10%, Entity centrality from 25%), and
*Halldór Laxness / Timothy Leary* (only Entity centrality at the low budgets).
