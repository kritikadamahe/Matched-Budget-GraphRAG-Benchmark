"""
The planner report for the logged 75-question HotpotQA benchmark: real questions with real gold
paragraphs, real answers, real measured costs, and no API call. Lets the planner be shown and
checked against the paper without spending anything. (No vector baseline was run there.)
"""

from __future__ import annotations

import json
from functools import lru_cache

from planner import costs
from planner.pilot import PilotSpec, build_report, make_config

RUN = costs.RUN
BUDGETS = [10, 25, 50, 100]
STRATEGIES = ["random", "ketrag", "lazygraphrag", "fastgraphrag"]


def _cid(s: str, b: int, shuffle: int = 0) -> str:
    return f"hotpotqa__{s}__b{b:03d}__seed42" + (f"__rs{shuffle}" if s == "random" else "")


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=1)
def report() -> dict:
    chunks = json.loads((costs.CORPUS / "chunks.json").read_text())
    qs = json.loads((costs.CORPUS / "questions.json").read_text())
    gold = {q["question_id"]: [c["chunk_id"] for c in chunks if q["question_id"] in c["is_gold_for_question_ids"]] for q in qs}
    titles = {c["chunk_id"]: c["source_title"] for c in chunks}
    questions = [{"id": q["question_id"], "question": q["question"], "answer": q["answer"], "gold_chunks": gold[q["question_id"]],
                  "gold_titles": [titles[c] for c in gold[q["question_id"]]], "hops": len(gold[q["question_id"]])} for q in qs]

    scores = {}
    for s in STRATEGIES:
        for b in BUDGETS:
            per_shuffle = []
            for sh in ([0, 1, 2] if s == "random" else [0]):
                cdir = RUN / "conditions" / _cid(s, b, sh)
                sel = set(json.loads((cdir / "extraction" / "selection.json").read_text())["selected_chunk_ids"])
                passages = {p["question_id"]: set(p["retrieval"]["passage_chunk_ids"]) for p in _jsonl(cdir / "predictions.jsonl")}
                rows = {}
                for r in _jsonl(cdir / "scored.jsonl"):
                    g = gold[r["question_id"]]
                    rows[r["question_id"]] = {"answer": r["predicted_answer"], "f1": r["f1"] or 0.0, "em": r["em"] or 0.0,   # failed call = wrong
                                              "selected": sum(c in sel for c in g) / len(g),
                                              "retrieved": sum(c in passages.get(r["question_id"], ()) for c in g) / len(g)}
                per_shuffle.append(rows)
            # Random: the mean of its shuffles per question, as in the paper
            scores[(s, b)] = {q: {"answer": per_shuffle[0][q]["answer"],
                                  **{m: sum(p[q][m] for p in per_shuffle) / len(per_shuffle)
                                     for m in ("f1", "em", "selected", "retrieved")}} for q in per_shuffle[0]}

    base = json.loads((RUN / "conditions" / _cid("ketrag", 100) / "condition_result.json").read_text())["config"]
    spec = PilotSpec(n_questions=len(questions), strategies=STRATEGIES, primary="ketrag", budgets=BUDGETS,
                     price_in=base["extraction_price_input_per_1m"], price_out=base["extraction_price_output_per_1m"])
    full = json.loads((RUN / "conditions" / _cid("random", 100) / "extraction" / "run_summary.json").read_text())
    return build_report(
        kind="sample", title="Sample: HotpotQA benchmark (75 questions, 759 chunks)", spec=spec, cfg=make_config(spec),
        questions=questions, strategies=STRATEGIES, budgets=BUDGETS, scores=scores, vector_rows=None,
        all_words=[len(c["text"].split()) for c in chunks], n_docs=len({c["source_doc_id"] for c in chunks}),
        sample_n=len(chunks), cal=costs.benchmark_calibration(),
        pilot_spend={"usd": None, "measured_full_extraction_usd": full["cost_if_uncached_usd"],
                     "note": "Logged benchmark run: questions and gold paragraphs from HotpotQA, no pilot calls."},
        extra_caveats=["Sample report: the 75 questions are HotpotQA's own, and the whole corpus was used (no sampling). "
                       f"Measured cost of extracting all 759 chunks in the real run: ${full['cost_if_uncached_usd']:.2f}."])
