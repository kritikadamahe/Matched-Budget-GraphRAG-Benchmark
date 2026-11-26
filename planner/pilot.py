"""
The pilot: the matched-budget benchmark, run on a sample of the user's own corpus.

    sample documents (up to `sample_chunks` chunks)  ->  generate test questions (known sources)
    -> rank the sample with each selector            ->  extract the WHOLE sample once (cached)
    -> for each selector x budget: graph from the top-k extractions -> PageRank retrieval -> answer -> EM/F1
    -> vector-RAG baseline (no extraction at all) on the same questions
    -> project indexing cost to the full corpus with token usage measured in the pilot
    -> recommend the smallest budget that keeps `target` of full-extraction quality

Extracting the whole sample once is enough for every condition: a smaller budget is a subset of
the extracted chunks (the same property the benchmark relies on), so no chunk is paid for twice.
A hard spending cap covers every call; the pilot stops (keeping cached work) when it would pass it.
"""

from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation.bootstrap import bootstrap_ci
from evaluation.metrics import best_scores
from extraction import build_extractor
from extraction.cache import ExtractionCache
from extraction.pipeline import run_extraction
from generation.answerer import build_answerer
from generation.cache import QueryCache
from generation.llm_client import ChatJSONClient
from generation.prompts import NOT_FOUND
from graph.graph_builder import build_graph
from planner import costs
from planner.questions import QuestionGenerator, generate
from retrieval.ppr_retrieval import GraphRetriever
from src.chunking import chunk_corpus
from src.config import load_config
from src.embeddings import embedder_from_config
from src.parallel import llm_workers, map_ordered
from strategies import build_strategy

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "benchmark_75q_hotpotqa.yaml"
NAMES = {"random": "Random", "ketrag": "KET-RAG-style", "lazygraphrag": "Topic coverage", "fastgraphrag": "Entity centrality"}


class PilotBudgetExceeded(RuntimeError):
    pass


@dataclass
class PilotSpec:
    n_questions: int = 30
    sample_chunks: int = 200
    strategies: list[str] = field(default_factory=lambda: ["random", "ketrag"])
    primary: str = "ketrag"
    budgets: list[int] = field(default_factory=lambda: [10, 25, 50, 75, 100])
    target: float = 0.90
    max_usd: float = 1.00
    model: str = ""                   # "" = the model in the base config
    price_in: float = 1.22
    price_out: float = 1.79
    backend: str = "openai"           # "mock" only for offline wiring tests: never a real result
    seed: int = 42


def make_config(spec: PilotSpec):
    base = load_config(BASE_CONFIG)
    model = spec.model or base.extraction_model
    same = model == base.extraction_model
    return base.model_copy(update={
        "extraction_backend": spec.backend, "generation_backend": spec.backend,
        "extraction_model": model, "generation_model": model,
        "extraction_price_input_per_1m": spec.price_in, "extraction_price_output_per_1m": spec.price_out,
        "generation_price_input_per_1m": spec.price_in, "generation_price_output_per_1m": spec.price_out,
        # reasoning_effort "none" is a Cerebras/qwen setting; other providers may reject it
        "extraction_reasoning_effort": base.extraction_reasoning_effort if same else None,
        "generation_reasoning_effort": base.generation_reasoning_effort if same else None,
        "seed": spec.seed,
    })


def chunk_documents(documents: dict) -> list:
    return chunk_corpus(documents, 250, 40)


def sample_chunks(chunks: list, cap: int, seed: int = 42) -> list:
    """Whole documents in random order until `cap` chunks (a very long document contributes its
    first chunks only). Keeps documents intact so two-hop questions and entity links survive."""
    if len(chunks) <= cap:
        return list(chunks)
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.source_doc_id, []).append(c)
    docs = list(by_doc)
    random.Random(seed).shuffle(docs)
    out = []
    for d in docs:
        out.extend(by_doc[d][: cap - len(out)])
        if len(out) >= cap:
            break
    order = {c.chunk_id: i for i, c in enumerate(chunks)}
    return sorted(out, key=lambda c: order[c.chunk_id])


def estimate(spec: PilotSpec, chunks: list) -> dict:
    """Upper-bound pilot cost (ignores the cache), from the benchmark calibration."""
    cal = costs.benchmark_calibration()
    sample = sample_chunks(chunks, spec.sample_chunks, spec.seed)
    words = [len(c.text.split()) for c in sample]
    ext = costs.extraction_table(words, [100], cal, spec.price_in, spec.price_out)[0]["usd"]
    avg = float(np.mean(words)) if words else 0
    n_gen = int(spec.n_questions * 1.5) + 2
    qgen = costs.usd(n_gen * (1.5 * avg * 1.33 + 180), n_gen * 60, spec.price_in, spec.price_out)
    n_answers = spec.n_questions * (len(spec.strategies) * (len(spec.budgets) - 1) + 2)
    ans = costs.query_cost(n_answers, cal, spec.price_in, spec.price_out)
    total = ext + qgen + ans
    return {"sample_chunks": len(sample), "extraction_usd": round(ext, 4), "questions_usd": round(qgen, 4),
            "answers_usd": round(ans, 4), "n_answer_calls": n_answers, "total_usd": round(total, 4),
            "within_cap": total <= spec.max_usd}


class _Spend:
    def __init__(self, cap: float):
        self.cap, self.usd = cap, 0.0

    def add(self, usd: float | None):
        self.usd += usd or 0.0

    def check(self, upcoming: float, what: str):
        if self.usd + upcoming > self.cap:
            raise PilotBudgetExceeded(f"stopping before {what}: spent ${self.usd:.3f}, next step needs about "
                                      f"${upcoming:.3f}, cap is ${self.cap:.2f}. Raise the cap and re-run; "
                                      f"finished calls are cached and will not be paid for again.")


def _vector_context(question: str, chunk_ids: list, chunk_vecs: np.ndarray, chunk_text: dict, embedder,
                    max_words: int) -> tuple[str, list]:
    sims = chunk_vecs @ np.asarray(embedder.transform([question]))[0]
    used, parts, budget = [], [], max_words
    for i in np.argsort(-sims):
        text = chunk_text[chunk_ids[i]]
        n = len(text.split()) + 1
        if n <= budget:
            used.append(chunk_ids[i]); parts.append(f"[{len(parts) + 1}] {text}"); budget -= n
        if budget < 40:
            break
    return "Passages:\n" + "\n".join(parts), used


def recommend(curve: dict, budgets: list[int], target: float, cost_rows: list[dict],
              vector: dict | None) -> dict:
    """Smallest budget whose mean F1 reaches target x F1 at 100%, linearly interpolated between the
    measured budgets and rounded UP to a multiple of 5%. Costs from the full-corpus projection."""
    full = curve[100]["mean"]
    cost_at = {r["budget_pct"]: r["usd"] for r in cost_rows}
    full_usd = cost_at[100]
    out = {"full_f1": full, "target": target, "full_usd": full_usd, "verdicts": []}
    if full < 0.15:
        out.update(status="inconclusive", budget_pct=100, usd=full_usd)
        out["verdicts"].append("Even full extraction answers few pilot questions (F1 < 0.15). The questions may not suit "
                               "this corpus, or the corpus may need a different retrieval setup. Do not size a budget from this pilot.")
        return out
    goal = target * full
    prev_b, prev_f = 0, 0.0
    chosen = 100
    for b in budgets:
        f = curve[b]["mean"]
        if f >= goal:
            interp = prev_b + (goal - prev_f) / (f - prev_f) * (b - prev_b) if f > prev_f else b
            chosen = min(b, max(5, int(np.ceil(interp / 5) * 5)))
            break
        prev_b, prev_f = b, f
    # cost of an interpolated budget: linear in the number of chunks
    usd = round(full_usd * chosen / 100, 4)
    measured_f = np.interp(chosen, [0] + budgets, [0.0] + [curve[b]["mean"] for b in budgets])
    lo = max([b for b in budgets if b < chosen], default=0)
    out.update(status="ok", budget_pct=chosen, usd=usd, f1=round(float(measured_f), 4), interpolated=chosen not in budgets,
               retained=round(float(measured_f / full), 3) if full else None,
               savings_usd=round(full_usd - usd, 4), savings_pct=round(100 - chosen, 1))
    v = out["verdicts"]
    v.append(f"Extracting {chosen}% of the corpus is projected to keep about {round(100 * measured_f / full)}% of the "
             f"answer quality of full extraction, for ${usd:,.2f} instead of ${full_usd:,.2f}.")
    if chosen not in budgets:
        v.append(f"{chosen}% is interpolated between the measured budgets {lo}% and {min(b for b in budgets if b > chosen)}%: "
                 f"no measured budget below {min(b for b in budgets if b > chosen)}% reached the target. Confirm it with a pilot at {chosen}% before committing.")
    if vector is not None:
        if vector["mean"] >= full - 0.02:
            v.append(f"Plain vector RAG (no LLM extraction, F1 {vector['mean']:.2f}) matches full GraphRAG "
                     f"(F1 {full:.2f}) on these questions: GraphRAG indexing is hard to justify for this corpus.")
        elif vector["mean"] >= measured_f:
            v.append(f"At {chosen}%, GraphRAG (F1 {measured_f:.2f}) does not beat plain vector RAG (F1 {vector['mean']:.2f}). "
                     f"GraphRAG pays off only near full extraction (F1 {full:.2f}).")
        else:
            v.append(f"GraphRAG at {chosen}% beats plain vector RAG (F1 {measured_f:.2f} vs {vector['mean']:.2f}).")
    return out


def run(spec: PilotSpec, documents: dict, progress=lambda stage, done, total, msg="": None) -> dict:
    started = time.perf_counter()
    cfg = make_config(spec)
    spend = _Spend(spec.max_usd)
    all_chunks = chunk_documents(documents)
    sample = sample_chunks(all_chunks, spec.sample_chunks, spec.seed)
    chunk_text = {c.chunk_id: c.text for c in sample}
    if len(sample) < 20:
        raise ValueError(f"only {len(sample)} chunks: upload more text (a pilot needs at least 20 chunks)")
    budgets = sorted(set(spec.budgets) | {100})
    strategies = list(dict.fromkeys(spec.strategies + [spec.primary]))
    est = estimate(spec, all_chunks)

    # 1. questions --------------------------------------------------------------------------
    progress("questions", 0, spec.n_questions, "writing test questions from your documents")
    client = None
    if spec.backend == "openai":
        client = ChatJSONClient(model=cfg.generation_model, temperature=0.0, max_output_tokens=400,
                                max_retries=cfg.generation_max_retries, timeout_s=cfg.generation_request_timeout_s,
                                price_input_per_1m=spec.price_in, price_output_per_1m=spec.price_out,
                                reasoning_effort=cfg.generation_reasoning_effort)
    spend.check(est["questions_usd"], "question generation")
    gen = QuestionGenerator(spec.backend, client, QueryCache(ROOT / "cache" / "answers"), cfg.generation_model)
    questions = generate(sample, spec.n_questions, gen, lambda d, t: progress("questions", d, t))
    spend.add(gen.usage["usd"])
    if len(questions) < 10:
        raise ValueError(f"only {len(questions)} usable questions could be written; upload more (or more factual) text")

    # 2. rankings (selectors never see questions or gold) -----------------------------------
    progress("ranking", 0, len(strategies), "ranking chunks with each selector")
    blind = [dataclasses.replace(c, is_gold_for_question_ids=[]) for c in sample]
    rankings = {}
    for i, s in enumerate(strategies):
        rankings[s] = list(build_strategy(cfg.model_copy(update={"strategy": s, "random_shuffle_seed": 0})).rank(blind))
        progress("ranking", i + 1, len(strategies))

    # 3. extraction of the whole sample, once ------------------------------------------------
    progress("extraction", 0, len(sample), "LLM entity and relation extraction")
    spend.check(est["extraction_usd"] * 0.5, "extraction")          # the cap is enforced per call below
    extractor = build_extractor(cfg)
    run_ = run_extraction(sample, rankings[strategies[0]], 1.0, extractor, ExtractionCache(ROOT / "cache" / "extractions"),
                          max_cost_usd=spec.max_usd - spend.usd,
                          progress=lambda d, t: progress("extraction", d, t))
    spend.add(run_.summary.get("cost_spent_usd") or 0.0)
    by_id = {r.chunk_id: r for r in run_.results}

    # 4. answer every question in every condition --------------------------------------------
    embedder = embedder_from_config(cfg)
    answerer = build_answerer(cfg)
    n = len(sample)
    conditions = [(s, b) for s in strategies for b in budgets]
    graphs: dict[frozenset, GraphRetriever] = {}
    scores: dict[tuple, dict] = {}
    q_cost = costs.query_cost(len(questions), costs.benchmark_calibration(), spec.price_in, spec.price_out)

    def answer_all(contexts: list[str]) -> list:
        return map_ordered(lambda qc: answerer.answer(qc[0]["question"], qc[1]), list(zip(questions, contexts)), llm_workers())

    for ci, (s, b) in enumerate(conditions):
        progress("answering", ci, len(conditions) + 1, f"{NAMES[s]} at {b}%")
        k = max(1, int(b / 100 * n + 0.5))
        selected = rankings[s][:k]
        key = frozenset(selected)
        if key not in graphs:                                 # 100% is the same graph for every selector
            graph, _ = build_graph([by_id[c] for c in selected if c in by_id])
            graphs[key] = GraphRetriever(graph, chunk_text, embedder, k_seeds=cfg.retrieval_k_seeds,
                                         damping=cfg.retrieval_damping, top_m=cfg.retrieval_top_m,
                                         max_context_words=cfg.retrieval_max_context_words, max_facts=cfg.retrieval_max_facts)
        retr = [graphs[key].retrieve(q["question"]) for q in questions]
        spend.check(q_cost, f"answering ({NAMES[s]}, {b}%)")
        results = answer_all([r.context for r in retr])
        sel = set(selected)
        rows = {}
        for q, r, a in zip(questions, retr, results):
            spend.add(0.0 if a.cache_hit else a.usage.cost_usd)
            em, f1 = best_scores(a.answer, [q["answer"]])
            rows[q["id"]] = {"answer": a.answer, "f1": f1, "em": em,
                             "selected": sum(c in sel for c in q["gold_chunks"]) / len(q["gold_chunks"]),
                             "retrieved": sum(c in r.chunk_ids for c in q["gold_chunks"]) / len(q["gold_chunks"])}
        scores[(s, b)] = rows

    # 5. vector-RAG baseline (no LLM indexing) -----------------------------------------------
    progress("answering", len(conditions), len(conditions) + 1, "vector-RAG baseline")
    ids = [c.chunk_id for c in sample]
    vecs = np.asarray(embedder.fit_transform([chunk_text[i] for i in ids]))
    vctx = [_vector_context(q["question"], ids, vecs, chunk_text, embedder, cfg.retrieval_max_context_words) for q in questions]
    spend.check(q_cost, "the vector-RAG baseline")
    vres = answer_all([c for c, _ in vctx])
    vector_rows = {}
    for q, (_, used), a in zip(questions, vctx, vres):
        spend.add(0.0 if a.cache_hit else a.usage.cost_usd)
        em, f1 = best_scores(a.answer, [q["answer"]])
        vector_rows[q["id"]] = {"answer": a.answer, "f1": f1, "em": em,
                                "retrieved": sum(c in used for c in q["gold_chunks"]) / len(q["gold_chunks"])}

    # 6. projection, calibrated on the pilot's own extraction calls --------------------------
    measured = [(len(chunk_text[r.chunk_id].split()), r.usage.input_tokens, r.usage.output_tokens)
                for r in run_.results if r.usage is not None and r.usage.input_tokens]
    ans_usage = [a.usage for a in answerer.results if a.usage is not None and a.usage.input_tokens]
    cal = (costs.from_measurements([m[0] for m in measured], [m[1] for m in measured], [m[2] for m in measured],
                                   np.mean([u.input_tokens for u in ans_usage]) if ans_usage else costs.benchmark_calibration().query_in,
                                   np.mean([u.output_tokens for u in ans_usage]) if ans_usage else costs.benchmark_calibration().query_out,
                                   f"{len(measured)} extraction calls on your documents (this pilot)")
           if len(measured) >= 10 else costs.benchmark_calibration())
    progress("report", 0, 1, "building the report")
    report = build_report(
        kind="pilot" if spec.backend != "mock" else "simulated",
        title="Pilot on your documents", spec=spec, cfg=cfg, questions=questions, strategies=strategies,
        budgets=budgets, scores=scores, vector_rows=vector_rows,
        all_words=[len(c.text.split()) for c in all_chunks], n_docs=len(documents), sample_n=len(sample), cal=cal,
        pilot_spend={"usd": round(spend.usd, 4), "estimate_usd": est["total_usd"], "cap_usd": spec.max_usd,
                     "extraction_calls": run_.summary.get("n_extractor_calls"), "answer_calls": len(answerer.results),
                     "question_calls": gen.usage["calls"], "seconds": round(time.perf_counter() - started, 1)})
    progress("report", 1, 1, "done")
    return report


def summarise(values: list[float]) -> dict:
    ci = bootstrap_ci(values, n_resamples=2000)
    return {"mean": ci["mean"], "low": ci["low"], "high": ci["high"], "n": ci["n"]}


def build_report(*, kind, title, spec, cfg, questions, strategies, budgets, scores, vector_rows, all_words,
                 n_docs, sample_n, cal, pilot_spend, extra_caveats=()) -> dict:
    """One report schema for pilots and for the benchmark sample."""
    qids = [q["id"] for q in questions]
    quality = {s: {b: summarise([scores[(s, b)][q]["f1"] for q in qids]) for b in budgets} for s in strategies}
    em = {s: {b: round(float(np.mean([scores[(s, b)][q]["em"] for q in qids])), 4) for b in budgets} for s in strategies}
    evidence = {s: {b: {"selected": round(float(np.mean([scores[(s, b)][q]["selected"] for q in qids])), 4),
                        "retrieved": round(float(np.mean([scores[(s, b)][q]["retrieved"] for q in qids])), 4)}
                    for b in budgets} for s in strategies}
    vector = summarise([vector_rows[q]["f1"] for q in qids]) if vector_rows else None
    cost_rows = costs.extraction_table(all_words, budgets, cal, spec.price_in, spec.price_out)
    rec = recommend(quality[spec.primary], budgets, spec.target, cost_rows, vector)
    sd = float(np.std([scores[(spec.primary, 100)][q]["f1"] - scores[(spec.primary, budgets[0])][q]["f1"] for q in qids]))
    mde = round(2.8 * sd / np.sqrt(len(qids)), 3) if qids else None
    examples = []
    for q in questions[:12]:
        examples.append({"question": q["question"], "answer": q["answer"], "gold_titles": q.get("gold_titles", []),
                         "hops": q.get("hops"),
                         "by_budget": {b: {"answer": scores[(spec.primary, b)][q["id"]]["answer"],
                                           "f1": scores[(spec.primary, b)][q["id"]]["f1"]} for b in budgets},
                         "vector": ({"answer": vector_rows[q["id"]]["answer"], "f1": vector_rows[q["id"]]["f1"]}
                                    if vector_rows else None)})
    caveats = [
        f"Quality is measured on {len(qids)} test questions; differences smaller than about {mde} F1 between two "
        f"budgets cannot be detected reliably (80% power)." if mde is not None else "",
        f"The pilot used {sample_n} of {len(all_words)} chunks. Costs are projected to the full corpus with "
        f"token usage measured on {cal.source}; quality is assumed to scale with the extracted share.",
        "The matched-budget study behind this tool found that on pooled multi-hop QA corpora the question-blind "
        "selectors keep about as much evidence as random selection: the budget matters far more than the selector.",
        *extra_caveats,
    ]
    if kind == "simulated":
        caveats.insert(0, "SIMULATED: mock extractor and answerer. These numbers test the wiring only and mean nothing.")
    return {
        "kind": kind, "title": title, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": cfg.extraction_model, "price_in": spec.price_in, "price_out": spec.price_out,
        "corpus": {"n_documents": n_docs, "n_chunks": len(all_words), "n_words": int(sum(all_words)), "sample_chunks": sample_n},
        "n_questions": len(qids), "strategies": strategies, "names": {s: NAMES[s] for s in strategies},
        "primary": spec.primary, "budgets": budgets, "target": spec.target,
        "quality": quality, "em": em, "evidence": evidence, "vector": vector,
        "cost_projection": cost_rows, "calibration": costs.calibration_dict(cal),
        "recommendation": rec, "mde_f1": mde, "pilot_spend": pilot_spend, "examples": examples,
        "caveats": [c for c in caveats if c],
    }
