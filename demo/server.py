"""
Presentation demo: a local web UI over the 75-question HotpotQA benchmark.

    .venv/bin/python -m demo.server            # then open http://127.0.0.1:8765
    .venv/bin/python -m demo.server --port 9000 --allow-live

The "Budget planner" tab (planner/, demo/planner_api.py) sizes an extraction budget for the
user's own documents: free cost projection, and a paid pilot with --allow-live.

It drives the REAL pipeline modules (strategies, graph, PageRank retriever, answer
generator) on the graphs saved by the benchmark run. For the 75 benchmark questions
every answer is already in cache/answers, so replaying them costs nothing and returns
exactly what the run logged. A typed-in question needs a new LLM call: it is answered
only with --allow-live (uses the key in .env); otherwise the UI shows retrieval only.

Standard library only (http.server); no extra dependencies.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import best_scores                      # noqa: E402
from extraction.base_extractor import ExtractionAbort           # noqa: E402
from generation.answerer import build_answerer                  # noqa: E402
from graph.graph_builder import load_graph                      # noqa: E402
from retrieval.ppr_retrieval import retriever_from_config       # noqa: E402
from src.config import ExperimentConfig                         # noqa: E402
from src.prepare_data import build                              # noqa: E402
from strategies import build_strategy                           # noqa: E402
from demo import planner_api                                    # noqa: E402

RUN = ROOT / "results" / "experiments" / "benchmark_75q_hotpotqa"
SELECTION_STUDY = ROOT / "results" / "selection_study" / "hotpotqa_musique_500q"
STATIC = Path(__file__).resolve().parent / "static"
STRATEGIES = ["random", "ketrag", "lazygraphrag", "fastgraphrag"]
BUDGETS = [10, 25, 50, 100]
SHUFFLES = [0, 1, 2]
# Questions where the strategies or budgets disagree: good to show on screen.
FEATURED = [
    "5ac3a76e554299741d48a2be",   # With or Without You: KET-RAG right at 10%, the rest only at 25%+
    "5a7557d75542992d0ec05f68",   # Roseanne star in Scream 2
    "5ab97d0a5542996be202051e",   # John Andre: nobody at 10%, entity centrality from 25%
    "5ac54c435542993e66e822d1",   # Cale Gundy's brother
    "5a877dd65542993e715abf79",   # Laxness / Leary: only entity centrality at low budgets
    "5a7153d05542994082a3e7dc",   # Nusretiye Mosque
]

LOCK = threading.Lock()
ALLOW_LIVE = False


def condition_id(strategy: str, budget: int, shuffle: int = 0) -> str:
    cid = f"hotpotqa__{strategy}__b{budget:03d}__seed42"
    return cid + (f"__rs{shuffle}" if strategy == "random" else "")


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------- loaded once
BASE_CFG = ExperimentConfig(**_read_json(RUN / "conditions" / condition_id("ketrag", 50) / "condition_result.json")["config"])
CHUNKS, QUESTIONS, _ = build(BASE_CFG)
CHUNK_BY_ID = {c.chunk_id: c for c in CHUNKS}
QUESTION_BY_ID = {q["question_id"]: q for q in QUESTIONS}
GOLD_CHUNKS = {q["question_id"]: [c.chunk_id for c in CHUNKS if q["question_id"] in c.is_gold_for_question_ids]
               for q in QUESTIONS}


def cfg_for(cid: str) -> ExperimentConfig:
    return ExperimentConfig(**_read_json(RUN / "conditions" / cid / "condition_result.json")["config"])


def compute_rankings() -> dict[str, list[str]]:
    """Full ranking of all chunks per strategy (budget-independent), as the runner does it.
    Selectors see the chunks only: gold labels are stripped first."""
    blind = [dataclasses.replace(c, is_gold_for_question_ids=[]) for c in CHUNKS]
    out = {}
    for s in STRATEGIES:
        for shuffle in (SHUFFLES if s == "random" else [None]):
            cfg = BASE_CFG.model_copy(update={"strategy": s, "random_shuffle_seed": shuffle})
            key = s if shuffle is None else f"random_rs{shuffle}"
            out[key] = list(build_strategy(cfg).rank(blind))
    return out


@lru_cache(maxsize=None)
def retriever_for(cid: str):
    cfg = cfg_for(cid)
    graph = load_graph(RUN / "conditions" / cid / "graph" / "graph.json")
    return cfg, graph, retriever_from_config(cfg, graph, CHUNKS)


@lru_cache(maxsize=None)
def logged(cid: str) -> dict[str, dict]:
    return {r["question_id"]: r for r in _read_jsonl(RUN / "conditions" / cid / "scored.jsonl")}


def score_grid() -> dict:
    """question_id -> condition_id -> {f1, em, answer}: every logged answer, for the comparison view."""
    grid: dict[str, dict] = {}
    for s in STRATEGIES:
        for b in BUDGETS:
            for shuffle in (SHUFFLES if s == "random" else [0]):
                cid = condition_id(s, b, shuffle)
                for qid, r in logged(cid).items():
                    grid.setdefault(qid, {})[cid] = {"f1": r["f1"], "em": r["em"], "answer": r["predicted_answer"]}
    return grid


def meta() -> dict:
    graph_stats = {}
    for s in STRATEGIES:
        for b in BUDGETS:
            cid = condition_id(s, b)
            st = _read_json(RUN / "conditions" / cid / "graph" / "graph_stats.json")
            graph_stats[cid] = {k: st[k] for k in ("n_selected", "n_nodes", "n_edges")}
    featured = [q for q in FEATURED if q in QUESTION_BY_ID]
    order = featured + [q["question_id"] for q in QUESTIONS if q["question_id"] not in featured]
    return {
        "strategies": STRATEGIES, "budgets": BUDGETS, "shuffles": SHUFFLES, "allow_live": ALLOW_LIVE,
        "model": BASE_CFG.generation_model, "embedding_model": BASE_CFG.embedding_model,
        "n_chunks": len(CHUNKS),
        "chunks": [{"id": c.chunk_id, "title": c.source_title, "gold": c.is_gold_for_question_ids} for c in CHUNKS],
        "questions": [{"id": qid, "question": QUESTION_BY_ID[qid]["question"], "answer": QUESTION_BY_ID[qid]["answer"],
                       "featured": qid in featured, "gold_chunks": GOLD_CHUNKS[qid]} for qid in order],
        "rankings": RANKINGS,
        "graph_stats": graph_stats,
        "scores": score_grid(),
        "e2e": _read_csv(RUN / "analysis" / "real" / "tables" / "performance.csv"),
        "e2e_evidence": _read_csv(RUN / "analysis" / "real" / "tables" / "evidence.csv"),
        "selection_study": _read_csv(SELECTION_STUDY / "performance.csv"),
    }


def ask(body: dict) -> dict:
    strategy, budget = body["strategy"], int(body["budget"])
    cid = condition_id(strategy, budget, int(body.get("shuffle", 0)))
    qid = body.get("question_id")
    question = QUESTION_BY_ID[qid]["question"] if qid else body["question"].strip()
    with LOCK:                       # the embedder and caches are not meant for concurrent use
        cfg, graph, retriever = retriever_for(cid)
        rr = retriever.retrieve(question)
        # Try the cache first with a client that cannot send anything; go live only if allowed.
        try:
            result = build_answerer(cfg, dry_run=True).answer(question, rr.context)
        except ExtractionAbort:                     # not cached: the disabled client refused to send
            result = None
        live_used = False
        if result is None or (not result.cache_hit and result.status != "skipped_empty_context"):
            if ALLOW_LIVE:
                result = build_answerer(cfg).answer(question, rr.context)
                live_used = True
            else:
                result = None

    rank_key = strategy if strategy != "random" else f"random_rs{int(body.get('shuffle', 0))}"
    selected = set(RANKINGS[rank_key][: _k(budget)])
    gold = GOLD_CHUNKS.get(qid, []) if qid else []
    top = set(rr.top_entities)
    edges = []
    for u, v, d in graph.edges(data=True):
        if u in top and v in top and u != v:
            edges.append({"source": u, "target": v, "label": d.get("description", "")})
    out = {
        "condition_id": cid,
        "question": question,
        "graph": {"n_nodes": graph.number_of_nodes(), "n_edges": graph.number_of_edges()},
        "n_selected": len(selected),
        "seeds": rr.seeds,
        "entities": [{"id": n, "name": graph.nodes[n]["name"], "type": graph.nodes[n].get("type", "OTHER"),
                      "seed": n in rr.seeds, "rank": i} for i, n in enumerate(rr.top_entities)],
        "edges": edges,
        "facts": rr.facts,
        "passages": [{"id": c, "title": CHUNK_BY_ID[c].source_title, "text": CHUNK_BY_ID[c].text,
                      "gold": c in gold} for c in rr.chunk_ids],
        "context_words": rr.n_words,
        "gold": [{"id": c, "title": CHUNK_BY_ID[c].source_title, "selected": c in selected,
                  "retrieved": c in rr.chunk_ids} for c in gold],
        "answer": None, "live_used": live_used,
    }
    if result is not None:
        out["answer"] = {"text": result.answer, "reasoning": result.reasoning, "status": result.status,
                         "cache_hit": result.cache_hit, "tokens_in": result.usage.input_tokens,
                         "tokens_out": result.usage.output_tokens}
    if qid:
        q = QUESTION_BY_ID[qid]
        out["gold_answer"] = q["answer"]
        if result is not None:
            em, f1 = best_scores(result.answer, [q["answer"], *q.get("answer_aliases", [])])
            out["em"], out["f1"] = em, f1
        rec = logged(cid).get(qid)
        if rec:
            out["logged"] = {"answer": rec["predicted_answer"], "f1": rec["f1"], "em": rec["em"]}
    return out


def _k(budget_pct: int) -> int:
    return max(1, int(budget_pct / 100 * len(CHUNKS) + 0.5))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _planner(self, method: str, path: str, body):
        code, payload, ctype = planner_api.handle(method, path, body)
        if isinstance(payload, bytes):
            return self._send(code, payload, ctype)
        self._json(payload, code)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/planner/"):
            return self._planner("GET", self.path, None)
        if path == "/api/meta":
            return self._json(META)
        if path.startswith("/api/chunk/"):
            c = CHUNK_BY_ID.get(path.rsplit("/", 1)[1])
            return self._json({"id": c.chunk_id, "title": c.source_title, "text": c.text} if c else {}, 200 if c else 404)
        name = "index.html" if path == "/" else path.lstrip("/")
        file = (STATIC / name).resolve()
        if STATIC in file.parents and file.is_file():
            ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                     ".md": "text/markdown", ".txt": "text/plain"}.get(file.suffix, "text/plain")
            return self._send(200, file.read_bytes(), ctype + "; charset=utf-8")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path.startswith("/api/planner/"):
                return self._planner("POST", self.path, body)
            if self.path != "/api/ask":
                return self._send(404, b"not found", "text/plain")
            self._json(ask(body))
        except Exception as e:                       # show the reason in the UI rather than a blank page
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):
        if "/api/ask" in (args[0] if args else ""):
            sys.stderr.write("ask %s\n" % args[1])


def main(argv: list[str] | None = None) -> int:
    global ALLOW_LIVE, RANKINGS, META
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow-live", action="store_true",
                    help="allow paid LLM calls (key in .env): typed-in questions and budget-planner pilots")
    ap.add_argument("--planner-mock", action="store_true",
                    help="run planner pilots with the mock extractor/answerer (offline wiring test, results meaningless)")
    args = ap.parse_args(argv)
    ALLOW_LIVE = args.allow_live
    planner_api.SETTINGS.update(allow_live=args.allow_live, mock=args.planner_mock)
    print("ranking chunks with all four strategies ...", flush=True)
    RANKINGS = compute_rankings()
    META = meta()
    print("warming up the retriever ...", flush=True)
    retriever_for(condition_id("ketrag", 50))[2].retrieve(QUESTIONS[0]["question"])
    print(f"ready: http://127.0.0.1:{args.port}   (live LLM calls {'ON' if ALLOW_LIVE else 'off'})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


RANKINGS: dict = {}
META: dict = {}

if __name__ == "__main__":
    sys.exit(main())
