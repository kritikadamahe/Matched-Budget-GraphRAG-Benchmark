"""
HTTP endpoints of the budget planner (served by demo/server.py under /api/planner/).

    GET  /api/planner/config               defaults, calibration, whether pilots may run
    POST /api/planner/corpus               {files: [{name, data(base64)}]} -> corpus stats
    POST /api/planner/costs                {corpus_id, price_in, price_out} -> full-corpus cost table
    POST /api/planner/estimate             {corpus_id, spec} -> pilot cost estimate (no calls)
    POST /api/planner/pilot                {corpus_id, spec} -> job id (paid; needs --allow-live)
    GET  /api/planner/job/<id>             progress, then the report
    GET  /api/planner/sample               report from the logged benchmark (free)
    GET  /api/planner/report/<id>.md|.json download (id = job id or "sample")

Uploaded corpora and finished reports are kept under results/planner/<corpus_id>/.
"""

from __future__ import annotations

import base64
import json
import threading
import traceback
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from planner import costs, ingest, pilot, sample
from planner.report_md import to_markdown

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "results" / "planner"
BUDGETS = [10, 25, 50, 75, 100]
SPEC_FIELDS = {"n_questions": int, "sample_chunks": int, "strategies": list, "primary": str, "budgets": list,
               "target": float, "max_usd": float, "model": str, "price_in": float, "price_out": float}

SETTINGS = {"allow_live": False, "mock": False}
CORPORA: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
PILOT_LOCK = threading.Lock()


def _roundtrip(obj):
    """JSON-normalised copy (int dict keys become strings), as the browser and the exporter see it."""
    return json.loads(json.dumps(obj))


def _corpus(cid: str) -> dict:
    if cid not in CORPORA:
        path = STORE / cid / "documents.json"
        if not path.exists():
            raise KeyError(f"unknown corpus {cid!r}: upload the documents again")
        docs = json.loads(path.read_text())
        CORPORA[cid] = {"documents": docs, "chunks": pilot.chunk_documents(docs)}
    return CORPORA[cid]


def _spec(body: dict) -> pilot.PilotSpec:
    raw = body.get("spec", {})
    kw = {k: t(raw[k]) for k, t in SPEC_FIELDS.items() if k in raw and raw[k] not in (None, "")}
    if "strategies" in kw:
        kw["strategies"] = [s for s in kw["strategies"] if s in pilot.NAMES] or ["random"]
    if "budgets" in kw:
        kw["budgets"] = sorted({int(b) for b in kw["budgets"] if 1 <= int(b) <= 100} | {100})
    spec = pilot.PilotSpec(**kw)
    spec.n_questions = max(10, min(spec.n_questions, 150))
    spec.sample_chunks = max(30, min(spec.sample_chunks, 2000))
    spec.target = max(0.5, min(spec.target, 1.0))
    if spec.primary not in spec.strategies:
        spec.primary = spec.strategies[0]
    spec.backend = "mock" if SETTINGS["mock"] else "openai"
    return spec


def config() -> dict:
    cfg = pilot.make_config(pilot.PilotSpec())
    return {"model": cfg.extraction_model, "price_in": cfg.extraction_price_input_per_1m,
            "price_out": cfg.extraction_price_output_per_1m, "budgets": BUDGETS,
            "calibration": costs.calibration_dict(costs.benchmark_calibration()),
            "can_run_pilot": SETTINGS["allow_live"] or SETTINGS["mock"], "mock": SETTINGS["mock"],
            "strategies": pilot.NAMES, "formats": list(ingest.SUPPORTED)}


def upload(body: dict) -> dict:
    files = [(f["name"], base64.b64decode(f["data"])) for f in body.get("files", [])]
    if not files:
        raise ValueError("no files received")
    docs, report = ingest.parse_files(files)
    if not docs:
        raise ValueError("none of the files contained readable text: " + "; ".join(f"{r['file']}: {r.get('reason')}" for r in report))
    cid = ingest.corpus_id(files)
    (STORE / cid).mkdir(parents=True, exist_ok=True)
    (STORE / cid / "documents.json").write_text(json.dumps(docs))
    chunks = pilot.chunk_documents(docs)
    CORPORA[cid] = {"documents": docs, "chunks": chunks}
    return {"corpus_id": cid, "files": report, "n_documents": len(docs), "n_chunks": len(chunks),
            "n_words": sum(len(c.text.split()) for c in chunks)}


def cost_table(body: dict) -> dict:
    p_in, p_out = float(body["price_in"]), float(body["price_out"])
    cal = costs.benchmark_calibration()
    if body["corpus_id"] == "sample":
        words = [len(c["text"].split()) for c in json.loads((costs.CORPUS / "chunks.json").read_text())]
    else:
        words = [len(c.text.split()) for c in _corpus(body["corpus_id"])["chunks"]]
    rows = costs.extraction_table(words, BUDGETS, cal, p_in, p_out)
    return {"rows": rows, "per_1000_questions_usd": round(costs.query_cost(1000, cal, p_in, p_out), 4),
            "calibration": costs.calibration_dict(cal)}


def estimate(body: dict) -> dict:
    return pilot.estimate(_spec(body), _corpus(body["corpus_id"])["chunks"])


def start_pilot(body: dict) -> dict:
    if not (SETTINGS["allow_live"] or SETTINGS["mock"]):
        raise PermissionError("pilots make paid LLM calls: restart the server with --allow-live")
    cid, spec = body["corpus_id"], _spec(body)
    documents = _corpus(cid)["documents"]
    job_id = uuid.uuid4().hex[:10]
    job = JOBS[job_id] = {"id": job_id, "corpus_id": cid, "status": "queued", "stage": "", "done": 0, "total": 0,
                          "message": "waiting for the previous pilot to finish", "log": [], "report": None, "error": None}

    def progress(stage, done, total, msg=""):
        job.update(stage=stage, done=done, total=total, status="running")
        if msg:
            job["message"] = msg
            job["log"].append(msg)

    def work():
        with PILOT_LOCK:
            try:
                report = _roundtrip(pilot.run(spec, documents, progress))
                (STORE / cid / f"report_{job_id}.json").write_text(json.dumps(report, indent=1))
                job.update(status="done", report=report)
            except Exception as e:                   # noqa: BLE001 - reported to the UI
                traceback.print_exc()
                job.update(status="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


def _report(rid: str) -> dict:
    if rid == "sample":
        return _roundtrip(sample.report())
    job = JOBS.get(rid)
    if job and job["report"]:
        return job["report"]
    for path in STORE.glob(f"*/report_{rid}.json"):
        return json.loads(path.read_text())
    raise KeyError(f"no report {rid!r}")


def handle(method: str, path: str, body: dict | None):
    """Returns (status, payload, content_type); payload is a dict (JSON) or bytes."""
    url = urlsplit(path)
    path, query = url.path, parse_qs(url.query)
    try:
        if method == "GET" and path == "/api/planner/config":
            return 200, config(), None
        if method == "GET" and path == "/api/planner/sample":
            return 200, _report("sample"), None
        if method == "GET" and path.startswith("/api/planner/job/"):
            job = JOBS.get(path.rsplit("/", 1)[1])
            return (200, job, None) if job else (404, {"error": "unknown job"}, None)
        if method == "GET" and path.startswith("/api/planner/report/"):
            name = path.rsplit("/", 1)[1]
            rid, _, ext = name.rpartition(".")
            r = _report(rid)
            if "target" in query:                    # the target slider in the UI: re-derive the recommendation
                r = dict(r, recommendation=_roundtrip(pilot.recommend(
                    {int(b): v for b, v in r["quality"][r["primary"]].items()}, r["budgets"],
                    max(0.5, min(float(query["target"][0]), 1.0)), r["cost_projection"], r.get("vector"))))
                r["target"] = r["recommendation"]["target"]
            if ext == "md":
                return 200, to_markdown(r).encode(), "text/markdown; charset=utf-8"
            return 200, json.dumps(r, indent=1).encode(), "application/json"
        if method == "POST":
            fn = {"/api/planner/corpus": upload, "/api/planner/costs": cost_table,
                  "/api/planner/estimate": estimate, "/api/planner/pilot": start_pilot}.get(path)
            if fn:
                return 200, fn(body or {}), None
        return 404, {"error": "not found"}, None
    except PermissionError as e:
        return 403, {"error": str(e)}, None
    except (KeyError, ValueError) as e:
        return 400, {"error": str(e).strip("'\"")}, None
    except Exception as e:                           # noqa: BLE001
        traceback.print_exc()
        return 500, {"error": f"{type(e).__name__}: {e}"}, None
