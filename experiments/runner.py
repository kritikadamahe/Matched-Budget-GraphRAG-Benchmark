"""
Experiment runner (Phase 10): runs the existing pipeline for one condition, or for the whole matrix.

It contains NO strategy, extraction, graph, retrieval, generation or evaluation logic. For each
condition it calls, in order, the existing functions and records what they report:

    corpus      src.prepare_data.build(cfg)                        -> chunks, questions
    ranking     strategies.build_strategy(cfg).rank(chunks)        (budget-independent: computed once
                                                                    per dataset/strategy/seed, reused)
    selection   extraction.pipeline.select_for_extraction          (the budget gate: count checked
                                                                    against src.budget.selected_count)
    extraction  extraction.pipeline.run_extraction(...)            (BudgetGate, cache, spending cap)
    graph       graph.build.build_from_run_dir(...)
    retrieval   retrieval.ppr_retrieval.retriever_from_config(...).retrieve(question)
    generation  generation.answerer.build_answerer(cfg).answer(question, context)
    evaluation  evaluation: PredictionRecord.build, score_predictions, write_outputs

NO GOLD LEAKAGE. Strategies, extraction and retrieval get chunk COPIES whose gold labels
(is_gold_for_question_ids) are stripped. Retrieval and generation receive only question TEXT.
The gold answers are first read in the evaluation stage, after every prediction exists.

PER-CONDITION OUTPUT  <matrix_dir>/conditions/<condition_id>/
    extraction/  graph/  predictions.jsonl  scored.jsonl  eval_summary.json
    condition_result.json    <- the raw measurements; written LAST and atomically

SAFE RERUNS. A condition is "done" only if its condition_result.json says `completed`.
  - completed + same config hash  -> skipped (nothing is called)
  - completed + different config  -> refused as stale (--force to redo it); never silently overwritten
  - anything else (none, aborted, failed, completed_with_failures) -> run again. Cheap: the extraction,
    answer and judge caches are shared, so finished work is not paid for twice.
Every (re)run first deletes that condition's old outputs, so a folder never mixes two runs.
results.jsonl is REGENERATED from the per-condition files after every run: it can never hold duplicates.

FAILURES ARE EXPLICIT. Per-item failures (a chunk, an answer, a judge call) are measured outcomes and
are recorded, not hidden: the condition becomes `completed_with_failures`. A question whose retrieval
raised has no prediction and is listed in `question_failures`. A run-level error (bad key, quota,
spending cap) marks the condition `aborted` and stops the matrix. An unexpected error in one stage marks
the condition `failed` and the matrix goes on. A real backend is never replaced by the mock.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from evaluation.judge import build_judge
from evaluation.schemas import PredictionRecord, write_predictions
from evaluation.score import gold_from_questions, score_predictions, write_outputs
from experiments.conditions import Condition, MatrixSpec, condition_cfg, config_hash
from experiments.schemas import (RESULT_FILE, RUNNER_SCHEMA_VERSION, STAGES, ConditionResult, QuestionFailure,
                                 StageRecord, read_condition_result, write_condition_result)
from extraction import build_extractor
from extraction.base_extractor import ExtractionAbort
from extraction.cache import ExtractionCache
from extraction.pipeline import BudgetViolation, run_extraction, select_for_extraction
from extraction.run import make_run_id
from extraction.schemas import ExtractionInput
from generation.answerer import build_answerer, summarize_query_usage
from graph.build import build_from_run_dir
from graph.graph_builder import load_graph
from retrieval.ppr_retrieval import retriever_from_config
from src.budget import selected_count
from src.config import ExperimentConfig
from src.corpus import PROJECT_ROOT
from src.prepare_data import build as build_corpus
from src.prepare_data import corpus_id, fingerprint
from strategies import build_strategy

OUTPUT_FILES = ("predictions.jsonl", "scored.jsonl", "eval_summary.json", RESULT_FILE)
OUTPUT_DIRS = ("extraction", "graph")
GENERATION_OK = ("ok", "skipped_empty_context")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ----------------------------------------------------------------------------- components
@dataclass
class Components:
    """The existing building blocks the runner calls. The defaults ARE the real modules; the field
    exists only so tests can substitute spies and failure doubles without any API call."""
    load_corpus: Callable = build_corpus
    build_strategy: Callable = build_strategy
    build_extractor: Callable = build_extractor
    build_answerer: Callable = build_answerer
    build_judge: Callable = build_judge
    make_retriever: Callable = retriever_from_config


@dataclass
class RunnerContext:
    base_cfg: ExperimentConfig
    matrix_id: str
    matrix_dir: Path
    comps: Components = field(default_factory=Components)
    no_judge: bool = False
    log: Callable[[str], None] = lambda message: None
    _corpora: dict = field(default_factory=dict)
    _rankings: dict = field(default_factory=dict)

    def condition_dir(self, cond: Condition) -> Path:
        return self.matrix_dir / "conditions" / cond.condition_id

    # -- corpus: built once per dataset/seed; gold labels are stripped from what the pipeline sees
    def corpus(self, cfg: ExperimentConfig):
        key = (cfg.dataset, cfg.data_source, cfg.num_questions, cfg.seed, cfg.chunk_size_words,
               cfg.chunk_overlap_words)
        reused = key in self._corpora
        if not reused:
            raw_chunks, questions, _documents = self.comps.load_corpus(cfg)
            n_labelled = sum(1 for c in raw_chunks if c.is_gold_for_question_ids)
            clean = [dataclasses.replace(c, is_gold_for_question_ids=[]) for c in raw_chunks]
            self._corpora[key] = (clean, questions, n_labelled)
        clean, questions, n_labelled = self._corpora[key]
        return key, clean, questions, n_labelled, reused

    # -- ranking: a strategy's ranking never depends on the budget, so compute it once and reuse it
    def ranking(self, cfg: ExperimentConfig, corpus_key, chunks):
        key = (corpus_key, cfg.strategy, cfg.ketrag_mode, cfg.ketrag_knn_k, cfg.lazy_n_clusters,
               cfg.fast_use_spacy, cfg.embedding_backend, cfg.embedding_model)
        reused = key in self._rankings
        if not reused:
            strategy = self.comps.build_strategy(cfg)
            t = time.perf_counter()
            ranked = list(strategy.rank(chunks))
            self._rankings[key] = (ranked, time.perf_counter() - t)
        ranked, seconds = self._rankings[key]
        return ranked, seconds, reused


# ----------------------------------------------------------------------------- stage helper
class _Stop(Exception):
    """A stage ended the condition. kind: 'aborted' (run-level error) or 'failed' (unexpected error)."""

    def __init__(self, kind: str, stage: str, cause: BaseException):
        super().__init__(f"{stage}: {cause}")
        self.kind, self.stage, self.cause = kind, stage, cause


@contextmanager
def _stage(res: ConditionResult, name: str):
    rec = StageRecord(status="ok")
    res.stages[name] = rec
    started = time.perf_counter()
    try:
        yield rec
    except ExtractionAbort as e:                      # bad key, no quota, spending cap, fatal API error
        rec.status, rec.error = "aborted", str(e)
        raise _Stop("aborted", name, e) from e
    except Exception as e:                            # anything unexpected (incl. BudgetViolation)
        rec.status, rec.error = "failed", f"{type(e).__name__}: {e}"
        raise _Stop("failed", name, e) from e
    finally:
        rec.elapsed_seconds = round(time.perf_counter() - started, 3)


def _reset_condition_dir(cond_dir: Path) -> None:
    """Delete this condition's old outputs, so the folder only ever describes ONE run."""
    for name in OUTPUT_DIRS:
        shutil.rmtree(cond_dir / name, ignore_errors=True)
    for name in OUTPUT_FILES:
        (cond_dir / name).unlink(missing_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- summaries
_EXTRACTION_KEYS = (
    "backend", "model", "is_mock", "prompt_version", "n_chunks_total", "n_selected", "selected_fingerprint",
    "n_processed", "n_ok", "n_failed", "n_truncated", "status_counts", "n_cache_hits", "n_extractor_calls",
    "n_api_attempts", "usage_complete", "n_usage_unknown", "tokens_spent_this_run", "cost_spent_usd",
    "cost_if_uncached_usd", "usage_known_part_lower_bound", "extraction_runtime_seconds_this_run",
    "wall_time_seconds", "validation_issues", "n_entities", "n_relationships", "estimate_before_run", "aborted",
)
_GRAPH_KEYS = ("n_chunks_used", "n_chunks_skipped_not_ok", "n_selected_without_result", "n_nodes", "n_edges",
               "n_isolated_nodes", "n_components", "largest_component_size", "dropped_self_loops",
               "type_distribution", "provenance_check")


def _pick(d: dict, keys) -> dict:
    return {k: d.get(k) for k in keys}


def _cost_part(complete: bool, spent, uncached, lower_bound) -> dict:
    return {"usage_complete": complete, "cost_spent_usd": spent, "cost_if_uncached_usd": uncached,
            "known_cost_lower_bound_usd": lower_bound}


def _extraction_cost(summary: dict) -> dict:
    lb = summary.get("usage_known_part_lower_bound")
    return _cost_part(summary["usage_complete"], summary["cost_spent_usd"], summary["cost_if_uncached_usd"],
                      None if lb is None else lb["cost_if_uncached_usd"])


def _combine_costs(parts: dict[str, dict]) -> dict:
    """Totals over the stages that ran. Unknown is never free: if any included usage is unknown the
    total is None and only a lower bound is given."""
    vals = list(parts.values())
    complete = all(p["usage_complete"] for p in vals)
    spent_known = all(p["cost_spent_usd"] is not None for p in vals)
    lower = sum(p["cost_if_uncached_usd"] if p["usage_complete"] else p["known_cost_lower_bound_usd"] for p in vals)
    return {
        **parts,
        "stages_included": list(parts),
        "usage_complete": complete,
        "total_cost_spent_usd": round(sum(p["cost_spent_usd"] for p in vals), 6) if spent_known else None,
        "total_cost_if_uncached_usd": round(sum(p["cost_if_uncached_usd"] for p in vals), 6) if complete else None,
        "known_cost_lower_bound_usd": None if complete else round(lower, 6),
        "note": ("spent = money paid in THIS run (cache hits cost 0); if_uncached = the same work from scratch. "
                 "Compare conditions on if_uncached. Extraction, generation and judge cost are kept apart."),
    }


# ----------------------------------------------------------------------------- one condition
def run_condition(cond: Condition, ctx: RunnerContext) -> ConditionResult:
    """Run ONE condition end to end and write its condition_result.json. Never raises for a
    pipeline problem: the problem is recorded in the result (a KeyboardInterrupt still propagates)."""
    cond_dir = ctx.condition_dir(cond)
    _reset_condition_dir(cond_dir)
    started_at, t0 = _utcnow(), time.perf_counter()
    res = ConditionResult(
        condition_id=cond.condition_id, matrix_id=ctx.matrix_id, dataset=cond.dataset,
        data_source=ctx.base_cfg.data_source, strategy=cond.strategy, budget=cond.budget,
        budget_pct=cond.budget_pct, seed=cond.seed, started_at_utc=started_at,
        stages={name: StageRecord() for name in STAGES},
    )
    try:
        cfg = condition_cfg(ctx.base_cfg, cond, ctx.matrix_id)
    except ValueError as e:                            # e.g. faithful KET-RAG with TF-IDF embeddings
        res.error = f"invalid configuration for this condition: {e}"
        return _finish(res, cond_dir, None, t0, ctx, None, None, None)
    run_id = make_run_id(cfg)
    res.run_id, res.corpus_id, res.config_hash, res.config = run_id, corpus_id(cfg), config_hash(cfg), cfg.model_dump()

    chunks = questions = ranked = None
    extraction_summary = graph_stats = eval_summary = answerer = None
    records: list[PredictionRecord] = []
    stop: _Stop | None = None
    try:
        # 1. corpus --------------------------------------------------------------------------
        with _stage(res, "corpus") as rec:
            key, chunks, questions, n_labelled, reused = ctx.corpus(cfg)
            res.n_chunks_total, res.n_questions = len(chunks), len(questions)
            rec.details = {"n_chunks": len(chunks), "n_questions": len(questions), "reused_in_memory": reused,
                           "all_chunks_fingerprint": fingerprint([c.chunk_id for c in chunks]),
                           "gold_labels_stripped_from_n_chunks": n_labelled}
        # 2. ranking (the strategy sees chunks only: no budget, no questions, no gold) ---------
        with _stage(res, "ranking") as rec:
            ranked, ranking_seconds, reused = ctx.ranking(cfg, key, chunks)
            rec.details = {"strategy": cfg.strategy, "n_ranked": len(ranked), "ranking_seconds": round(ranking_seconds, 3),
                           "reused_in_memory": reused}
        # 3. selection: the budget gate, BEFORE any extractor exists ---------------------------
        with _stage(res, "selection") as rec:
            expected = selected_count(len(chunks), cfg.budget)
            selection = select_for_extraction(chunks, ranked, cfg.budget)
            if len(selection.selected_ids) != expected:
                raise BudgetViolation(f"selected {len(selection.selected_ids)} chunks, expected {expected}")
            res.n_selected_expected, res.n_selected = expected, len(selection.selected_ids)
            res.selected_fingerprint = selection.fingerprint
            rec.details = {"n_chunks_total": len(chunks), "budget": cfg.budget, "budget_pct": cond.budget_pct,
                           "n_selected_expected": expected, "n_selected": len(selection.selected_ids),
                           "selected_fingerprint": selection.fingerprint}
        # 4. extraction (existing pipeline: BudgetGate + cache + spending cap) -----------------
        with _stage(res, "extraction") as rec:
            extractor = ctx.comps.build_extractor(cfg)
            cache = ExtractionCache(_resolve(cfg.cache_dir) / "extractions", enabled=cfg.extraction_cache_enabled)
            run_meta = {"run_id": run_id, "experiment_id": cfg.experiment_id, "condition_id": cond.condition_id,
                        "strategy": cfg.strategy, "seed": cfg.seed, "dataset": cfg.dataset,
                        "data_source": cfg.data_source, "corpus_id": res.corpus_id,
                        "all_chunks_fingerprint": fingerprint([c.chunk_id for c in chunks]),
                        "ranking_seconds": round(ranking_seconds, 3), "config": cfg.model_dump()}
            try:
                run = run_extraction(chunks, ranked, cfg.budget, extractor, cache,
                                     max_cost_usd=cfg.extraction_max_cost_usd, run_dir=cond_dir / "extraction",
                                     run_meta=run_meta)
                extraction_summary = run.summary
            except ExtractionAbort:
                extraction_summary = _read_json(cond_dir / "extraction" / "run_summary.json")   # partial usage
                if extraction_summary is not None:
                    rec.details = _pick(extraction_summary, _EXTRACTION_KEYS)
                raise
            rec.details = _pick(extraction_summary, _EXTRACTION_KEYS)
            if extraction_summary["n_selected"] != expected:
                raise BudgetViolation(f"extracted {extraction_summary['n_selected']} chunks, expected {expected}")
        # 5. graph ------------------------------------------------------------------------------
        with _stage(res, "graph") as rec:
            (cond_dir / "graph").mkdir(parents=True, exist_ok=True)
            graph_stats = build_from_run_dir(cond_dir / "extraction", cond_dir / "graph")
            rec.details = _pick(graph_stats, _GRAPH_KEYS)
        # 6. retrieval: question TEXT only; a failing question is recorded, not hidden ------------
        retrieved = []
        with _stage(res, "retrieval") as rec:
            retriever = ctx.comps.make_retriever(cfg, load_graph(cond_dir / "graph" / "graph.json"), chunks)
            for q in questions:
                try:
                    retrieved.append((q, retriever.retrieve(q["question"])))
                except Exception as e:
                    res.question_failures.append(QuestionFailure(
                        question_id=str(q["question_id"]), stage="retrieval", error=f"{type(e).__name__}: {e}"))
            if not retrieved:
                raise RuntimeError("retrieval failed for every question")
            rec.details = {
                "n_questions": len(questions), "n_retrieved": len(retrieved),
                "n_failed": len(res.question_failures),
                "n_empty_context": sum(1 for _q, r in retrieved if not r.context.strip()),
                "n_empty_graph": sum(1 for _q, r in retrieved if r.empty_graph),
                "mean_context_words": round(sum(r.n_words for _q, r in retrieved) / len(retrieved), 2),
            }
        # 7. generation: gets (question text, context), nothing else ----------------------------
        with _stage(res, "generation") as rec:
            answerer = ctx.comps.build_answerer(cfg)
            for q, rr in retrieved:
                answer = answerer.answer(q["question"], rr.context)
                records.append(PredictionRecord.build(
                    {"question_id": q["question_id"], "question": q["question"]}, answer, cfg=cfg, retrieval=rr,
                    generation_settings=answerer.settings(), run_id=run_id))
            rec.details = {**summarize_query_usage(answerer.results), "settings": answerer.settings(),
                           "n_predictions": len(records)}
        write_predictions(cond_dir / "predictions.jsonl", records)
        # 8. evaluation: the first place gold answers are read -------------------------------------
        with _stage(res, "evaluation") as rec:
            gold = gold_from_questions(questions)
            judge = None if ctx.no_judge else ctx.comps.build_judge(cfg)
            run_eval = score_predictions(records, gold, judge)
            write_outputs(cond_dir, run_eval)
            eval_summary = run_eval.summary
            rec.details = {"metrics": eval_summary["metrics"], "outcome_counts": eval_summary["outcome_counts"],
                           "judge": {k: eval_summary["judge"].get(k) for k in
                                     ("enabled", "backend", "model", "accuracy", "n_verdicts", "n_judge_failed")},
                           "reportable": eval_summary["reportable"]}
            if eval_summary["aborted"]:
                rec.status, rec.error = "aborted", eval_summary["aborted"]["message"]
    except _Stop as s:
        stop = s
        if records and not (cond_dir / "predictions.jsonl").exists():
            write_predictions(cond_dir / "predictions.jsonl", records)      # keep partial answers for debugging
    return _finish(res, cond_dir, stop, t0, ctx, extraction_summary, answerer, eval_summary, graph_stats, records)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _finish(res: ConditionResult, cond_dir: Path, stop: _Stop | None, t0: float, ctx: RunnerContext,
            extraction_summary, answerer, eval_summary, graph_stats=None, records=()) -> ConditionResult:
    """Derive status, problems, cost, metrics and reportability from what the stages reported; write
    condition_result.json (last, atomically)."""
    cfg_data = res.config or ctx.base_cfg.model_dump()
    res.n_predictions = len(records)
    statuses = [res.stages[s].status for s in STAGES]
    problems: list[str] = []
    warnings: list[str] = []

    # -- cost: only the stages that ran; unknown stays unknown
    parts: dict[str, dict] = {}
    if extraction_summary:
        parts["extraction"] = _extraction_cost(extraction_summary)
    gen_usage = summarize_query_usage(answerer.results) if answerer is not None else None
    if gen_usage:
        parts["generation"] = _cost_part(gen_usage["usage_complete"], gen_usage["cost_spent_usd"],
                                         gen_usage["cost_if_uncached_usd"], gen_usage["known_cost_lower_bound_usd"])
    if eval_summary and eval_summary["cost"]["judge"] is not None:
        j = eval_summary["cost"]["judge"]
        parts["judge"] = _cost_part(j["usage_complete"], j["cost_spent_usd"], j["cost_if_uncached_usd"],
                                    j["known_cost_lower_bound_usd"])
    res.cost = _combine_costs(parts) if parts else {}
    if gen_usage:
        res.cost["generation_detail"] = gen_usage
    if parts and not res.cost["usage_complete"]:
        warnings.append("token usage is unknown for some calls: cost totals are null, not zero")

    # -- problems: failures that were measured, not hidden
    if extraction_summary and extraction_summary.get("n_failed"):
        problems.append(f"{extraction_summary['n_failed']} chunk extraction(s) failed "
                        f"(status counts {extraction_summary['status_counts']})")
    if res.question_failures:
        problems.append(f"{len(res.question_failures)} question(s) have no prediction (retrieval raised)")
    if gen_usage:
        bad = {k: v for k, v in gen_usage["status_counts"].items() if k not in GENERATION_OK}
        if bad:
            problems.append(f"answer generation failed for some questions: {bad}")
    if eval_summary:
        if eval_summary["judge"].get("n_judge_failed"):
            problems.append(f"{eval_summary['judge']['n_judge_failed']} judge call(s) failed")

    if "aborted" in statuses:
        res.status = "aborted"
    elif "failed" in statuses or any(s in ("not_run",) for s in statuses) or res.error:
        res.status = "failed"
    else:
        res.status = "completed_with_failures" if problems else "completed"
    if stop is not None:
        res.error = f"stage '{stop.stage}' {stop.kind}: {stop.cause}"
    res.problems, res.warnings, res.clean = problems, warnings, res.status == "completed"

    # -- headline metrics (copied from the evaluation summary: one file per condition is enough)
    if eval_summary:
        m, j = eval_summary["metrics"], eval_summary["judge"]
        res.metrics = {
            "em": m["em"], "f1": m["f1"], "em_failures_as_wrong": m["em_failures_as_wrong"],
            "f1_failures_as_wrong": m["f1_failures_as_wrong"], "n_scored": eval_summary["n_scored"],
            "n_records": eval_summary["n_records"], "n_failed_generations": eval_summary["n_failed"],
            "outcome_counts": eval_summary["outcome_counts"], "n_abstained": eval_summary["abstentions"]["n"],
            "judge_enabled": j["enabled"], "judge_accuracy": j.get("accuracy"), "judge_n_verdicts": j.get("n_verdicts"),
            "judge_n_failed": j.get("n_judge_failed"),
        }

    # -- mock / reportability: a mock ingredient (by config OR by what actually ran) or an unfinished
    #    condition is not reportable
    reasons: list[str] = []

    def flag(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if cfg_data["data_source"] == "mock":
        flag("mock dataset fixture (data_source: mock)")
    if cfg_data["extraction_backend"] == "mock" or (extraction_summary and extraction_summary.get("is_mock")):
        flag("mock extraction backend")
    if cfg_data["generation_backend"] == "mock" or (answerer is not None and answerer.settings()["backend"] == "mock"):
        flag("mock generation backend")
    if not ctx.no_judge and (cfg_data["judge_backend"] == "mock" or (eval_summary and eval_summary["is_mock_judge"])):
        flag("mock judge backend")
    mock_reasons = list(reasons)
    if res.status not in ("completed", "completed_with_failures"):
        reasons.append(f"condition did not complete (status {res.status})")
    if eval_summary:
        reasons += [r for r in eval_summary["not_reportable_reasons"] if "mock" not in r and r not in reasons]
    res.is_mock, res.not_reportable_reasons = bool(mock_reasons), reasons
    res.reportable = not reasons

    res.finished_at_utc = _utcnow()
    res.wall_seconds = round(time.perf_counter() - t0, 3)
    write_condition_result(cond_dir, res)
    return res


# ----------------------------------------------------------------------------- the matrix
@dataclass
class MatrixRun:
    matrix_dir: Path
    results: list[ConditionResult] = field(default_factory=list)       # conditions run in this invocation
    skipped_completed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)                     # completed under a different config: refused
    stopped: dict | None = None                                        # why the matrix stopped early, if it did


def run_matrix(conditions: list[Condition], ctx: RunnerContext, *, force: bool = False,
               max_total_usd: float | None = None, all_conditions: list[Condition] | None = None) -> MatrixRun:
    """Run the given conditions in order. Resumes safely (see the module docstring), stops at the
    first run-level error, and optionally stops BEFORE a condition once the money spent in this
    invocation reaches max_total_usd (or cannot be known)."""
    out = MatrixRun(matrix_dir=ctx.matrix_dir)
    spent, spend_known = 0.0, True
    total = len(conditions)
    for i, cond in enumerate(conditions, 1):
        cond_dir = ctx.condition_dir(cond)
        existing = read_condition_result(cond_dir)
        if existing is not None and existing.status == "completed" and not force:
            try:
                current_hash = config_hash(condition_cfg(ctx.base_cfg, cond, ctx.matrix_id))
            except ValueError:
                current_hash = None
            if current_hash == existing.config_hash:
                ctx.log(f"[{i}/{total}] {cond.condition_id}: already completed - skipped")
                out.skipped_completed.append(cond.condition_id)
                continue
            ctx.log(f"[{i}/{total}] {cond.condition_id}: STALE - completed under a different config; "
                    f"refusing to overwrite (use --force to redo it)")
            out.stale.append(cond.condition_id)
            continue
        if max_total_usd is not None and (not spend_known or spent >= max_total_usd):
            why = ("spent so far is unknown, so the spending limit cannot be checked" if not spend_known
                   else f"spent ${spent:.4f} reached the limit ${max_total_usd:.4f}")
            out.stopped = {"reason": "max_total_usd", "before_condition": cond.condition_id, "message": why}
            ctx.log(f"STOPPED before {cond.condition_id}: {why}")
            break

        ctx.log(f"[{i}/{total}] {cond.condition_id}: running ...")
        res = run_condition(cond, ctx)
        out.results.append(res)
        total_spent = res.cost.get("total_cost_spent_usd") if res.cost else 0.0
        if total_spent is None:
            spend_known = False
        else:
            spent += total_spent
        ctx.log(f"[{i}/{total}] {cond.condition_id}: {res.status} in {res.wall_seconds}s"
                + (f" - {res.error}" if res.error else ""))
        if res.status == "aborted":
            out.stopped = {"reason": "aborted", "condition_id": cond.condition_id, "message": res.error}
            break
    write_results_index(ctx.matrix_dir, [c.condition_id for c in (all_conditions or conditions)])
    return out


def write_results_index(matrix_dir: Path, ordered_ids: list[str]) -> Path:
    """results.jsonl: one row per condition that has a result, REBUILT from the per-condition files
    (matrix order first), so it can never contain duplicates. `config` is left out of the rows (it is
    in each condition_result.json and in matrix.json)."""
    conditions_dir = matrix_dir / "conditions"
    found = {p.name: read_condition_result(p) for p in sorted(conditions_dir.glob("*")) if p.is_dir()}
    order = [i for i in ordered_ids if found.get(i) is not None]
    order += [i for i in sorted(found) if found[i] is not None and i not in order]
    matrix_dir.mkdir(parents=True, exist_ok=True)
    path = matrix_dir / "results.jsonl"
    tmp = matrix_dir / f"results.jsonl.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for cid in order:
            f.write(json.dumps(found[cid].model_dump(mode="json", exclude={"config"}), ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def write_manifest(ctx: RunnerContext, spec: MatrixSpec, all_conditions: list[Condition]) -> Path:
    """matrix.json: the base config, the matrix spec and the full list of conditions."""
    ctx.matrix_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION, "matrix_id": ctx.matrix_id, "written_at_utc": _utcnow(),
        "spec": spec.model_dump(), "n_conditions": len(all_conditions),
        "conditions": [{"condition_id": c.condition_id, "dataset": c.dataset, "strategy": c.strategy,
                        "budget": c.budget, "budget_pct": c.budget_pct, "seed": c.seed} for c in all_conditions],
        "base_config": ctx.base_cfg.model_dump(),
        "note": "dataset / strategy / budget in base_config are placeholders: each condition overrides them.",
    }
    path = ctx.matrix_dir / "matrix.json"
    tmp = ctx.matrix_dir / f"matrix.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------- safety
def paid_backends(cfg: ExperimentConfig, no_judge: bool) -> list[str]:
    """Which stages of this config would make real (paid) API calls."""
    out = []
    if cfg.extraction_backend == "openai":
        out.append(f"extraction ({cfg.extraction_model})")
    if cfg.generation_backend == "openai":
        out.append(f"generation ({cfg.generation_model})")
    if not no_judge and cfg.judge_backend == "openai":
        out.append(f"judge ({cfg.judge_model})")
    return out


def preflight(cfg: ExperimentConfig, ctx: RunnerContext) -> None:
    """Build every client once BEFORE any condition runs, so a missing API key stops the whole run
    up front (MissingAPIKeyError, an ExtractionAbort). Builds only: nothing is sent."""
    ctx.comps.build_extractor(cfg)
    ctx.comps.build_answerer(cfg)
    if not ctx.no_judge:
        ctx.comps.build_judge(cfg)


# ----------------------------------------------------------------------------- dry-run plan
def plan_matrix(conditions: list[Condition], ctx: RunnerContext) -> dict:
    """What a run WOULD do, with no API call and no key needed: per-condition counts, existing
    status, and ESTIMATED costs. Estimates ignore the shared caches (so are upper bounds): chunks
    selected by several conditions are extracted once, which `extraction_cache_shared_upper_bound`
    reflects: at most every chunk of a dataset once."""
    rows, notes = [], []
    per_dataset_chunk_cost: dict = {}
    for cond in conditions:
        cfg = condition_cfg(ctx.base_cfg, cond, ctx.matrix_id)
        existing = read_condition_result(ctx.condition_dir(cond))
        status = "pending" if existing is None else existing.status
        if existing is not None and existing.status == "completed" and existing.config_hash != config_hash(cfg):
            status = "stale"
        row = {"condition_id": cond.condition_id, "dataset": cond.dataset, "strategy": cond.strategy,
               "budget_pct": cond.budget_pct, "seed": cond.seed, "status": status,
               "n_chunks_total": None, "n_selected": None, "n_questions": None,
               "est_extraction_usd": None, "est_generation_usd": None, "est_judge_usd": None}
        try:
            key, chunks, questions, _n, _r = ctx.corpus(cfg)
        except (FileNotFoundError, ValueError) as e:
            note = f"{cond.dataset}: corpus not available ({e}); counts and estimates left blank"
            if note not in notes:
                notes.append(note)
            rows.append(row)
            continue
        row.update(n_chunks_total=len(chunks), n_selected=selected_count(len(chunks), cfg.budget),
                   n_questions=len(questions))
        extractor = ctx.comps.build_extractor(cfg, dry_run=True)
        if key not in per_dataset_chunk_cost:
            costs = [extractor.estimate_cost_usd(ExtractionInput(chunk_id=c.chunk_id, text=c.text)) for c in chunks]
            per_dataset_chunk_cost[key] = (sum(costs) / len(costs) if costs else 0.0, len(chunks))
        mean_cost, _n_chunks = per_dataset_chunk_cost[key]
        row["est_extraction_usd"] = round(mean_cost * row["n_selected"], 6)
        answerer = ctx.comps.build_answerer(cfg, dry_run=True)
        full_context = " ".join(["word"] * cfg.retrieval_max_context_words)       # worst case: a full context
        row["est_generation_usd"] = round(sum(answerer.estimate_cost_usd(q["question"], full_context)
                                              for q in questions), 6)
        if ctx.no_judge:
            row["est_judge_usd"] = 0.0
        else:
            judge = ctx.comps.build_judge(cfg, dry_run=True)
            row["est_judge_usd"] = round(sum(judge.estimate_cost_usd(q["question"], str(q["answer"]),
                                                                     [str(a) for a in q.get("answer_aliases", [])],
                                                                     str(q["answer"])) for q in questions), 6)
        rows.append(row)

    def total(field_name):
        vals = [r[field_name] for r in rows]
        return None if (not vals or any(v is None for v in vals)) else round(sum(vals), 6)

    shared_bound = None
    if per_dataset_chunk_cost and all(r["est_extraction_usd"] is not None for r in rows):
        by_dataset: dict = {}
        for r in rows:
            by_dataset.setdefault(r["dataset"], []).append(r["est_extraction_usd"])
        caps = {}
        for key, (mean_cost, n_chunks) in per_dataset_chunk_cost.items():
            caps[key[0]] = caps.get(key[0], 0.0) + mean_cost * n_chunks
        shared_bound = round(sum(min(sum(v), caps.get(d, sum(v))) for d, v in by_dataset.items()), 6)
    return {
        "n_conditions": len(rows), "conditions": rows, "notes": notes,
        "totals": {"est_extraction_usd_sum_of_conditions": total("est_extraction_usd"),
                   "extraction_cache_shared_upper_bound_usd": shared_bound,
                   "est_generation_usd_upper_bound": total("est_generation_usd"),
                   "est_judge_usd": total("est_judge_usd")},
        "estimate_note": ("ESTIMATES only (~1.33 tokens/word; extraction uses the corpus's mean chunk cost; "
                          "generation assumes a full retrieval context for every question). Real cost is "
                          "recorded from the token counts the API reports."),
    }
