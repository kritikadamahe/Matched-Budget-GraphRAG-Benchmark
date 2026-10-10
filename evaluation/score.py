"""
Score predictions (Phase 8, Evaluation). This is NOT the experiment runner: it scores
the predictions it is given and does not loop over strategies, budgets or seeds.

    python -m evaluation.score --predictions results/<...>/predictions.jsonl --config CONFIG
    python -m evaluation.score --predictions P --config CONFIG --no-judge     # EM / F1 only
    python -m evaluation.score --predictions P --config CONFIG --dry-run      # judge cost estimate

Input : predictions.jsonl (one evaluation.schemas.PredictionRecord per line) and the gold
        answers, taken from the config's corpus (or from --questions <questions.json>).
Output: scored.jsonl (one ScoredRecord per prediction) and eval_summary.json, next to the
        predictions file (or in --out-dir).

HOW ONE PREDICTION IS SCORED
  failed      the answer call failed (api_error / malformed / refused / truncated). No EM, F1
              or verdict (None, not 0): it is reported separately and never silently becomes
              a wrong answer. Headline numbers are over the non-failed records, and a second
              pair of numbers counts failures as wrong, side by side.
  abstained   the answer is "not found" (or retrieval was empty and no call was made). EM = F1 = 0,
              counted as an abstention, never as a match. No judge call; the verdict is False.
  empty       the model replied with a blank answer. EM = F1 = 0; no judge call; verdict False.
  answered    EM / F1 (SQuAD-style normalisation, best over gold + aliases) and, if a judge is
              given, one judge call that sees only question, reference and candidate answer.

MOCK RESULTS. Any record whose answer or judge came from a mock backend sets is_mock=true
in scored.jsonl and the summary, and the summary says reportable=false with the reasons.
assert_reportable(summary) raises NotReportableError, so the future runner / analysis can
refuse to put such numbers in a table or plot.

Exit codes: 0 all scored cleanly | 1 finished with problems (failed generations, judge
failures, or unknown token usage) | 2 stopped (bad input, missing key, fatal API error).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from evaluation.judge import Judge, build_judge
from evaluation.metrics import best_scores
from evaluation.normalize import normalize_answer
from evaluation.prompts import JUDGE_PROMPT_VERSION
from evaluation.schemas import (EVAL_SCHEMA_VERSION, GoldAnswer, JudgeResult, PredictionRecord,
                                ScoredRecord, read_predictions, write_jsonl)
from extraction.base_extractor import ExtractionAbort
from generation.answerer import summarize_query_usage
from generation.prompts import NOT_FOUND

GENERATION_OK = ("ok", "skipped_empty_context")
JUDGE_FAILURES = ("api_error", "malformed", "refused", "truncated")
_ABSTENTION = normalize_answer(NOT_FOUND)

LIMITATIONS = [
    "Seed coupling: the one config `seed` selects the questions (hence the corpus, N and the "
    "budget counts) AND drives strategy randomness, so Random's seeds also change the corpus. "
    "Scores are comparable across strategies only within one seed. To be addressed in the "
    "experiment runner.",
    "Budget rounding: src/budget.selected_count rounds half up, while the KET-RAG paper uses "
    "ceil(beta * N) (e.g. N=204 at 5%: 10 chunks vs 11). Identical for every strategy, so "
    "internally fair, but not identical to KET-RAG's own budget.",
    "EM / F1 normalisation is SQuAD-style with two small documented extensions (Unicode NFKC, "
    "all Unicode punctuation removed).",
    "The judge also sees MuSiQue answer aliases as 'other acceptable answers'; HotpotQA has none.",
    "An answer of exactly 'not found' is always an abstention (score 0), even if a gold answer "
    "were literally that text.",
]


class NotReportableError(RuntimeError):
    """Raised when mock or incomplete results are about to be used as benchmark results."""


# --------------------------------------------------------------------------------- gold
def gold_from_questions(questions: list[dict]) -> dict[str, GoldAnswer]:
    """Gold answers by question_id from question dicts (src.corpus format or questions.json)."""
    gold: dict[str, GoldAnswer] = {}
    for q in questions:
        qid = str(q["question_id"])
        if qid in gold:
            raise ValueError(f"duplicate question_id in the gold data: {qid!r}")
        gold[qid] = GoldAnswer(question_id=qid, answer=str(q["answer"]),
                               aliases=[str(a) for a in q.get("answer_aliases", [])])
    return gold


def load_gold_file(path: str | Path) -> dict[str, GoldAnswer]:
    """Gold from a questions.json written by src.prepare_data."""
    with open(path, encoding="utf-8") as f:
        return gold_from_questions(json.load(f))


# ------------------------------------------------------------------------ one prediction
def is_abstention(answer: str) -> bool:
    return normalize_answer(answer) == _ABSTENTION


def _outcome(record: PredictionRecord) -> str:
    a = record.answer
    if a.status not in GENERATION_OK:
        return "failed"
    if a.status == "skipped_empty_context" or is_abstention(a.answer):
        return "abstained"
    if not a.answer.strip():
        return "empty"
    return "answered"


def score_record(record: PredictionRecord, gold: GoldAnswer, judge: Judge | None = None, *,
                 no_judge_status: str = "disabled") -> ScoredRecord:
    """Score one prediction. May call the judge (only for an answered record). Raises
    ExtractionAbort if the judge hits a run-level error."""
    a = record.answer
    outcome = _outcome(record)
    if outcome == "failed":
        em = f1 = None
    elif outcome in ("abstained", "empty"):
        em = f1 = 0.0
    else:
        em, f1 = best_scores(a.answer, gold.all_answers)

    judge_result: JudgeResult | None = None
    judge_correct: bool | None = None
    if judge is None:
        judge_status = no_judge_status                       # "disabled" or "not_run"
    elif outcome == "failed":
        judge_status = "skipped_failed_generation"
    elif outcome == "abstained":
        judge_status, judge_correct = "skipped_abstained", False
    elif outcome == "empty":
        judge_status, judge_correct = "skipped_empty", False
    else:
        judge_result = judge.judge(record.question, gold.answer, gold.aliases, a.answer)
        judge_status, judge_correct = judge_result.status, judge_result.correct

    return ScoredRecord(
        question_id=record.question_id, run_id=record.run_id, experiment_id=record.experiment_id,
        dataset=record.dataset, corpus_id=record.corpus_id, strategy=record.strategy,
        budget=record.budget, seed=record.seed,
        gold_answer=gold.answer, gold_aliases=list(gold.aliases),
        predicted_answer=a.answer, generation_status=a.status, generation_error=a.error,
        outcome=outcome, em=em, f1=f1, judge_correct=judge_correct, judge_status=judge_status,
        judge=judge_result,
        is_mock_generation=record.is_mock,
        is_mock_judge=bool(judge is not None and judge.backend == "mock"),
    )


# ------------------------------------------------------------------------- many records
@dataclass
class EvaluationRun:
    scored: list[ScoredRecord]
    summary: dict


def _condition_key(r: PredictionRecord) -> tuple:
    c = r.condition()
    return tuple(c[k] for k in sorted(c)) + (r.question_id,)


def score_predictions(records: list[PredictionRecord], gold: dict[str, GoldAnswer],
                      judge: Judge | None = None) -> EvaluationRun:
    """Score every record. A judge run-level error stops judging but keeps EM / F1 for all
    records (the rest get judge_status "not_run") and marks the summary as aborted."""
    if not records:
        raise ValueError("no predictions to score")
    keys = [_condition_key(r) for r in records]
    if len(set(keys)) != len(keys):
        dup = next(k for k, n in Counter(keys).items() if n > 1)
        raise ValueError(f"duplicate prediction for the same condition and question: {dup[-1]!r}")
    unknown = sorted({r.question_id for r in records} - set(gold))
    if unknown:
        raise ValueError(f"no gold answer for question_id(s): {unknown[:5]}")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    scored: list[ScoredRecord] = []
    aborted: ExtractionAbort | None = None
    for rec in records:
        g = gold[rec.question_id]
        if aborted is None:
            try:
                scored.append(score_record(rec, g, judge))
                continue
            except ExtractionAbort as e:
                aborted = e
        scored.append(score_record(rec, g, None, no_judge_status="not_run"))

    summary = summarize(scored, records, judge, aborted, started, time.perf_counter() - t0)
    return EvaluationRun(scored=scored, summary=summary)


# ------------------------------------------------------------------------------ summary
def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _cost_section(records: list[PredictionRecord], judge_results: list[JudgeResult], judge_on: bool) -> dict:
    gen = summarize_query_usage([r.answer for r in records])
    jud = summarize_query_usage(judge_results) if judge_on else None
    parts = [p for p in (gen, jud) if p is not None]
    complete = all(p["usage_complete"] for p in parts)
    spent_known = all(p["cost_spent_usd"] is not None for p in parts)
    lower = sum(p["cost_if_uncached_usd"] if p["usage_complete"] else p["known_cost_lower_bound_usd"]
                for p in parts)
    return {
        "generation": gen,
        "judge": jud,
        "usage_complete": complete,
        # Unknown is never free: if any usage is unknown the totals are None and only a lower bound is given.
        "total_cost_if_uncached_usd": round(sum(p["cost_if_uncached_usd"] for p in parts), 6) if complete else None,
        "total_cost_spent_usd": round(sum(p["cost_spent_usd"] for p in parts), 6) if spent_known else None,
        "known_cost_lower_bound_usd": None if complete else round(lower, 6),
        "note": ("Costs are query-time / evaluation costs, reported apart from extraction (indexing) cost. "
                 "Use total_cost_if_uncached_usd (cache hits at their original cost), not the spent figure, "
                 "when comparing conditions."),
    }


def summarize(scored: list[ScoredRecord], records: list[PredictionRecord], judge: Judge | None,
              aborted: ExtractionAbort | None, started_at: str, wall_seconds: float) -> dict:
    n = len(scored)
    outcomes = Counter(s.outcome for s in scored)
    failed = [s for s in scored if s.outcome == "failed"]
    non_failed = [s for s in scored if s.outcome != "failed"]
    ems = [s.em for s in non_failed]
    f1s = [s.f1 for s in non_failed]

    judge_results = [s.judge for s in scored if s.judge is not None]
    verdicts = [s.judge_correct for s in scored if s.judge_correct is not None]
    judge_failed = [s for s in scored if s.judge_status in JUDGE_FAILURES]

    models = {r.answer.model for r in records}
    same_model = bool(judge is not None and judge.model in models)
    is_mock_generation = any(s.is_mock_generation for s in scored)
    is_mock_judge = any(s.is_mock_judge for s in scored)

    if judge is None:
        judge_section: dict = {"enabled": False}
    else:
        judge_section = {
            "enabled": True, "backend": judge.backend, "model": judge.model,
            "prompt_version": JUDGE_PROMPT_VERSION, "same_model_as_generator": same_model,
            "n_verdicts": len(verdicts), "n_correct": sum(verdicts),
            "n_incorrect": len(verdicts) - sum(verdicts),
            "accuracy": round(sum(verdicts) / len(verdicts), 6) if verdicts else None,
            "n_calls_made": len(judge_results),
            "n_judge_failed": len(judge_failed),
            "judge_failed_question_ids": [s.question_id for s in judge_failed],
            "status_counts": dict(sorted(Counter(s.judge_status for s in scored).items())),
            "note": ("accuracy = judged-correct / records with a verdict. Abstained and empty answers count "
                     "as incorrect (no call is made); failed generations, failed judge calls and records "
                     "the judge never reached have no verdict and are excluded."),
        }

    not_reportable: list[str] = []
    if is_mock_generation:
        not_reportable.append("answers were produced by a mock generation backend")
    if is_mock_judge:
        not_reportable.append("verdicts were produced by a mock judge backend")
    if aborted is not None:
        not_reportable.append("the run was stopped before every record was judged")

    cost = _cost_section(records, judge_results, judge is not None)
    warnings: list[str] = []
    if failed:
        warnings.append(f"{len(failed)} generation(s) failed and are excluded from the headline EM/F1")
    if judge_failed:
        warnings.append(f"{len(judge_failed)} judge call(s) failed; those records have no verdict")
    if not cost["usage_complete"]:
        warnings.append("token usage is unknown for some calls: cost totals are None, not zero")
    if same_model:
        warnings.append("the judge model is the same as the generation model: possible self-preference "
                        "bias in the judge verdicts; document or use a different judge_model")
    if judge is None:
        warnings.append("no judge was run: only EM and F1 are reported")

    conditions = []
    for r in records:
        c = r.condition()
        if c not in conditions:
            conditions.append(c)

    return {
        "eval_schema_version": EVAL_SCHEMA_VERSION,
        "started_at_utc": started_at,
        "wall_time_seconds": round(wall_seconds, 3),
        "n_conditions": len(conditions),
        "conditions": conditions,
        "n_records": n,
        "n_scored": len(non_failed),
        "n_failed": len(failed),
        "outcome_counts": {k: outcomes.get(k, 0) for k in ("answered", "abstained", "empty", "failed")},
        "metrics": {
            "em": _mean(ems), "f1": _mean(f1s), "n": len(non_failed),
            "em_failures_as_wrong": round(sum(ems) / n, 6) if n else None,
            "f1_failures_as_wrong": round(sum(f1s) / n, 6) if n else None,
            "n_all": n,
            "note": ("em / f1: mean over records whose generation did not fail (an abstention scores 0). "
                     "*_failures_as_wrong: the same over ALL records, a failed generation counting 0."),
        },
        "failures": {
            "n": len(failed),
            "status_counts": dict(sorted(Counter(s.generation_status for s in failed).items())),
            "question_ids": [s.question_id for s in failed],
        },
        "abstentions": {
            "n": outcomes.get("abstained", 0),
            "rate_of_scored": round(outcomes.get("abstained", 0) / len(non_failed), 6) if non_failed else None,
            "n_skipped_empty_context": sum(1 for s in scored if s.generation_status == "skipped_empty_context"),
            "n_empty_answers": outcomes.get("empty", 0),
        },
        "judge": judge_section,
        "cost": cost,
        "is_mock": is_mock_generation or is_mock_judge,
        "is_mock_generation": is_mock_generation,
        "is_mock_judge": is_mock_judge,
        "reportable": not not_reportable,
        "not_reportable_reasons": not_reportable,
        "warnings": warnings,
        "limitations": LIMITATIONS,
        "aborted": None if aborted is None else {"reason": "fatal_error", "message": str(aborted)},
    }


def assert_reportable(summary: dict) -> None:
    """Call before using a summary in a results table or plot."""
    if not summary.get("reportable", False):
        reasons = "; ".join(summary.get("not_reportable_reasons", [])) or "unknown reason"
        raise NotReportableError(f"these results are not reportable: {reasons}")


# -------------------------------------------------------------------------------- output
def write_outputs(out_dir: str | Path, run: EvaluationRun) -> None:
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "scored.jsonl", run.scored)
    tmp = out_dir / f"eval_summary.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(run.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_dir / "eval_summary.json")


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="predictions.jsonl (PredictionRecord per line)")
    ap.add_argument("--config", required=True, help="config with the corpus (gold) and judge_* settings")
    ap.add_argument("--questions", help="questions.json to take gold answers from instead of the config's corpus")
    ap.add_argument("--out-dir", help="where to write scored.jsonl and eval_summary.json (default: next to the predictions)")
    ap.add_argument("--no-judge", action="store_true", help="EM and F1 only")
    ap.add_argument("--dry-run", action="store_true", help="estimate the judge cost; send and write nothing")
    args = ap.parse_args(argv)

    try:
        from src.config import load_config
        cfg = load_config(args.config)
        if args.questions:
            gold = load_gold_file(args.questions)
        else:
            from src.prepare_data import build
            gold = gold_from_questions(build(cfg)[1])
        records = read_predictions(args.predictions)
        judge = None if args.no_judge else build_judge(cfg, dry_run=args.dry_run)

        if args.dry_run:
            if not records:
                raise ValueError("no predictions to score")
            unknown = sorted({r.question_id for r in records} - set(gold))
            if unknown:
                raise ValueError(f"no gold answer for question_id(s): {unknown[:5]}")
            todo = [r for r in records if _outcome(r) == "answered"]
            est = 0.0 if judge is None else sum(
                judge.estimate_cost_usd(r.question, gold[r.question_id].answer,
                                        gold[r.question_id].aliases, r.answer.answer) for r in todo)
            print(f"DRY RUN - nothing sent or written. {len(records)} predictions, {len(todo)} need a judge call, "
                  f"estimated judge cost ${est:.5f} (ESTIMATE: ~1.33 tokens/word, ~{40} output tokens each)")
            return 0

        run = score_predictions(records, gold, judge)
    except (ExtractionAbort, ValueError, FileNotFoundError, ImportError) as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.predictions).resolve().parent
    write_outputs(out_dir, run)

    s = run.summary
    m, j = s["metrics"], s["judge"]
    print(f"scored {s['n_records']} predictions -> {out_dir}")
    print(f"  outcomes: {s['outcome_counts']}")
    print(f"  EM {_fmt(m['em'])}  F1 {_fmt(m['f1'])}  (failures counted as wrong: "
          f"EM {_fmt(m['em_failures_as_wrong'])}  F1 {_fmt(m['f1_failures_as_wrong'])})")
    if j["enabled"]:
        print(f"  judge {j['backend']}/{j['model']}: accuracy {_fmt(j['accuracy'])} "
              f"({j['n_correct']}/{j['n_verdicts']} verdicts), judge failures {j['n_judge_failed']}")
    c = s["cost"]
    print(f"  cost if uncached: {_fmt(c['total_cost_if_uncached_usd'])} USD"
          + ("" if c["usage_complete"] else f"   (UNKNOWN - lower bound {c['known_cost_lower_bound_usd']} USD)"))
    if s["is_mock"]:
        print("  MOCK RESULTS - testing only, NOT reportable.")
    elif not s["reportable"]:
        print(f"  NOT REPORTABLE: {'; '.join(s['not_reportable_reasons'])}", file=sys.stderr)
    for w in s["warnings"]:
        print(f"  WARNING: {w}", file=sys.stderr)
    if s["aborted"]:
        print(f"  STOPPED EARLY: {s['aborted']['message']}", file=sys.stderr)
        return 2
    return 1 if (s["n_failed"] or j.get("n_judge_failed") or not c["usage_complete"]) else 0


if __name__ == "__main__":
    sys.exit(main())
