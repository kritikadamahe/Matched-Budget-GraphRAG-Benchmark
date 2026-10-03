"""
Extraction pipeline (Phase 4).

    all chunks + strategy ranking + budget
        -> select_top_budget()            (existing code, unchanged)
        -> BudgetGate: only these ids may reach the extractor
        -> for each selected chunk: cache hit? reuse : call extractor, then cache
        -> results/extractions/<run_id>/{extractions.jsonl, selection.json, run_summary.json}

HOW "ONLY BUDGET-SELECTED CHUNKS REACH THE EXTRACTOR" IS ENFORCED (layers)
1. By construction: run_extraction() takes (all chunks, ranking, budget) and does
   the selection itself with the existing select_top_budget(). There is no
   function that sends a hand-picked chunk list to an extractor.
2. The extractor is wrapped in a BudgetGate holding the selected ids. Any other
   id raises BudgetViolation - the run stops before anything is sent.
3. The extractor is only ever handed an ExtractionInput(chunk_id, text): no gold
   labels, no questions, no answers.
4. The cache is looked up per SELECTED chunk only, so cached extractions of
   unselected chunks can never leak into a run's results.
5. selection.json stores exactly what was selected, so a finished run can be
   audited afterwards.
The tests in tests/test_extraction_pipeline.py prove each layer.

COST ACCOUNTING (blueprint section M)
- cost_spent_usd      : money actually spent in THIS run (cache hits cost 0).
- cost_if_uncached_usd: what the same extraction costs from scratch, counting
                        cached entries at their original cost. THIS is the number
                        to use for a strategy's cost on the budget-vs-quality plot,
                        otherwise a cache hit would make a strategy look free.
- UNKNOWN IS NOT FREE: if an API response had no usage information, that chunk's
  usage is unknown (None). Any total that includes it becomes None (null in the
  JSON), `usage_complete` is False, and the known part is reported separately as a
  lower bound. A cost of None can never be mistaken for $0. (The spending-cap guard
  assumes the pre-run ESTIMATE for such a call so it cannot be bypassed; the
  estimate is a safety guard only and is never recorded as actual usage.)
  Results with unknown usage are not cached, so the next run extracts them again.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from extraction.base_extractor import ExtractionAbort, Extractor
from extraction.cache import ExtractionCache
from extraction.schemas import ExtractionInput, ExtractionResult
from src.budget import select_top_budget, selected_count
from src.chunking import Chunk
from src.prepare_data import fingerprint

ESTIMATE_NOTE = (
    "ESTIMATE: ~1.33 tokens per word + prompt overhead, and an assumed 250 output tokens per chunk, "
    "at the configured prices. Real cost is recorded from the token counts the API reports."
)


class BudgetViolation(RuntimeError):
    """A chunk outside the budget-selected set was about to reach the extractor."""


class CostCapExceeded(ExtractionAbort):
    """The spending cap would be exceeded; the run stops before spending more."""


# --------------------------------------------------------------------------- selection
@dataclass(frozen=True)
class Selection:
    selected_ids: tuple[str, ...]           # in rank order (best first)
    selected_chunks: tuple[Chunk, ...]
    n_total: int
    budget: float
    fingerprint: str                        # of the selected ids, in order


def select_for_extraction(chunks: list[Chunk], ranked_ids: list[str], budget: float) -> Selection:
    """Apply the budget to a strategy's ranking, using the EXISTING select_top_budget()."""
    by_id = {c.chunk_id: c for c in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("chunk_ids must be unique")
    if sorted(ranked_ids) != sorted(by_id):
        raise ValueError("the ranking must contain every chunk_id exactly once")

    selected_ids = select_top_budget(list(ranked_ids), budget)
    if len(selected_ids) != selected_count(len(chunks), budget):
        raise BudgetViolation(
            f"budget selection returned {len(selected_ids)} chunks, expected "
            f"{selected_count(len(chunks), budget)}"
        )
    return Selection(
        selected_ids=tuple(selected_ids),
        selected_chunks=tuple(by_id[i] for i in selected_ids),
        n_total=len(chunks), budget=budget, fingerprint=fingerprint(selected_ids),
    )


class BudgetGate:
    """Wraps an extractor so it can only be called for budget-selected chunk ids."""

    def __init__(self, extractor: Extractor, allowed_ids):
        self._extractor = extractor
        self._allowed = frozenset(allowed_ids)

    def extract(self, item: ExtractionInput) -> ExtractionResult:
        if item.chunk_id not in self._allowed:
            raise BudgetViolation(
                f"chunk {item.chunk_id} is not budget-selected and must not reach the extractor"
            )
        return self._extractor.extract(item)


# ------------------------------------------------------------------------------ dry run
def plan_extraction(chunks, ranked_ids, budget, extractor: Extractor,
                    cache: ExtractionCache | None = None) -> dict:
    """What a run WOULD do, without calling the extractor at all (used by --dry-run
    and by the pre-flight spending check)."""
    selection = select_for_extraction(chunks, ranked_ids, budget)
    settings = extractor.cache_settings()
    n_cached, estimated = 0, 0.0
    for chunk in selection.selected_chunks:
        item = ExtractionInput(chunk_id=chunk.chunk_id, text=chunk.text)
        if cache is not None and cache.get(settings, item) is not None:
            n_cached += 1
        else:
            estimated += extractor.estimate_cost_usd(item)
    return {
        "backend": extractor.name, "model": extractor.model,
        "n_chunks_total": selection.n_total, "budget": budget,
        "n_selected": len(selection.selected_ids),
        "n_already_cached": n_cached,
        "n_to_extract": len(selection.selected_ids) - n_cached,
        "estimated_cost_usd": round(estimated, 6),
        "estimate_note": ESTIMATE_NOTE,
        "selected_fingerprint": selection.fingerprint,
    }


# ----------------------------------------------------------------------------- the run
@dataclass
class ExtractionRun:
    selection: Selection
    results: list[ExtractionResult]
    summary: dict


def run_extraction(
    chunks: list[Chunk],
    ranked_ids: list[str],
    budget: float,
    extractor: Extractor,
    cache: ExtractionCache | None = None,
    *,
    max_cost_usd: float | None = None,
    run_dir: str | Path | None = None,
    run_meta: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ExtractionRun:
    """Select by budget, then extract ONLY the selected chunks (cache first).

    Raises CostCapExceeded before any call if the estimate is above max_cost_usd,
    and stops mid-run (saving partial results) if actual spending would pass it.
    Because the cap is checked against an estimate before each call, the final
    spend can overshoot by at most the estimation error of one call.
    ExtractionAbort (bad key, no quota...) also stops the run and saves what was done.
    """
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()

    selection = select_for_extraction(chunks, ranked_ids, budget)
    items = [ExtractionInput(chunk_id=c.chunk_id, text=c.text) for c in selection.selected_chunks]
    settings = extractor.cache_settings()

    plan = plan_extraction(chunks, ranked_ids, budget, extractor, cache)
    if max_cost_usd is not None and plan["estimated_cost_usd"] > max_cost_usd:
        raise CostCapExceeded(
            f"estimated cost ${plan['estimated_cost_usd']:.4f} for {plan['n_to_extract']} chunk(s) "
            f"exceeds the spending cap ${max_cost_usd:.4f}; nothing was sent"
        )

    gate = BudgetGate(extractor, selection.selected_ids)
    results: list[ExtractionResult] = []
    spent = 0.0
    abort: ExtractionAbort | None = None
    try:
        for n, item in enumerate(items, 1):
            cached = cache.get(settings, item) if cache is not None else None
            if cached is not None:
                results.append(cached)
            else:
                if max_cost_usd is not None and spent + extractor.estimate_cost_usd(item) > max_cost_usd:
                    raise CostCapExceeded(
                        f"spending cap ${max_cost_usd:.4f} reached (spent ${spent:.4f}); stopped "
                        f"after {len(results)} of {len(items)} chunks. Re-run to resume - finished "
                        f"chunks are cached."
                    )
                result = gate.extract(item)
                if result.chunk_id != item.chunk_id:
                    raise BudgetViolation(
                        f"extractor returned chunk {result.chunk_id} for request {item.chunk_id}"
                    )
                # Unknown cost must not look free to the cap guard: assume the estimate.
                # (Guard only - the result keeps usage=None; nothing is recorded as actual.)
                spent += (result.usage.cost_usd if result.usage.cost_usd is not None
                          else extractor.estimate_cost_usd(item))
                if cache is not None:
                    cache.put(settings, item, result)
                results.append(result)
            if progress is not None:
                progress(n, len(items))
    except ExtractionAbort as e:
        abort = e

    if not {r.chunk_id for r in results} <= set(selection.selected_ids):
        raise BudgetViolation("results contain chunks outside the budget-selected set")

    summary = _summarize(selection, results, extractor, plan, abort, started_at,
                         time.perf_counter() - t0, run_meta)
    if run_dir is not None:
        _write_outputs(Path(run_dir), selection, results, summary, run_meta)
    if abort is not None:
        raise abort
    return ExtractionRun(selection=selection, results=results, summary=summary)


# -------------------------------------------------------------------------------- output
def _summarize(selection, results, extractor, plan, abort, started_at, wall_seconds, run_meta) -> dict:
    fresh = [r for r in results if not r.cache_hit]
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    # Usage totals. A total that includes an UNKNOWN usage is itself unknown (None),
    # never a number that silently treats the missing part as zero.
    unknown_all = [r for r in results if not r.usage.known]
    unknown_fresh = [r for r in fresh if not r.usage.known]
    known_fresh = [r for r in fresh if r.usage.known]
    known_all = [r for r in results if r.usage.known]

    def total(rs, field, ndigits=None):
        value = sum(getattr(r.usage, field) for r in rs)
        return round(value, ndigits) if ndigits is not None else value

    usage_complete = not unknown_all
    summary = {
        **(run_meta or {}),
        "backend": extractor.name,
        "model": extractor.model,
        "is_mock": extractor.name == "mock",
        "prompt_version": results[0].prompt_version if results else None,
        "cache_settings": extractor.cache_settings(),
        "started_at_utc": started_at,
        "wall_time_seconds": round(wall_seconds, 3),
        "budget": selection.budget,
        "n_chunks_total": selection.n_total,
        "n_selected": len(selection.selected_ids),
        "selected_fingerprint": selection.fingerprint,
        "n_processed": len(results),
        "status_counts": by_status,
        "n_ok": by_status.get("ok", 0),
        "n_failed": sum(v for k, v in by_status.items() if k != "ok"),
        "n_truncated": by_status.get("truncated", 0),
        "failed_chunk_ids": [r.chunk_id for r in results if r.status != "ok"],
        "n_cache_hits": len(results) - len(fresh),
        "n_extractor_calls": len(fresh),
        "n_api_attempts": sum(r.attempts for r in fresh),
        "usage_complete": usage_complete,                 # False => some totals below are None
        "n_usage_unknown": len(unknown_all),
        "usage_unknown_chunk_ids": [r.chunk_id for r in unknown_all],
        "tokens_spent_this_run": {
            "input": None if unknown_fresh else total(fresh, "input_tokens"),
            "output": None if unknown_fresh else total(fresh, "output_tokens"),
        },
        "cost_spent_usd": None if unknown_fresh else total(fresh, "cost_usd", 6),
        "cost_if_uncached_usd": None if unknown_all else total(results, "cost_usd", 6),
        # Only when something is unknown: what IS known (a LOWER BOUND, never the real total).
        "usage_known_part_lower_bound": None if usage_complete else {
            "tokens_input_this_run": total(known_fresh, "input_tokens"),
            "tokens_output_this_run": total(known_fresh, "output_tokens"),
            "cost_spent_usd": total(known_fresh, "cost_usd", 6),
            "cost_if_uncached_usd": total(known_all, "cost_usd", 6),
        },
        "extraction_runtime_seconds_this_run": round(sum(r.runtime_seconds for r in fresh), 3),
        "validation_issues": sum(r.validation_issues for r in results),
        "n_entities": sum(len(r.entities) for r in results),
        "n_relationships": sum(len(r.relationships) for r in results),
        "estimate_before_run": {k: plan[k] for k in ("n_already_cached", "n_to_extract", "estimated_cost_usd")},
        "aborted": None,
    }
    if abort is not None:
        summary["aborted"] = {
            "reason": "cost_cap" if isinstance(abort, CostCapExceeded) else "fatal_error",
            "message": str(abort),
        }
    return summary


def _write_outputs(run_dir: Path, selection: Selection, results, summary, run_meta) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_doc = {
        **{k: v for k, v in (run_meta or {}).items() if k in ("run_id", "strategy", "seed")},
        "budget": selection.budget,
        "n_chunks_total": selection.n_total,
        "n_selected": len(selection.selected_ids),
        "selected_fingerprint": selection.fingerprint,
        "selected_chunk_ids": list(selection.selected_ids),   # rank order, best first
    }
    (run_dir / "selection.json").write_text(json.dumps(selection_doc, indent=2), encoding="utf-8")
    with open(run_dir / "extractions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
