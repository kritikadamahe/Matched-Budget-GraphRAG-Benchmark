"""
Run ONE extraction (Phase 4) for ONE config. This is NOT the benchmark runner:
it does not loop over strategies, budgets or seeds.

    python -m extraction.run --config configs/extraction_dev.yaml --dry-run
    python -m extraction.run --config configs/extraction_dev.yaml

Steps: rebuild the corpus (deterministic) -> strategy ranks all chunks -> budget
picks the top fraction -> ONLY those chunks are extracted (cache first) ->
results/extractions/<run_id>/ is written.

--dry-run shows how many chunks would be sent, how many are already cached and an
estimated cost, and makes no extractor call at all (it works without an API key).

Exit codes: 0 all chunks extracted and usage/cost fully known |
            1 finished with problems (failed or truncated chunks, or token usage
              missing from the API so the cost is unknown) |
            2 stopped (bad config, missing key, fatal API error, spending cap).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from extraction import build_extractor
from extraction.base_extractor import ExtractionAbort
from extraction.cache import ExtractionCache
from extraction.pipeline import plan_extraction, run_extraction
from src.config import ExperimentConfig, load_config
from src.corpus import PROJECT_ROOT
from src.prepare_data import build, corpus_id, fingerprint
from strategies import build_strategy


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _money(value) -> str:
    """Dollar amount for display; None (unknown) is shown as UNKNOWN, never as $0."""
    return "UNKNOWN" if value is None else f"${value}"


def make_run_id(cfg: ExperimentConfig) -> str:
    model = cfg.extraction_model if cfg.extraction_backend == "openai" else "mock"
    raw = (f"{cfg.experiment_id}__{cfg.extraction_backend}-{model}__"
           f"{cfg.strategy}_b{cfg.budget:.2f}__{corpus_id(cfg)}")
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be extracted and the estimated cost; call nothing")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
        chunks, _questions, _documents = build(cfg)
        strategy = build_strategy(cfg)           # raises for lazygraphrag_native (no budget)
        t = time.perf_counter()
        ranked = strategy.rank(chunks)
        ranking_seconds = time.perf_counter() - t
        extractor = build_extractor(cfg, dry_run=args.dry_run)
        cache = ExtractionCache(_resolve(cfg.cache_dir) / "extractions", enabled=cfg.extraction_cache_enabled)

        if args.dry_run:
            plan = plan_extraction(chunks, ranked, cfg.budget, extractor, cache)
            print("DRY RUN - nothing is sent to any extractor")
            print(f"  strategy: {cfg.strategy}   budget: {cfg.budget:.0%}")
            for k, v in plan.items():
                print(f"  {k}: {v}")
            if cfg.extraction_max_cost_usd is not None:
                verdict = "within" if plan["estimated_cost_usd"] <= cfg.extraction_max_cost_usd else "ABOVE"
                print(f"  spending cap ${cfg.extraction_max_cost_usd}: estimate is {verdict} the cap")
            return 0

        run_id = make_run_id(cfg)
        run_dir = _resolve(cfg.output_dir) / "extractions" / run_id
        run_meta = {
            "run_id": run_id, "experiment_id": cfg.experiment_id, "strategy": cfg.strategy,
            "seed": cfg.seed, "dataset": cfg.dataset, "data_source": cfg.data_source,
            "corpus_id": corpus_id(cfg),
            "all_chunks_fingerprint": fingerprint([c.chunk_id for c in chunks]),
            "ranking_seconds": round(ranking_seconds, 3),
            "config": cfg.model_dump(),
        }

        def progress(done: int, total: int) -> None:
            if done == total or done % 10 == 0:
                print(f"  extracted {done}/{total}", flush=True)

        run = run_extraction(chunks, ranked, cfg.budget, extractor, cache,
                             max_cost_usd=cfg.extraction_max_cost_usd, run_dir=run_dir,
                             run_meta=run_meta, progress=progress)
    except (ExtractionAbort, ValueError, ImportError, FileNotFoundError) as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2

    s = run.summary
    print(f"run: {s['run_id']}")
    print(f"  backend/model: {s['backend']}/{s['model']}" + ("   (MOCK - testing only)" if s["is_mock"] else ""))
    print(f"  selected {s['n_selected']} of {s['n_chunks_total']} chunks (budget {s['budget']:.0%})")
    print(f"  ok: {s['n_ok']}  failed: {s['n_failed']}  cache hits: {s['n_cache_hits']}  "
          f"extractor calls: {s['n_extractor_calls']}")
    print(f"  tokens this run: {s['tokens_spent_this_run']}")
    print(f"  cost spent this run: {_money(s['cost_spent_usd'])}   "
          f"cost if uncached: {_money(s['cost_if_uncached_usd'])}")
    print(f"  entities: {s['n_entities']}  relationships: {s['n_relationships']}  "
          f"validation issues: {s['validation_issues']}")
    print(f"  output: {run_dir}")
    problems = False
    if s["n_failed"]:
        print(f"  WARNING: failed chunks: {s['failed_chunk_ids']}", file=sys.stderr)
        if s["n_truncated"]:
            print(f"  {s['n_truncated']} chunk(s) were TRUNCATED (reply hit the output limit). Re-running "
                  f"as-is would truncate again: raise extraction_max_output_tokens first.", file=sys.stderr)
        problems = True
    if not s["usage_complete"]:
        known = s["usage_known_part_lower_bound"]
        print(f"  WARNING: the API reported no token usage for {s['n_usage_unknown']} chunk(s), so the "
              f"cost is UNKNOWN (known part only: ${known['cost_if_uncached_usd']}, a lower bound). "
              f"Do not use this run's cost figures as complete.", file=sys.stderr)
        problems = True
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
