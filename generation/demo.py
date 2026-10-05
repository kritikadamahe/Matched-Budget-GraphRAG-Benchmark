"""
Answer a few questions end to end (blueprint Phase 10 checkpoint: "output parses
to a clean short answer").

    python -m extraction.run    --config CONFIG     # 1. extract (budget-selected chunks)
    python -m graph.build       --config CONFIG     # 2. build the graph
    python -m generation.demo   --config CONFIG --n 3
    python -m generation.demo   --config CONFIG --n 20 --dry-run   # cost estimate, nothing sent

With strategy: lazygraphrag_native the native L4 path is used instead (no graph:
relevance checks pick the passages, then the same answer generator answers).
Uses generation_backend from the config: "mock" (free, default) or "openai".
Gold answers are shown AFTER answering, for the human reader only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from extraction.base_extractor import ExtractionAbort
from extraction.run import make_run_id
from generation.answerer import build_answerer, build_relevance_checker, summarize_query_usage
from graph.graph_builder import load_graph
from native import build_native
from retrieval.ppr_retrieval import retriever_from_config
from src.config import load_config
from src.corpus import PROJECT_ROOT
from src.prepare_data import build


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="estimate the answer cost; send nothing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    chunks, questions, _ = build(cfg)
    questions = questions[: args.n]
    native = cfg.strategy == "lazygraphrag_native"
    try:
        answerer = build_answerer(cfg, dry_run=args.dry_run)
        if native:
            system = build_native(cfg)
            system.index(chunks)
            checker = build_relevance_checker(cfg, dry_run=args.dry_run)
        else:
            out = Path(cfg.output_dir)
            out = out if out.is_absolute() else PROJECT_ROOT / out
            graph_path = out / "graphs" / make_run_id(cfg) / "graph.json"
            if not graph_path.exists():
                print(f"STOPPED: no graph at {graph_path} - run extraction.run and graph.build first",
                      file=sys.stderr)
                return 2
            retriever = retriever_from_config(cfg, load_graph(graph_path), chunks)

        if args.dry_run:
            if native:
                print("DRY RUN: native L4 relevance checks cannot be estimated without running them; "
                      f"at most {cfg.native_relevance_budget} checks per question.")
                return 0
            est = sum(answerer.estimate_cost_usd(q["question"], retriever.retrieve(q["question"]).context)
                      for q in questions)
            print(f"DRY RUN - nothing sent. backend {cfg.generation_backend}, {len(questions)} questions, "
                  f"estimated answer cost ${est:.5f} (ESTIMATE: ~1.33 tokens/word, ~120 output tokens each)")
            return 0

        for q in questions:
            if native:
                found, info = system.retrieve(q["question"], checker)
                context = system.build_context(found, cfg.retrieval_max_context_words)
            else:
                context = retriever.retrieve(q["question"]).context
            r = answerer.answer(q["question"], context)
            print("=" * 100)
            print(f"Q: {q['question']}")
            print(f"Predicted: {r.answer!r}   [status {r.status}, {r.context_words} context words"
                  f"{', cached' if r.cache_hit else ''}]")
            if r.reasoning:
                print(f"Reasoning: {r.reasoning}")
            print(f"[after the fact] gold: {q['answer']!r}")
    except ExtractionAbort as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2

    print("=" * 100)
    print(f"answers ({cfg.generation_backend}"
          f"{' - MOCK, testing only' if cfg.generation_backend == 'mock' else ''}): "
          f"{summarize_query_usage(answerer.results)}")
    if native:
        print(f"relevance checks: {summarize_query_usage(checker.results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
