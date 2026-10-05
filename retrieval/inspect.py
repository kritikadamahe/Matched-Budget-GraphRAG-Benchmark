"""
Inspect retrieval for a few questions (blueprint Phase 9 checkpoint:
"manually inspect retrieved context for 3 questions").

    python -m extraction.run --config CONFIG      # 1. extract
    python -m graph.build   --config CONFIG       # 2. build the graph
    python -m retrieval.inspect --config CONFIG --n 3

For each question prints the seeds, the top entities, the context that the answer
model would receive, and - AFTERWARDS, for the human reader only - whether the
question's gold chunks were among the retrieved passages. Retrieval itself never
sees gold labels.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from extraction.run import make_run_id
from graph.graph_builder import load_graph
from retrieval.ppr_retrieval import retriever_from_config
from src.config import load_config
from src.corpus import PROJECT_ROOT
from src.prepare_data import build


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=3, help="number of questions to show")
    ap.add_argument("--full", action="store_true", help="print the whole context, not a preview")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(cfg.output_dir)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    graph_path = out / "graphs" / make_run_id(cfg) / "graph.json"
    if not graph_path.exists():
        print(f"STOPPED: no graph at {graph_path} - run extraction.run and graph.build first", file=sys.stderr)
        return 2

    chunks, questions, _ = build(cfg)
    graph = load_graph(graph_path)
    retriever = retriever_from_config(cfg, graph, chunks)
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
          f"(strategy {cfg.strategy}, budget {cfg.budget:.0%})\n")

    for q in questions[: args.n]:
        r = retriever.retrieve(q["question"])
        gold = {c.chunk_id for c in chunks if q["question_id"] in c.is_gold_for_question_ids}
        print("=" * 100)
        print(f"Q: {q['question']}\nGold answer: {q['answer']}")
        if r.empty_graph:
            print("  (empty graph - no context)")
            continue
        names = lambda keys: [graph.nodes[k]["name"] for k in keys]
        print(f"Seeds: {names(r.seeds)}")
        print(f"Top entities: {names(r.top_entities[:10])}{' ...' if len(r.top_entities) > 10 else ''}")
        print(f"Context: {r.n_words} words, {len(r.facts)} facts, {len(r.chunk_ids)} passages")
        print(f"[after the fact] gold chunks retrieved: {len(gold & set(r.chunk_ids))}/{len(gold)}")
        body = r.context if args.full else textwrap.shorten(r.context.replace("\n", " | "), 900)
        print(textwrap.indent(body, "  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
