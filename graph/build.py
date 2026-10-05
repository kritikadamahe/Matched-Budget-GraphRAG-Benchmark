"""
Build the knowledge graph for ONE extraction run (blueprint Phase 8).

    python -m extraction.run --config configs/extraction_dev.yaml     # extraction first
    python -m graph.build   --config configs/extraction_dev.yaml      # then the graph

Reads results/extractions/<run_id>/{extractions.jsonl, selection.json}, builds the
graph, checks provenance (the Phase 8 checkpoint) and writes
results/graphs/<run_id>/{graph.json, graph_stats.json}.
(--run-dir points at an extraction run folder directly, instead of a config.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from extraction.run import make_run_id
from extraction.schemas import ExtractionResult
from graph.graph_builder import build_graph, check_provenance, save_graph
from src.config import load_config
from src.corpus import PROJECT_ROOT


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def build_from_run_dir(run_dir: Path, out_dir: Path) -> dict:
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    with open(run_dir / "extractions.jsonl", encoding="utf-8") as f:
        results = [ExtractionResult.model_validate_json(line) for line in f if line.strip()]

    G, stats = build_graph(results)
    check_provenance(G, selection["selected_chunk_ids"])
    missing = set(selection["selected_chunk_ids"]) - {r.chunk_id for r in results}
    stats = {
        "run_id": run_dir.name,
        "budget": selection["budget"],
        "n_chunks_total": selection["n_chunks_total"],
        "n_selected": selection["n_selected"],
        "n_selected_without_result": len(missing),   # e.g. a run stopped by the spending cap
        **stats,
        "provenance_check": "passed",
    }
    save_graph(G, out_dir / "graph.json")
    (out_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="the config the extraction run used")
    group.add_argument("--run-dir", help="an extraction run folder (results/extractions/<run_id>)")
    args = ap.parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
        run_dir = _resolve(cfg.output_dir) / "extractions" / make_run_id(cfg)
        out_dir = _resolve(cfg.output_dir) / "graphs" / make_run_id(cfg)
    else:
        run_dir = Path(args.run_dir)
        out_dir = run_dir.parent.parent / "graphs" / run_dir.name
    if not (run_dir / "extractions.jsonl").exists():
        print(f"STOPPED: no extraction run at {run_dir} - run `python -m extraction.run` first", file=sys.stderr)
        return 2
    try:
        stats = build_from_run_dir(run_dir, out_dir)
    except ValueError as e:                 # provenance check failed
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2

    print(f"graph: {out_dir / 'graph.json'}")
    for k in ("n_selected", "n_chunks_used", "n_chunks_skipped_not_ok", "n_nodes", "n_edges",
              "n_isolated_nodes", "n_components", "largest_component_size", "dropped_self_loops",
              "type_distribution", "provenance_check"):
        print(f"  {k}: {stats[k]}")
    if stats["n_chunks_skipped_not_ok"] or stats["n_selected_without_result"]:
        print("  WARNING: some selected chunks contributed nothing (failed or missing extraction)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
