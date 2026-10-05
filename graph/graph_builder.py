"""
Knowledge-graph construction (blueprint Phase 8, §I).

    extraction results (budget-selected chunks only)
        -> one node per entity, merged across chunks by normalised name
        -> one edge per relationship, relation text as an edge attribute
        -> every node and edge remembers the chunks it came from (provenance)

Built identically for every strategy and budget, so the graph differs between
conditions ONLY because different chunks were extracted (§N).

DECISIONS (agreed for Phase 8)
- Name merging: light cleanup, exact rules, no guessing (normalize_entity):
  Unicode NFKC, lowercase, "." and apostrophes removed, other punctuation -> space,
  a leading "the" dropped, whitespace collapsed. "The Beatles", "Beatles" and
  "beatles." become one node; "U.S." -> "us"; "St. Louis" -> "st louis".
  No embedding-based entity resolution (blueprint §I, §S: a confound risk).
- Type clashes: one node per name; its `type` is the majority type (ties broken
  alphabetically), and `type_counts` keeps every type seen, for analysis.
- Descriptions: every distinct description is kept as a list. No LLM summarisation,
  so indexing cost stays exactly the extraction cost the budget measures.
- Only chunks with status "ok" contribute. Failed/truncated chunks are counted in
  the stats, never silently treated as "no entities".
- Unselected chunks are not indexed at all, for every strategy (no KET-RAG-style
  keyword graph), so selection stays the only variable.
- A relationship whose two ends merge into the same node is dropped and counted.

STORAGE: JSON (networkx node-link format) - portable and safe to share, unlike
pickle (networkx 3 also removed its gpickle helpers).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx

from extraction.schemas import ExtractionResult

_DROP = re.compile(r"[.'’`]")          # removed outright: "U.S." -> "US", "O'Neil" -> "ONeil"
_SPACE = re.compile(r"[^\w\s]")             # any other punctuation becomes a space
_LEADING_THE = re.compile(r"^the\s+")


def normalize_entity(name: str) -> str:
    """Node key for an entity name. Returns "" if nothing meaningful is left."""
    s = unicodedata.normalize("NFKC", name).lower()
    s = _DROP.sub("", s)
    s = _SPACE.sub(" ", s).replace("_", " ")
    s = " ".join(s.split())
    s = _LEADING_THE.sub("", s)
    return s


def _majority(counter: Counter) -> str:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def build_graph(results: Iterable[ExtractionResult]) -> tuple[nx.MultiDiGraph, dict]:
    """Build the knowledge graph for ONE condition. Returns (graph, build_stats).
    Deterministic: the same results in any order give the same graph."""
    results = sorted(results, key=lambda r: r.chunk_id)
    ok = [r for r in results if r.status == "ok"]

    names: dict[str, Counter] = {}          # key -> surface forms
    types: dict[str, Counter] = {}
    descriptions: dict[str, list[str]] = {}
    chunks_of: dict[str, set[str]] = {}
    edges: list[tuple[str, str, str, str]] = []
    dropped_empty_names = dropped_self_loops = 0

    for r in ok:
        key_of: dict[str, str] = {}
        for e in r.entities:
            key = normalize_entity(e.name)
            if not key:
                dropped_empty_names += 1
                continue
            key_of[e.name] = key
            names.setdefault(key, Counter())[e.name] += 1
            types.setdefault(key, Counter())[e.type] += 1
            descriptions.setdefault(key, [])
            chunks_of.setdefault(key, set()).add(r.chunk_id)
            desc = " ".join(e.description.split())
            if desc and desc not in descriptions.setdefault(key, []):
                descriptions[key].append(desc)
        for rel in r.relationships:
            s, t = key_of.get(rel.source), key_of.get(rel.target)
            if s is None or t is None:
                continue                     # endpoint had an empty normalised name
            if s == t:
                dropped_self_loops += 1      # e.g. "The Beatles" -> "Beatles" after merging
                continue
            edges.append((s, t, " ".join(rel.description.split()), r.chunk_id))

    G = nx.MultiDiGraph()
    for key in sorted(names):
        G.add_node(
            key,
            name=_majority(names[key]),          # most frequent surface form
            type=_majority(types[key]),
            type_counts=dict(sorted(types[key].items())),
            descriptions=descriptions[key],
            chunk_ids=sorted(chunks_of[key]),
            mentions=sum(names[key].values()),
        )
    for s, t, desc, chunk_id in sorted(set(edges)):
        G.add_edge(s, t, description=desc, chunk_id=chunk_id)

    status_counts = Counter(r.status for r in results)
    stats = {
        "n_results": len(results),
        "n_chunks_used": len(ok),
        "n_chunks_skipped_not_ok": len(results) - len(ok),
        "skipped_chunk_ids": [r.chunk_id for r in results if r.status != "ok"],
        "status_counts": dict(sorted(status_counts.items())),
        "dropped_empty_names": dropped_empty_names,
        "dropped_self_loops": dropped_self_loops,
        **graph_stats(G),
    }
    return G, stats


def graph_stats(G: nx.MultiDiGraph) -> dict:
    """Size/shape numbers for the graph-statistics table (blueprint §T) and sparsity risk (§S)."""
    n = G.number_of_nodes()
    components = list(nx.weakly_connected_components(G)) if n else []
    return {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "n_isolated_nodes": sum(1 for v in G if G.degree(v) == 0),
        "n_components": len(components),
        "largest_component_size": max((len(c) for c in components), default=0),
        "type_distribution": dict(sorted(Counter(d["type"] for _, d in G.nodes(data=True)).items())),
        "n_provenance_chunks": len({c for _, d in G.nodes(data=True) for c in d["chunk_ids"]}),
    }


def check_provenance(G: nx.MultiDiGraph, selected_chunk_ids: Iterable[str]) -> None:
    """Phase 8 checkpoint: no orphan provenance. Every node and edge must come from
    at least one chunk, and only from chunks that were budget-selected."""
    allowed = set(selected_chunk_ids)
    for v, d in G.nodes(data=True):
        if not d["chunk_ids"]:
            raise ValueError(f"node {v!r} has no provenance")
        outside = set(d["chunk_ids"]) - allowed
        if outside:
            raise ValueError(f"node {v!r} cites unselected chunks {sorted(outside)}")
    for u, v, d in G.edges(data=True):
        if d["chunk_id"] not in allowed:
            raise ValueError(f"edge {u!r}->{v!r} cites unselected chunk {d['chunk_id']!r}")
        if d["chunk_id"] not in G.nodes[u]["chunk_ids"] or d["chunk_id"] not in G.nodes[v]["chunk_ids"]:
            raise ValueError(f"edge {u!r}->{v!r} cites a chunk its endpoints were not extracted from")


def save_graph(G: nx.MultiDiGraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G, edges="edges")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")
