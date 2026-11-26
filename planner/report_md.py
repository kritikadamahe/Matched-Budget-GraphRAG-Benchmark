"""A planner report (dict) as a Markdown document to attach to a decision or a ticket."""

from __future__ import annotations


def _money(v) -> str:
    return "-" if v is None else f"${v:,.2f}" if v >= 0.1 else f"${v:,.4f}"


def to_markdown(r: dict) -> str:
    """r as it comes back from JSON (budget keys are strings)."""
    rec, c = r["recommendation"], r["corpus"]
    names, p = r["names"], r["primary"]
    lines = [
        f"# GraphRAG budget plan: {r['title']}",
        "",
        f"*Generated {r['created_utc']} · model `{r['model']}` at ${r['price_in']}/{r['price_out']} per 1M input/output tokens · "
        f"report type: **{r['kind']}***",
        "",
        "## Recommendation",
        "",
    ]
    if rec["status"] == "ok":
        lines += [
            f"**Extract {rec['budget_pct']}% of the corpus** ({_money(rec['usd'])} projected, vs {_money(rec['full_usd'])} for everything): "
            f"keeps about **{round(100 * rec['retained'])}%** of full-extraction answer quality "
            f"(F1 {rec['f1']:.3f} vs {rec['full_f1']:.3f}; target {round(100 * rec['target'])}%).",
            "",
        ]
    else:
        lines += ["**Inconclusive:** do not size a budget from this run.", ""]
    lines += [f"- {v}" for v in rec["verdicts"]] + [""]
    lines += [
        "## Corpus",
        "",
        f"{c['n_documents']:,} documents · {c['n_chunks']:,} chunks · {c['n_words']:,} words · pilot sample {c['sample_chunks']:,} chunks · "
        f"{r['n_questions']} test questions.",
        "",
        "## Budget table (full-corpus projection)",
        "",
        "| Budget | Chunks | Indexing cost | " + " | ".join(f"F1 {names[s]}" for s in r["strategies"]) + " | Evidence selected | Evidence retrieved |",
        "|---|---|---|" + "---|" * len(r["strategies"]) + "---|---|",
    ]
    cost = {row["budget_pct"]: row for row in r["cost_projection"]}
    for b in r["budgets"]:
        q = r["quality"]
        cells = [f"{q[s][str(b)]['mean']:.3f}" for s in r["strategies"]]
        ev = r["evidence"][p][str(b)]
        lines.append(f"| {b}% | {cost[b]['n_chunks']:,} | {_money(cost[b]['usd'])} | " + " | ".join(cells)
                     + f" | {ev['selected']:.0%} | {ev['retrieved']:.0%} |")
    if r.get("vector"):
        lines += ["", f"Vector-RAG baseline (no LLM extraction): F1 {r['vector']['mean']:.3f} "
                      f"[{r['vector']['low']:.3f}, {r['vector']['high']:.3f}]."]
    lines += ["", f"Recommendation and evidence columns use the **{names[p]}** selector. F1 is token-level, SQuAD-normalised.", ""]
    lines += ["## Method", "",
              "Matched-budget protocol: every condition uses the same chunks, extractor, graph builder, PageRank retriever and "
              "answer generator; only the share of chunks sent to the LLM extractor (and the selector that picks them) changes. "
              "The smallest budget whose mean F1 reaches the target share of full-extraction F1 is recommended (linear "
              "interpolation between measured budgets, rounded up to 5%).",
              "",
              f"Cost calibration: {r['calibration']['source']}. Per chunk: input ≈ {r['calibration']['in_base']:.0f} + "
              f"{r['calibration']['in_per_word']:.2f}×words tokens, output ≈ {r['calibration']['out_base']:.0f} + "
              f"{r['calibration']['out_per_word']:.2f}×words tokens.", ""]
    sp = r.get("pilot_spend") or {}
    if sp.get("usd") is not None:
        lines += [f"Pilot spend: {_money(sp['usd'])} (estimate {_money(sp.get('estimate_usd'))}, cap {_money(sp.get('cap_usd'))}), "
                  f"{sp.get('seconds')} s.", ""]
    lines += ["## Caveats", ""] + [f"- {x}" for x in r["caveats"]] + [""]
    return "\n".join(lines)
