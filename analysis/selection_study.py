"""
Selection-only study: do the "smart" strategies pick the gold (evidence) chunks more often than Random?

FREE. No LLM is called. Every budgeted strategy ranks chunks without an LLM (Random: seeded shuffle; KET-RAG:
keyword + embedding chunk graph; LazyGraphRAG: k-means topics; FastGraphRAG: entity centrality), and the budget
cutoff is the benchmark's own (src.budget via extraction.pipeline.select_for_extraction). Gold chunk labels are used
ONLY here, for scoring, exactly like the runner's evidence recall.

Per question q and condition (strategy, budget):
  selection_recall_q = |gold_q ∩ selected| / |gold_q|        (primary)
  all_gold_q         = 1 if every gold chunk of q is selected  (secondary: a multi-hop question needs every hop)
Random's per-question value is the mean over its shuffles (one strategy's luck, not extra samples - same rule as
Phase 12). Each smart strategy is compared with Random at each budget by the paired sign-flip test
(analysis.stats), with a bootstrap CI of the mean paired difference; Holm correction over strategies x budgets,
per dataset and metric.

    python -m analysis.selection_study --datasets hotpotqa,musique --num-questions 500
    python -m analysis.selection_study --datasets hotpotqa --num-questions 30 --random-seeds 0,1,2   # = main run
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import yaml

from analysis.stats import bootstrap_mean_ci, derive_seed, holm_adjust, sign_flip_test
from extraction.pipeline import select_for_extraction
from src.config import ExperimentConfig
from src.corpus import PROJECT_ROOT
from src.prepare_data import build as build_corpus
from strategies import build_strategy

SMART = ("ketrag", "lazygraphrag", "fastgraphrag")
METRICS = ("selection_recall", "all_gold")


def _base_cfg(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.pop("matrix", None)
    return data


def per_question(gold: dict[str, set[str]], selected: set[str], qids: list[str]) -> dict[str, np.ndarray]:
    rec = np.array([len(gold[q] & selected) / len(gold[q]) for q in qids])
    return {"selection_recall": rec, "all_gold": (rec == 1.0).astype(float)}


def run_dataset(base: dict, dataset: str, n_questions: int, budgets: list[float], random_seeds: list[int],
                n_resamples: int, n_permutations: int, log) -> dict:
    cfg = ExperimentConfig(**{**base, "dataset": dataset, "num_questions": n_questions, "strategy": "random"})
    raw_chunks, questions, _docs = build_corpus(cfg)
    gold: dict[str, set[str]] = {}
    for c in raw_chunks:
        for qid in c.is_gold_for_question_ids:
            gold.setdefault(str(qid), set()).add(c.chunk_id)
    # the strategies never see the gold labels
    chunks = [dataclasses.replace(c, is_gold_for_question_ids=[]) for c in raw_chunks]
    qids = [str(q["question_id"]) for q in questions if gold.get(str(q["question_id"]))]
    log(f"{dataset}: {len(questions)} questions ({len(qids)} with gold chunks), {len(chunks)} chunks")

    rankings, seconds = {}, {}
    arms = [("random", s) for s in random_seeds] + [(s, None) for s in SMART]
    for strategy, shuffle in arms:
        t = time.perf_counter()
        scfg = ExperimentConfig(**{**cfg.model_dump(), "strategy": strategy, "random_shuffle_seed": shuffle})
        rankings[(strategy, shuffle)] = list(build_strategy(scfg).rank(chunks))
        seconds[(strategy, shuffle)] = time.perf_counter() - t
        log(f"  ranked {strategy}{'' if shuffle is None else f' (shuffle {shuffle})'} in {seconds[(strategy, shuffle)]:.1f}s")

    # per-question values: values[metric][strategy][budget] -> array over qids (Random averaged over shuffles)
    values = {m: {s: {} for s in ("random",) + SMART} for m in METRICS}
    random_spread = {m: {} for m in METRICS}
    for b in budgets:
        per_shuffle = []
        for strategy, shuffle in arms:
            sel = set(select_for_extraction(chunks, rankings[(strategy, shuffle)], b).selected_ids)
            pq = per_question(gold, sel, qids)
            if strategy == "random":
                per_shuffle.append(pq)
            else:
                for m in METRICS:
                    values[m][strategy][b] = pq[m]
        for m in METRICS:
            stack = np.stack([p[m] for p in per_shuffle])
            values[m]["random"][b] = stack.mean(axis=0)
            means = stack.mean(axis=1)
            random_spread[m][b] = {"min": float(means.min()), "max": float(means.max()), "sd": float(means.std(ddof=1))
                                   if len(means) > 1 else None}

    rows = []
    for m in METRICS:
        for s in ("random",) + SMART:
            for b in budgets:
                ci = bootstrap_mean_ci(values[m][s][b], n_resamples=n_resamples, seed=derive_seed(0, dataset, m, s, b))
                rows.append({"dataset": dataset, "metric": m, "strategy": s, "budget_pct": round(b * 100),
                             "n_questions": len(qids), "mean": ci.mean, "ci_low": ci.low, "ci_high": ci.high,
                             **({"random_shuffle_min": random_spread[m][b]["min"],
                                 "random_shuffle_max": random_spread[m][b]["max"]} if s == "random" else {})})
    comps = []
    for m in METRICS:
        family = []
        for s in SMART:
            for b in budgets:
                d = values[m][s][b] - values[m]["random"][b]
                seed = derive_seed(0, dataset, m, s, b, "test")
                test = sign_flip_test(d, n_permutations=n_permutations, seed=seed)
                ci = bootstrap_mean_ci(d, n_resamples=n_resamples, seed=seed)
                family.append({"dataset": dataset, "metric": m, "strategy": s, "vs": "random",
                               "budget_pct": round(b * 100), "n_questions": len(qids), "mean_diff": ci.mean,
                               "ci_low": ci.low, "ci_high": ci.high, "n_better": int((d > 1e-12).sum()),
                               "n_worse": int((d < -1e-12).sum()), "p_value": test.p_value,
                               "test_method": test.method})
        for row, ph in zip(family, holm_adjust([r["p_value"] for r in family])):
            row["p_holm"] = ph
            row["significant_holm_0.05"] = ph < 0.05
        comps.extend(family)
    return {"dataset": dataset, "n_questions": len(questions), "n_questions_with_gold": len(qids),
            "n_chunks": len(chunks), "mean_gold_chunks_per_question": float(np.mean([len(gold[q]) for q in qids])),
            "ranking_seconds": {f"{k[0]}{'' if k[1] is None else '_rs' + str(k[1])}": round(v, 2)
                                                                             for k, v in seconds.items()},
            "performance": rows, "comparisons": comps}


def write_outputs(out_dir: Path, results: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
    import csv
    perf = [r for res in results for r in res["performance"]]
    comp = [r for res in results for r in res["comparisons"]]
    for name, rows in (("performance", perf), ("comparisons", comp)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    lines = ["# Selection-only study (no LLM calls)", "",
             f"Budgets: {meta['budgets_pct']}; Random shuffles: {meta['random_seeds']}; "
             f"Holm over 3 strategies x {len(meta['budgets_pct'])} budgets per dataset and metric.", ""]
    for res in results:
        ds = res["dataset"]
        lines += [f"## {ds}: {res['n_questions_with_gold']} questions, {res['n_chunks']} chunks, "
                  f"{res['mean_gold_chunks_per_question']:.2f} gold chunks per question", ""]
        for m in METRICS:
            lines += [f"### {m} (mean [95% CI])", "",
                      "| strategy | " + " | ".join(f"{b}%" for b in meta["budgets_pct"]) + " |",
                      "|---|" + "---|" * len(meta["budgets_pct"])]
            for s in ("random",) + SMART:
                cells = [r for r in res["performance"] if r["metric"] == m and r["strategy"] == s]
                lines.append(f"| {s} | " + " | ".join(f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
                                                    for r in cells) + " |")
            lines += ["", f"Paired difference vs Random ({m}): mean [95% CI], Holm p", "",
                      "| strategy | " + " | ".join(f"{b}%" for b in meta["budgets_pct"]) + " |",
                      "|---|" + "---|" * len(meta["budgets_pct"])]
            for s in SMART:
                cells = [r for r in res["comparisons"] if r["metric"] == m and r["strategy"] == s]
                lines.append(f"| {s} | " + " | ".join(
                    f"{r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p={r['p_holm']:.3g}"
                    + ("*" if r["significant_holm_0.05"] else "") for r in cells) + " |")
            lines.append("")
    lines.append("`*` = significant after Holm at 0.05. Selection only: says nothing about answer quality.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(out_dir: Path, results: list[dict], meta: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 4.6), squeeze=False)
    styles = {"random": ("#1f6fd1", "o", "-"), "ketrag": ("#e8662f", "s", "--"),
              "lazygraphrag": ("#17a37a", "^", "-."), "fastgraphrag": ("#e0729f", "D", ":")}
    for ax, res in zip(axes[0], results):
        for s in ("random",) + SMART:
            rows = [r for r in res["performance"] if r["metric"] == "selection_recall" and r["strategy"] == s]
            x = [r["budget_pct"] for r in rows]
            c, mk, ls = styles[s]
            ax.plot(x, [r["mean"] for r in rows], color=c, marker=mk, ls=ls, label=s)
            ax.fill_between(x, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], color=c, alpha=0.12)
        ax.plot([0, 100], [0, 1], color="0.6", lw=0.8, ls=(0, (2, 3)), label="budget fraction (y = x)")
        ax.set(title=f"{res['dataset']}  (n = {res['n_questions_with_gold']} questions)", xlabel="extraction budget (%)",
               ylabel="selection recall of gold chunks", xlim=(0, 80), ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Selection-only study: share of each question's gold chunks inside the budget (no LLM)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"selection_recall.{ext}", dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/benchmark_1m_hotpotqa.yaml",
                    help="base config for chunking / strategy knobs (its matrix block is ignored)")
    ap.add_argument("--datasets", default="hotpotqa,musique")
    ap.add_argument("--num-questions", type=int, default=500)
    ap.add_argument("--budgets", default="0.05,0.10,0.25,0.50,0.75")
    ap.add_argument("--random-seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--n-permutations", type=int, default=20_000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    budgets = [float(b) for b in args.budgets.split(",")]
    seeds = [int(s) for s in args.random_seeds.split(",")]
    datasets = args.datasets.split(",")
    base = _base_cfg(PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config))
    out = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "results" / "selection_study" / f"{'_'.join(datasets)}_{args.num_questions}q"
    log = lambda m: print(m, flush=True)  # noqa: E731
    results = [run_dataset(base, ds, args.num_questions, budgets, seeds, args.n_resamples, args.n_permutations, log)
               for ds in datasets]
    meta = {"config": args.config, "num_questions": args.num_questions, "budgets_pct": [round(b * 100) for b in budgets],
            "random_seeds": seeds, "n_resamples": args.n_resamples, "n_permutations": args.n_permutations,
            "llm_calls": 0, "note": "selection only; gold labels used for scoring only"}
    write_outputs(out, results, meta)
    plot(out, results, meta)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
