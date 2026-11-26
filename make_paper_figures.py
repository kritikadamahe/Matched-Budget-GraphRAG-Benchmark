"""Regenerate paper Figs 2-4 from the saved result CSVs (Fig 1 is a schematic, no data).

Run from the repo root:  python3 make_paper_figures.py
Outputs: paper_figures/fig2_selection_recall.{pdf,png}, fig3_f1_vs_budget, fig4_loss_analysis
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SEL = ROOT / "results/selection_study/hotpotqa_musique_500q"
TAB = ROOT / "results/experiments/benchmark_75q_hotpotqa/analysis/real/tables"
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)

# internal name -> (paper label, colour, marker, linestyle)
STY = {
    "random":       ("Random",         "#44546a", "o", "-"),
    "ketrag":       ("KET-RAG",        "#e8710a", "s", "--"),
    "lazygraphrag": ("Topic coverage", "#1a9a6c", "^", "-."),
    "fastgraphrag": ("Entity centrality", "#d65db1", "D", ":"),
}
SELECTORS = ["ketrag", "lazygraphrag", "fastgraphrag"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- Fig 2: selection recall gain over Random (500 q/dataset) -> comparisons.csv
cmp_ = pd.read_csv(SEL / "comparisons.csv")
cmp_ = cmp_[(cmp_.metric == "selection_recall") & (cmp_.vs == "random")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, ds, title in zip(axes, ["hotpotqa", "musique"], ["HotpotQA", "MuSiQue"]):
    d = cmp_[cmp_.dataset == ds]
    budgets = sorted(d.budget_pct.unique())
    xs = range(len(budgets))
    for k, s in enumerate(SELECTORS):
        lab, c, m, ls = STY[s]
        r = d[d.strategy == s].sort_values("budget_pct")
        x = [i + (k - 1) * 0.08 for i in xs]          # small sideways shift
        y = r.mean_diff * 100
        ax.errorbar(x, y, yerr=[y - r.ci_low * 100, r.ci_high * 100 - y], color=c,
                    marker=m, ls=ls, capsize=2, lw=1.2, ms=4, label=lab)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{b}%" for b in budgets])
    ax.set_title(f"{title} ($M$=500)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Extraction budget")
axes[0].set_ylabel("Gain in selection recall\nover Random (points)")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.28), ncol=3, frameon=False, fontsize=8)
save(fig, "fig2_selection_recall")

# ---- Fig 3: end-to-end token F1 vs budget (75 q) -> performance.csv
perf = pd.read_csv(TAB / "performance.csv")
perf = perf[perf.metric == "f1"]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for k, s in enumerate(STY):
    lab, c, m, ls = STY[s]
    r = perf[perf.strategy == s].sort_values("budget_pct")
    x = [str(b) for b in r.budget_pct]
    xs = [list(sorted(perf.budget_pct.unique())).index(b) + (k - 1.5) * 0.04 for b in r.budget_pct]
    ax.errorbar(xs, r["mean"], yerr=[r["mean"] - r.ci_low, r.ci_high - r["mean"]],
                color=c, marker=m, ls=ls, capsize=2, lw=1.2, ms=4, label=lab)
bs = sorted(perf.budget_pct.unique())
ax.set_xticks(range(len(bs))); ax.set_xticklabels([f"{b}%" for b in bs])
ax.set_xlabel("Extraction budget"); ax.set_ylabel("Token F1"); ax.set_ylim(bottom=0)
ax.legend(fontsize=7, frameon=True, loc="upper left")
save(fig, "fig3_f1_vs_budget")

# ---- Fig 4: where quality is lost, 25% and 50% budgets -> evidence.csv + performance.csv
ev = pd.read_csv(TAB / "evidence.csv")
def get(df, metric, s, b):
    return df[(df.metric == metric) & (df.strategy == s) & (df.budget_pct == b)]["mean"].iloc[0]
order = ["random", "ketrag", "lazygraphrag", "fastgraphrag"]
short = ["Rand.", "KET", "Topic", "Entity"]
series = [("Selected", "selection_recall", ev, "#0b5aa8"),
          ("Retrieved", "retrieval_recall", ev, "#5aa9e6"),
          ("Answer F1", "f1", perf, "#f0a30a")]
fig, ax = plt.subplots(figsize=(3.6, 2.8))
w, gap = 0.26, 5
for pi, b in enumerate([25, 50]):
    for si, (lab, met, df, col) in enumerate(series):
        xs = [pi * (len(order) + 1) + i + (si - 1) * w for i in range(len(order))]
        ax.bar(xs, [get(df, met, s, b) for s in order], w, color=col,
               label=lab if pi == 0 else None)
    ax.text(pi * (len(order) + 1) + 1.5, 0.66, f"{b}% budget", ha="center", style="italic", fontsize=8)
ticks = [pi * (len(order) + 1) + i for pi in range(2) for i in range(len(order))]
ax.set_xticks(ticks); ax.set_xticklabels(short * 2, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Share or score"); ax.set_ylim(0, 0.72)
ax.legend(ncol=3, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "fig4_loss_analysis")
print("saved to", OUT)
