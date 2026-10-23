"""
The machine-readable analysis summary (Phase 12, substep 4A).

`build_summary(analysis, evidence=None)` turns a loaded `AnalysisInput` into ONE `AnalysisSummary`: arm
descriptions, paired comparisons with their multiple-testing families, Random's shuffle variability, missing-data
accounting, cost / token / runtime means, optional evidence metrics, a 100%-budget anchor check and provenance.
`analysis.tables` renders it; `summary_to_json` serialises it.

DETERMINISM. The summary contains no timestamps, no absolute paths, no wall-clock values. All randomness comes from
`analysis.stats.derive_seed` (SHA-256 of the cell identity and `base_seed`). Same inputs + same settings + same
library versions give a byte-identical JSON document (tested). Library versions are recorded so a difference
between two machines can be traced.

MOCK DATA PRODUCES NO FINDINGS. When `analysis.status == "mock_layout_test"`:
  * the paired comparisons are listed (so a table layout can be checked) but every inferential field is None:
    no p-value, no confidence interval of a difference, no Holm / BH adjustment, no significance flag
    (`suppressed=True`); the statistical tests are not even run;
  * `headline.significant_primary_tests` is None and `headline.statement` says MOCK - NOT RESULTS;
  * `banner` carries the watermark text that tables and plots print.
Arm means and their bootstrap intervals are kept, because they are layout content, not claims.

WHAT IS CLAIMED. For real data the summary is mechanical: counts, means, intervals, p-values and which tests
pass Holm at alpha. It does not interpret them. It also cannot prove that a real API produced the answers
(see analysis/loader.py); the caveats of the loader are carried through verbatim.

HYPOTHESIS FAMILIES. One family = all pairwise strategy comparisons of one (dataset, metric, missing-policy)
over the budgets below the anchor budget (the pre-specified design gives 6 pairs x 5 budgets = 30 tests). Holm
(confirmatory) and Benjamini-Hochberg (exploratory) are applied inside a family only. F1 and judge accuracy are the
primary metrics; EM is secondary (its own families, labelled `secondary`). The 100% budget is not in any family:
it is a sanity-check anchor (`anchor_checks`), reported, never forced.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy
from pydantic import BaseModel, Field

from analysis import evidence as ev
from analysis import random_avg as ra
from analysis import stats as st
from analysis.schemas import (ANALYSIS_SCHEMA_VERSION, METRICS, PRIMARY_METRICS, AnalysisInput, AnalysisStatus)

SUMMARY_SCHEMA_VERSION = "1"
MOCK_BANNER = "MOCK — NOT RESULTS"
PARTIAL_BANNER = "PARTIAL RUN — not every planned condition is present"
ATTRITION_WARNING_FRACTION = 0.02
ANCHOR_BUDGET_PCT = 100
STRATEGY_ORDER = ("random", "ketrag", "lazygraphrag", "fastgraphrag")
COST_MEASURES = ("cost_total_usd", "tokens_total_input", "tokens_total_output", "runtime_total_s")
POLICIES = {"complete_case": "primary_complete_case", "as_wrong": "sensitivity_missing_as_wrong"}


# ============================================================================== records
class ArmRow(BaseModel):
    role: str                                   # primary | secondary | evidence
    analysis_note: str = ""
    attrition_fraction: float | None = None
    attrition_warning: bool = False
    arm: st.ArmDescription


class ComparisonRow(BaseModel):
    analysis: str                               # primary_complete_case | sensitivity_missing_as_wrong
    dataset: str
    metric: str
    role: str
    budget_pct: int
    strategy_a: str
    strategy_b: str
    n_universe: int = 0
    n_pairs: int = 0
    n_dropped: int = 0
    mean_a: float | None = None
    mean_b: float | None = None
    mean_diff: float | None = None              # A - B
    ci_low: float | None = None
    ci_high: float | None = None
    dz: float | None = None
    n_a_better: int | None = None
    n_b_better: int | None = None
    n_tied: int | None = None
    test_method: str | None = None
    p_value: float | None = None
    p_holm: float | None = None
    p_bh: float | None = None
    significant_holm: bool | None = None
    mcnemar_p: float | None = None
    flags: list[str] = Field(default_factory=list)
    suppressed: bool = False                    # mock data: inference deliberately not computed
    skipped_reason: str | None = None


class FamilyInfo(BaseModel):
    dataset: str
    metric: str
    role: str
    analysis: str
    n_tests: int                                # comparisons listed in the family
    n_tests_computed: int
    n_tests_skipped: int
    alpha: float
    correction: str = "holm (confirmatory) / benjamini-hochberg (exploratory)"
    inference_suppressed: bool = False
    complete: bool = True                       # no test skipped


class ShuffleVariability(BaseModel):
    dataset: str
    budget_pct: int
    metric: str
    n_shuffles: int
    per_shuffle: list[dict]
    mean: float | None
    sd: float | None
    min: float | None
    max: float | None


class CostRow(BaseModel):
    dataset: str
    strategy: str
    budget_pct: int
    measure: str
    n_records: int
    n_known: int
    n_lower_bound: int
    n_unknown: int
    mean: float | None = None
    min: float | None = None
    max: float | None = None


class AnchorCheck(BaseModel):
    dataset: str
    metric: str
    budget_pct: int
    strategy_a: str
    strategy_b: str
    n_pairs: int
    identical_outcomes: bool | None            # every paired difference exactly 0 (None: nothing to pair)
    selected_chunks_identical: bool | None     # None: a selection file is missing
    note: str = ""


class AbstentionRow(BaseModel):
    """How often a strategy's generator answered "not found" (the runner's `outcome == "abstained"`), per cell.
    Counting only: no test, no interval. Same definition as the runner's `abstentions.rate_of_scored`: abstained /
    questions whose generation did not fail (answered + abstained + empty). Failed generations and questions with no
    prediction are not in the denominator (they have no answer to be an abstention) and are counted separately.
    A Random cell has one run per shuffle: `rate` is the mean of the per-run rates, `rate_min` / `rate_max` their range;
    the n_* counts are sums over the runs."""
    dataset: str
    strategy: str
    budget_pct: int
    n_runs: int
    n_questions: int                 # questions in the dataset's universe (per run)
    n_generated: int                 # answered + abstained + empty, summed over runs
    n_abstained: int
    n_failed: int
    n_no_prediction: int
    n_runs_with_rate: int            # runs with at least one generated answer
    rate: float | None
    rate_min: float | None
    rate_max: float | None


class Headline(BaseModel):
    statement: str
    n_primary_tests: int | None = None
    significant_primary_tests: int | None = None   # None for mock: no significance claims


class Provenance(BaseModel):
    analysis_schema_version: str
    summary_schema_version: str
    git_commit: str | None
    matrix_dir_name: str
    matrix_ids: list[str]
    shared_config_fingerprints: list[str]
    question_seed: dict[str, int]
    source_files: list[dict]
    input_digest: str
    settings: dict
    versions: dict


class AnalysisSummary(BaseModel):
    summary_schema_version: str = SUMMARY_SCHEMA_VERSION
    status: AnalysisStatus
    data_kind: str
    banner: str | None
    caveats: list[str]
    warnings: list[str]
    headline: Headline
    design: dict
    metrics: dict
    skipped_metrics: dict[str, str]
    families: list[FamilyInfo]
    arms: list[ArmRow]
    comparisons: list[ComparisonRow]
    shuffle_variability: list[ShuffleVariability]
    cost: list[CostRow]
    evidence: list[ArmRow]
    evidence_issues: list[dict]
    anchor_checks: list[AnchorCheck]
    abstention: list[AbstentionRow] = Field(default_factory=list)
    provenance: Provenance


# ============================================================================== helpers
def read_git_commit(start: str | Path) -> str | None:
    """The commit hash of the checkout containing `start`, read from .git/HEAD WITHOUT running git or changing
    anything. None when there is no repository or it cannot be read."""
    try:
        p = Path(start).resolve()
        for d in [p, *p.parents]:
            g = d / ".git"
            if g.is_file():                                   # worktree / submodule: "gitdir: <path>"
                text = g.read_text().strip()
                if text.startswith("gitdir:"):
                    g = (d / text.split(":", 1)[1].strip()).resolve()
            if g.is_dir():
                head = (g / "HEAD").read_text().strip()
                if not head.startswith("ref:"):
                    return head or None
                ref = head.split(":", 1)[1].strip()
                f = g / ref
                if f.is_file():
                    return f.read_text().strip() or None
                packed = g / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text().splitlines():
                        if line.endswith(" " + ref) and not line.startswith(("#", "^")):
                            return line.split()[0]
                return None
    except OSError:
        return None
    return None


def _ordered(strategies: Iterable[str]) -> list[str]:
    s = set(strategies)
    return [x for x in STRATEGY_ORDER if x in s] + sorted(s - set(STRATEGY_ORDER))


def _role(metric: str) -> str:
    return "primary" if metric in PRIMARY_METRICS else "secondary"


def _arm_row(arm: ra.ArmScores, role: str, *, n_resamples: int, level: float, base_seed: int) -> ArmRow:
    d = st.describe_arm(arm, n_resamples=n_resamples, level=level, base_seed=base_seed)
    frac = None if arm.n_universe == 0 else round(arm.n_missing / arm.n_universe, 9)
    return ArmRow(role=role, attrition_fraction=frac, attrition_warning=bool(frac and frac > ATTRITION_WARNING_FRACTION),
                  arm=d)


def _comparison_row(policy: str, sample: ra.PairedSample, comp: st.PairedComparison | None, role: str,
                    *, suppressed: bool, skipped: str | None = None) -> ComparisonRow:
    base = dict(analysis=POLICIES[policy], dataset=sample.dataset, metric=sample.metric, role=role,
                budget_pct=sample.budget_pct, strategy_a=sample.strategy_a, strategy_b=sample.strategy_b,
                n_universe=sample.n_universe, n_pairs=sample.n_pairs, n_dropped=sample.n_dropped)
    if skipped is not None:
        return ComparisonRow(**base, skipped_reason=skipped, flags=["skipped"])
    if suppressed:                                   # mock: descriptive layout fields only
        eff = st.paired_effect(sample.a, sample.b)
        return ComparisonRow(**base, mean_a=eff.mean_a, mean_b=eff.mean_b, mean_diff=eff.mean_diff,
                             n_a_better=eff.n_a_better, n_b_better=eff.n_b_better, n_tied=eff.n_tied,
                             suppressed=True, flags=["inference_suppressed_mock"])
    assert comp is not None
    return ComparisonRow(
        **base, mean_a=comp.effect.mean_a, mean_b=comp.effect.mean_b, mean_diff=comp.effect.mean_diff,
        ci_low=None if comp.ci is None else comp.ci.low, ci_high=None if comp.ci is None else comp.ci.high,
        dz=comp.effect.dz, n_a_better=comp.effect.n_a_better, n_b_better=comp.effect.n_b_better,
        n_tied=comp.effect.n_tied, test_method=comp.sign_flip.method, p_value=comp.sign_flip.p_value,
        p_holm=comp.p_holm, p_bh=comp.p_bh, significant_holm=comp.significant_holm,
        mcnemar_p=None if comp.mcnemar is None else comp.mcnemar.p_value, flags=list(comp.flags))


def _abstention_rows(observations, design: dict, cells: dict) -> list[AbstentionRow]:
    """One row per (dataset, strategy, budget), counted from the per-question outcomes (see AbstentionRow)."""
    runs: dict[tuple, dict] = {}
    for o in observations:
        r = runs.setdefault((o.dataset, o.strategy, o.budget_pct, o.shuffle_seed),
                            {"generated": 0, "abstained": 0, "failed": 0, "no_prediction": 0})
        if o.outcome == "no_prediction":
            r["no_prediction"] += 1
        elif o.outcome == "failed":
            r["failed"] += 1
        else:
            r["generated"] += 1
            r["abstained"] += o.outcome == "abstained"
    rows = []
    for ds in sorted(design["datasets"]):
        for strat in design["datasets"][ds]["strategies"]:
            for b in sorted(cells[ds][strat]):
                per = [v for (d, s, bb, _sh), v in sorted(runs.items(), key=lambda kv: str(kv[0][3]))
                       if (d, s, bb) == (ds, strat, b)]
                rates = [v["abstained"] / v["generated"] for v in per if v["generated"]]
                rows.append(AbstentionRow(
                    dataset=ds, strategy=strat, budget_pct=b, n_runs=len(per),
                    n_questions=design["datasets"][ds]["n_questions"],
                    n_generated=sum(v["generated"] for v in per), n_abstained=sum(v["abstained"] for v in per),
                    n_failed=sum(v["failed"] for v in per), n_no_prediction=sum(v["no_prediction"] for v in per),
                    n_runs_with_rate=len(rates), rate=round(sum(rates) / len(rates), 9) if rates else None,
                    rate_min=round(min(rates), 9) if rates else None, rate_max=round(max(rates), 9) if rates else None))
    return rows


def _input_digest(files: list[dict]) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda f: f["path"]):
        h.update(f"{f['path']}\0{f['sha256']}\n".encode())
    return h.hexdigest()


# ============================================================================== the summary
def build_summary(analysis: AnalysisInput, evidence: ev.EvidenceReport | None = None, *, base_seed: int = 0,
                  n_resamples: int = st.DEFAULT_RESAMPLES, n_permutations: int = st.DEFAULT_PERMUTATIONS,
                  alpha: float = st.DEFAULT_ALPHA, level: float = st.DEFAULT_LEVEL,
                  git_commit: str | None = None, anchor_pct: int = ANCHOR_BUDGET_PCT) -> AnalysisSummary:
    mock = analysis.status == "mock_layout_test"
    banner = MOCK_BANNER if mock else (PARTIAL_BANNER if analysis.status == "real_partial" else None)
    obs = analysis.observations
    kw = dict(n_resamples=n_resamples, level=level, base_seed=base_seed)

    skipped: dict[str, str] = {}
    metrics = list(METRICS)
    if analysis.judge_enabled is False:
        metrics.remove("judge_accuracy")
        skipped["judge_accuracy"] = "the judge was disabled in the matrix"

    datasets = sorted(analysis.question_ids)
    cells: dict[str, dict[str, set[int]]] = {}
    for o in obs:
        cells.setdefault(o.dataset, {}).setdefault(o.strategy, set()).add(o.budget_pct)

    design = {"datasets": {}, "n_conditions_loaded": analysis.report.n_conditions_loaded,
              "n_included": analysis.report.n_included, "n_planned": analysis.report.n_planned,
              "matrix_complete": analysis.report.matrix_complete, "judge_enabled": analysis.judge_enabled,
              "excluded_conditions": len(analysis.excluded)}
    for ds in datasets:
        strategies = _ordered(cells.get(ds, {}))
        budgets = sorted({b for s in cells.get(ds, {}).values() for b in s})
        shuffles = sorted({o.shuffle_seed for o in obs if o.dataset == ds and o.strategy == "random"
                           and o.shuffle_seed is not None})
        design["datasets"][ds] = {"n_questions": len(analysis.question_ids[ds]), "strategies": strategies,
                                  "budgets_pct": budgets, "random_shuffle_seeds": shuffles,
                                  "question_seed": analysis.question_seed.get(ds)}

    arm_cache: dict[tuple, ra.ArmScores] = {}

    def arm(ds, strat, b, metric) -> ra.ArmScores:
        k = (ds, strat, b, metric)
        if k not in arm_cache:
            arm_cache[k] = ra.build_arm(obs, dataset=ds, strategy=strat, budget_pct=b, metric=metric,
                                        universe=analysis.question_ids[ds])
        return arm_cache[k]

    # ---- arms --------------------------------------------------------------------------------------
    arms: list[ArmRow] = []
    warnings: list[str] = []
    for ds in datasets:
        for metric in metrics:
            for strat in design["datasets"][ds]["strategies"]:
                for b in sorted(cells[ds][strat]):
                    row = _arm_row(arm(ds, strat, b, metric), _role(metric), n_resamples=n_resamples, level=level,
                                   base_seed=base_seed)
                    arms.append(row)
                    if row.attrition_warning:
                        warnings.append(f"attrition above {ATTRITION_WARNING_FRACTION:.0%}: {ds}/{strat}/{b}%/{metric} "
                                        f"has {row.arm.n_missing} of {row.arm.n_universe} questions without a value "
                                        f"({row.arm.missing_reasons}); the primary analysis is complete-case")

    # ---- comparisons and families ------------------------------------------------------------------
    comparisons: list[ComparisonRow] = []
    families: list[FamilyInfo] = []
    for ds in datasets:
        strategies = design["datasets"][ds]["strategies"]
        for metric in metrics:
            for policy in POLICIES:
                fam_rows: list[ComparisonRow] = []
                computed: list[tuple[int, st.PairedComparison]] = []   # (index in fam_rows, comparison)
                for b in design["datasets"][ds]["budgets_pct"]:
                    if b >= anchor_pct:
                        continue
                    for sa, sb in itertools.combinations(strategies, 2):
                        if b not in cells[ds][sa] or b not in cells[ds][sb]:
                            fam_rows.append(ComparisonRow(
                                analysis=POLICIES[policy], dataset=ds, metric=metric, role=_role(metric), budget_pct=b,
                                strategy_a=sa, strategy_b=sb, skipped_reason="condition missing for one strategy",
                                flags=["skipped"]))
                            continue
                        sample = ra.pair_arms(arm(ds, sa, b, metric), arm(ds, sb, b, metric), missing=policy)
                        if mock:
                            fam_rows.append(_comparison_row(policy, sample, None, _role(metric), suppressed=True))
                        elif sample.n_pairs == 0:
                            fam_rows.append(_comparison_row(policy, sample, None, _role(metric), suppressed=False,
                                                            skipped="no paired questions with values"))
                        else:
                            comp = st.compare_paired(sample, n_resamples=n_resamples, n_permutations=n_permutations,
                                                     level=level, base_seed=base_seed)
                            computed.append((len(fam_rows), comp))
                            fam_rows.append(_comparison_row(policy, sample, comp, _role(metric), suppressed=False))
                if computed:
                    adjusted = st.apply_corrections([c for _i, c in computed], alpha=alpha)
                    for (i, _c), adj in zip(computed, adjusted):
                        fam_rows[i] = fam_rows[i].model_copy(update={
                            "p_holm": adj.p_holm, "p_bh": adj.p_bh, "significant_holm": adj.significant_holm})
                comparisons.extend(fam_rows)
                n_skip = sum(1 for r in fam_rows if r.skipped_reason)
                families.append(FamilyInfo(
                    dataset=ds, metric=metric, role=_role(metric), analysis=POLICIES[policy], n_tests=len(fam_rows),
                    n_tests_computed=len(computed), n_tests_skipped=n_skip if not mock else 0, alpha=alpha,
                    inference_suppressed=mock, complete=(n_skip == 0)))
                if not mock and n_skip:
                    warnings.append(f"family {ds}/{metric}/{POLICIES[policy]}: {n_skip} of {len(fam_rows)} tests skipped; "
                                    "Holm was applied over the tests that could be computed")

    # ---- Random's own variability --------------------------------------------------------------------
    shuffle_rows: list[ShuffleVariability] = []
    for ds in datasets:
        if not design["datasets"][ds]["random_shuffle_seeds"]:
            continue
        for metric in metrics:
            for b in sorted(cells[ds].get("random", ())):
                s = ra.shuffle_summary(obs, dataset=ds, budget_pct=b, metric=metric, universe=analysis.question_ids[ds])
                shuffle_rows.append(ShuffleVariability(
                    dataset=ds, budget_pct=b, metric=metric, n_shuffles=s.n_shuffles, mean=s.mean, sd=s.sd, min=s.min,
                    max=s.max, per_shuffle=[{"shuffle_seed": sid, "mean": m, "n": n} for sid, m, n in s.per_shuffle]))

    # ---- cost / tokens / runtime ----------------------------------------------------------------------
    groups: dict[tuple, list] = {}
    for r in analysis.records:
        if r.included:
            groups.setdefault((r.dataset, r.strategy, r.budget_pct), []).append(r)
    cost: list[CostRow] = []
    for (ds, strat, b), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], _ordered([kv[0][1]]), kv[0][2])):
        for name in COST_MEASURES:
            ms = [r.measures.get(name) for r in recs]
            known = [m.value for m in ms if m is not None and m.status == "measured" and m.value is not None]
            lower = sum(1 for m in ms if m is not None and m.status == "lower_bound")
            cost.append(CostRow(
                dataset=ds, strategy=strat, budget_pct=b, measure=name, n_records=len(recs), n_known=len(known),
                n_lower_bound=lower, n_unknown=len(recs) - len(known) - lower,
                mean=round(float(np.mean(known)), 9) if known else None,
                min=min(known) if known else None, max=max(known) if known else None))

    # ---- evidence (optional) --------------------------------------------------------------------------
    evidence_rows: list[ArmRow] = []
    evidence_issues: list[dict] = []
    if evidence is not None:
        evidence_issues = [i.model_dump(mode="json") for i in evidence.issues]
        for ds in datasets:
            for metric in ev.EVIDENCE_METRICS:
                for strat in design["datasets"][ds]["strategies"]:
                    for b in sorted(cells[ds][strat]):
                        a = ev.evidence_arm(evidence.rows, dataset=ds, strategy=strat, budget_pct=b, metric=metric,
                                            universe=analysis.question_ids[ds])
                        evidence_rows.append(_arm_row(a, "evidence", n_resamples=n_resamples, level=level,
                                                      base_seed=base_seed))

    # ---- 100% anchor ----------------------------------------------------------------------------------
    anchors: list[AnchorCheck] = []
    for ds in datasets:
        strategies = [s for s in design["datasets"][ds]["strategies"] if anchor_pct in cells[ds][s]]
        sel = {s: [c.selected_chunk_ids for c in analysis.conditions
                   if c.dataset == ds and c.strategy == s and c.budget_pct == anchor_pct] for s in strategies}

        def sets(s):
            return None if any(x is None for x in sel[s]) else {frozenset(x) for x in sel[s]}
        for sa, sb in itertools.combinations(strategies, 2):
            ssa, ssb = sets(sa), sets(sb)
            same_sel = None if ssa is None or ssb is None else (len(ssa | ssb) == 1)
            for metric in metrics:
                sample = ra.pair_arms(arm(ds, sa, anchor_pct, metric), arm(ds, sb, anchor_pct, metric))
                # tolerance, not ==: averaging Random's identical shuffles leaves ~1e-16 float noise
                ident = None if sample.n_pairs == 0 else bool(np.all(np.abs(sample.d) <= 1e-12))
                note = ("only the selected chunks were verified; graph / context inputs are not checked here"
                        if same_sel else "selected chunks differ or are unknown: do not expect identical outcomes")
                if ident is False and same_sel:
                    note = "outcomes differ although the selected chunks are identical: investigate, do not force equality"
                anchors.append(AnchorCheck(dataset=ds, metric=metric, budget_pct=anchor_pct, strategy_a=sa,
                                           strategy_b=sb, n_pairs=sample.n_pairs, identical_outcomes=ident,
                                           selected_chunks_identical=same_sel, note=note))

    # ---- headline ---------------------------------------------------------------------------------------
    if mock:
        headline = Headline(statement=f"{MOCK_BANNER}. Mock data checks table and plot layouts only; "
                                      "no result, comparison or significance is reported.")
    else:
        prim = [c for c in comparisons if c.analysis == POLICIES["complete_case"] and c.role == "primary"
                and c.p_value is not None]
        sig = sum(1 for c in prim if c.significant_holm)
        lead = "PARTIAL RUN. " if analysis.status == "real_partial" else ""
        headline = Headline(
            statement=(f"{lead}{sig} of {len(prim)} primary paired comparisons (complete-case; F1 and judge accuracy) pass "
                       f"Holm at alpha={alpha}. This is a mechanical count, not an interpretation."),
            n_primary_tests=len(prim), significant_primary_tests=sig)

    # ---- provenance -------------------------------------------------------------------------------------
    files = [f.model_dump(mode="json") for f in analysis.source_files]
    prov = Provenance(
        analysis_schema_version=ANALYSIS_SCHEMA_VERSION, summary_schema_version=SUMMARY_SCHEMA_VERSION,
        git_commit=git_commit, matrix_dir_name=Path(analysis.matrix_dir).name, matrix_ids=list(analysis.matrix_ids),
        shared_config_fingerprints=list(analysis.report.shared_config_fingerprints),
        question_seed=dict(sorted(analysis.question_seed.items())), source_files=files, input_digest=_input_digest(files),
        settings={"base_seed": base_seed, "n_resamples": n_resamples, "n_permutations": n_permutations, "alpha": alpha,
                  "ci_level": level, "exact_max_n": st.DEFAULT_EXACT_MAX_N, "anchor_budget_pct": anchor_pct,
                  "attrition_warning_fraction": ATTRITION_WARNING_FRACTION},
        versions={"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__})

    return AnalysisSummary(
        status=analysis.status, data_kind=analysis.data_kind, banner=banner, caveats=list(analysis.caveats),
        warnings=warnings + [f"{i.code}: {i.message}" for i in analysis.issues], headline=headline, design=design,
        metrics={"analysed": metrics, "primary": [m for m in metrics if m in PRIMARY_METRICS],
                 "secondary": [m for m in metrics if m not in PRIMARY_METRICS],
                 "evidence": list(ev.EVIDENCE_METRICS) if evidence is not None else []},
        skipped_metrics=skipped, families=families, arms=arms, comparisons=comparisons,
        shuffle_variability=shuffle_rows, abstention=_abstention_rows(obs, design, cells), cost=cost, evidence=evidence_rows, evidence_issues=evidence_issues,
        anchor_checks=anchors, provenance=prov)


# ============================================================================== output
def summary_to_json(summary: AnalysisSummary) -> str:
    """Canonical JSON: sorted keys, fixed indent, trailing newline. Same summary -> same bytes."""
    return json.dumps(summary.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n"


def write_summary(summary: AnalysisSummary, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_to_json(summary), encoding="utf-8", newline="\n")
    return path
