"""
Evidence metrics (Phase 12, increment 3): did the pipeline pick and retrieve the chunks a question needs?

For one question q under one condition, with
    G = q's gold chunks     (every chunk of q's gold supporting paragraphs; from the corpus, POST HOC only)
    S = the chunks the strategy SELECTED for extraction (extraction/selection.json)
    R = the chunks whose text was in the retrieval CONTEXT (predictions.jsonl: retrieval.passage_chunk_ids)

the metrics are four DIFFERENT things and are never mixed:

  selection_recall              |G ∩ S| / |G|        did the STRATEGY choose the gold chunks?
  retrieval_recall              |G ∩ R| / |G|        UNCONDITIONAL: did the gold chunks reach the context?
                                                     (R ⊆ S, so this is never above selection_recall)
  conditional_retrieval_recall  |G ∩ R| / |G ∩ S|    given a gold chunk was selected, did RETRIEVAL return it?
                                                     (isolates the retriever from the strategy)
  all_gold_selected             1[G ⊆ S]             every gold chunk selected (a multi-hop question needs all)
  all_gold_retrieved            1[G ⊆ R]             every gold chunk in the context
  all_gold_retrieved_given_selected   1[G ⊆ R] when G ⊆ S

UNDEFINED IS NOT ZERO. A metric whose denominator or inputs do not exist is None with a reason, and the
question is left out of that metric's mean (the counts are reported next to every mean):
  empty_gold                 G is empty (no gold chunk is known): every metric
  no_selection_file          S unknown (selection.json missing): the metrics that need S
  no_prediction              the question has no retrieval (retrieval raised): the metrics that need R
  no_selected_gold           |G ∩ S| = 0: conditional recall has no denominator
  not_all_gold_selected      G ⊄ S: all_gold_retrieved_given_selected is conditioned on G ⊆ S
  retrieved_not_in_selected  R ⊄ S (an inconsistency): conditional metrics are not valid conditional rates

CONSISTENCY CHECKS (reported as issues, never silently corrected)
  * R ⊆ S for every question: retrieval can only return chunks that were extracted into the graph;
  * S and R ⊆ the corpus's chunks; the gold source's chunk-id fingerprint and count equal what the runner recorded
    (so gold is never read from the wrong corpus);
  * the runner's own stored evidence figures (selection_recall, ..., expected_random_selection_recall) are
    reproduced from the per-question values.

GOLD IS POST HOC ONLY. Gold chunk labels are read here, after every pipeline stage has finished, from the corpus
(data/corpus/<corpus_id>/chunks.json, or a rebuild of the same corpus). This module imports none of the
selection, extraction, graph, retrieval or generation code and nothing from here is read by them (tested). The
runner strips the labels before the pipeline runs (Phase 10); this module cannot influence a run.

What the stored data CANNOT give: whether the retrieved text actually contains the answer, entity- or fact-level
recall, and precision beyond chunk identity. Those would need new instrumentation and are not computed.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

from pydantic import BaseModel, Field

from aggregation.schemas import Issue
from analysis.random_avg import ArmScores
from analysis.schemas import AnalysisInput, QuestionObservation
from experiments.schemas import read_condition_result
from src.corpus import PROJECT_ROOT

EVIDENCE_METRICS = ("selection_recall", "retrieval_recall", "conditional_retrieval_recall",
                    "all_gold_selected", "all_gold_retrieved", "all_gold_retrieved_given_selected")
Undefined = Literal["empty_gold", "no_selection_file", "no_prediction", "no_selected_gold", "not_all_gold_selected",
                    "retrieved_not_in_selected"]
TOL = 1.5e-6          # the runner rounds its stored figures to 6 decimals


class EvidenceError(RuntimeError):
    """Evidence cannot be computed honestly (no / wrong gold source ...)."""


def _r(x: float) -> float:
    return round(float(x), 9)


# ============================================================================== the pure core
class QuestionEvidence(BaseModel):
    condition_id: str = ""
    dataset: str = ""
    strategy: str = ""
    budget_pct: int = 0
    seed: int = 0
    shuffle_seed: int | None = None
    question_id: str = ""

    n_gold: int = 0
    n_selected_gold: int | None = None            # |G ∩ S|
    n_retrieved_gold: int | None = None           # |G ∩ R|
    n_selected: int | None = None                 # |S|
    n_retrieved: int | None = None                # |R|
    retrieved_not_selected: tuple[str, ...] = ()  # R \ S: must be empty

    selection_recall: float | None = None
    retrieval_recall: float | None = None
    conditional_retrieval_recall: float | None = None
    all_gold_selected: float | None = None        # 1.0 / 0.0
    all_gold_retrieved: float | None = None
    all_gold_retrieved_given_selected: float | None = None
    undefined: dict[str, str] = Field(default_factory=dict)     # metric -> reason it has no value


def question_evidence(gold: Iterable[str], selected: Iterable[str] | None, retrieved: Iterable[str] | None) -> QuestionEvidence:
    """The metrics for one question from its gold set G, selected set S (None = unknown) and retrieved set R
    (None = no retrieval happened). Pure: no I/O, no condition metadata."""
    G = frozenset(gold)
    S = None if selected is None else frozenset(selected)
    R = None if retrieved is None else frozenset(retrieved)
    out = QuestionEvidence(n_gold=len(G), n_selected=None if S is None else len(S),
                           n_retrieved=None if R is None else len(R))
    und: dict[str, str] = {}
    if S is not None and R is not None:
        out.retrieved_not_selected = tuple(sorted(R - S))
    if not G:
        out.undefined = {m: "empty_gold" for m in EVIDENCE_METRICS}
        return out
    n = len(G)

    # --- selection (needs S)
    if S is None:
        for m in ("selection_recall", "all_gold_selected"):
            und[m] = "no_selection_file"
    else:
        out.n_selected_gold = len(G & S)
        out.selection_recall = out.n_selected_gold / n
        out.all_gold_selected = 1.0 if G <= S else 0.0
    # --- unconditional retrieval (needs R only)
    if R is None:
        for m in ("retrieval_recall", "all_gold_retrieved"):
            und[m] = "no_prediction"
    else:
        out.n_retrieved_gold = len(G & R)
        out.retrieval_recall = out.n_retrieved_gold / n
        out.all_gold_retrieved = 1.0 if G <= R else 0.0
    # --- conditional metrics (need S and R, and a valid conditioning)
    cond_metrics = ("conditional_retrieval_recall", "all_gold_retrieved_given_selected")
    if S is None:
        und.update({m: "no_selection_file" for m in cond_metrics})
    elif R is None:
        und.update({m: "no_prediction" for m in cond_metrics})
    elif out.retrieved_not_selected:
        und.update({m: "retrieved_not_in_selected" for m in cond_metrics})
    else:
        if out.n_selected_gold:
            out.conditional_retrieval_recall = out.n_retrieved_gold / out.n_selected_gold
        else:
            und["conditional_retrieval_recall"] = "no_selected_gold"
        if G <= S:
            out.all_gold_retrieved_given_selected = 1.0 if G <= R else 0.0
        else:
            und["all_gold_retrieved_given_selected"] = "not_all_gold_selected"
    out.undefined = und
    return out


# ============================================================================== gold sources
@dataclass(frozen=True)
class GoldSource:
    """Gold chunk ids per question for one corpus, with the chunk ids needed to verify it is the right corpus."""
    gold: dict[str, frozenset[str]]       # question_id -> gold chunk ids (empty set if none)
    chunk_ids: tuple[str, ...]            # ALL chunk ids of the corpus, in corpus order
    origin: str                           # where it came from (provenance)

    @property
    def chunk_set(self) -> frozenset[str]:
        return frozenset(self.chunk_ids)


def gold_from_chunks(chunks: Iterable, origin: str = "chunks") -> GoldSource:
    """From Chunk objects or their dicts (src.chunking.Chunk / chunks.json rows) carrying is_gold_for_question_ids."""
    gold: dict[str, set[str]] = {}
    ids: list[str] = []
    for c in chunks:
        cid = c["chunk_id"] if isinstance(c, dict) else c.chunk_id
        labels = c.get("is_gold_for_question_ids", []) if isinstance(c, dict) else c.is_gold_for_question_ids
        ids.append(str(cid))
        for qid in labels or []:
            gold.setdefault(str(qid), set()).add(str(cid))
    if len(set(ids)) != len(ids):
        raise EvidenceError("the corpus has duplicate chunk ids")
    return GoldSource(gold={q: frozenset(v) for q, v in gold.items()}, chunk_ids=tuple(ids), origin=origin)


def gold_from_corpus_dir(corpus_dir: str | Path) -> GoldSource:
    """From data/corpus/<corpus_id>/chunks.json (written by src.prepare_data)."""
    path = Path(corpus_dir) / "chunks.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EvidenceError(f"cannot read {path}: {e}") from e
    return gold_from_chunks(rows, origin=str(path))


def _fingerprint(chunk_ids: Sequence[str]) -> str:
    from src.prepare_data import fingerprint
    return fingerprint(list(chunk_ids))


def find_gold_source(analysis: AnalysisInput, dataset: str, *, corpus_root: str | Path | None = None,
                     allow_rebuild: bool = False, corpus_builder: Callable | None = None) -> GoldSource:
    """The gold source of one dataset: data/corpus/<corpus_id>/chunks.json if it exists; otherwise, ONLY with
    allow_rebuild, a rebuild of the corpus from the condition's saved config with the project's own corpus builder
    (`corpus_builder(cfg) -> (chunks, questions, documents)`, default src.prepare_data.build). A rebuild with
    data_source huggingface uses the project's loader, which downloads the dataset if it is not cached yet."""
    conds = [c for c in analysis.conditions if c.dataset == dataset]
    if not conds:
        raise EvidenceError(f"no conditions for dataset {dataset!r}")
    corpus_id = conds[0].corpus_id
    root = Path(corpus_root) if corpus_root is not None else PROJECT_ROOT / "data" / "corpus"
    if corpus_id and (root / corpus_id / "chunks.json").exists():
        return gold_from_corpus_dir(root / corpus_id)
    if not allow_rebuild:
        raise EvidenceError(
            f"gold chunk labels for {dataset} (corpus {corpus_id!r}) were not found at {root / str(corpus_id) / 'chunks.json'}. "
            f"Run `python -m src.prepare_data --config <config>` to write them, or pass allow_rebuild=True to rebuild "
            f"the corpus from the saved config (a huggingface rebuild may download the dataset).")
    rec = next(r for r in analysis.records if r.condition_id == conds[0].condition_id)
    from src.config import ExperimentConfig
    from src.prepare_data import build
    try:
        cfg = ExperimentConfig(**rec.config)
        chunks, _questions, _docs = (corpus_builder or build)(cfg)
    except Exception as e:                                     # a rebuild failure must not look like "no gold"
        raise EvidenceError(f"rebuilding the corpus for {dataset} failed: {type(e).__name__}: {e}") from e
    return gold_from_chunks(chunks, origin=f"rebuilt from the saved config of {conds[0].condition_id}")


# ============================================================================== per condition
class MetricSummary(BaseModel):
    mean: float | None
    n_defined: int
    n_undefined: int
    undefined_reasons: dict[str, int] = Field(default_factory=dict)


class ConditionEvidence(BaseModel):
    condition_id: str
    dataset: str
    strategy: str
    budget_pct: int
    shuffle_seed: int | None = None
    n_questions: int
    n_empty_gold: int
    n_selected: int | None
    n_chunks_total: int
    n_questions_retrieved_not_selected: int = 0       # questions with R ⊄ S (must be 0)
    metrics: dict[str, MetricSummary]


def _summarise(rows: list[QuestionEvidence]) -> dict[str, MetricSummary]:
    out = {}
    for m in EVIDENCE_METRICS:
        vals = [getattr(r, m) for r in rows if getattr(r, m) is not None]
        reasons = Counter(r.undefined[m] for r in rows if m in r.undefined)
        out[m] = MetricSummary(mean=_r(math.fsum(vals) / len(vals)) if vals else None, n_defined=len(vals),
                               n_undefined=len(rows) - len(vals), undefined_reasons=dict(sorted(reasons.items())))
    return out


class EvidenceReport(BaseModel):
    status: str                                       # the analysis status this was computed under
    gold_origin: dict[str, str]
    rows: list[QuestionEvidence]
    conditions: list[ConditionEvidence]
    issues: list[Issue] = Field(default_factory=list)
    reconciled: bool                                  # no error issue: R ⊆ S, corpus checks and the runner's figures agree
    n_subset_violations: int = 0                      # questions with R ⊄ S, over all conditions


def _issue(code, severity, cid, message, expected=None, actual=None) -> Issue:
    return Issue(code=code, severity=severity, condition_id=cid, message=message, expected=expected, actual=actual)


def reconcile(ce: ConditionEvidence, stored: dict, n_selected: int | None, n_chunks_total: int | None) -> list[Issue]:
    """Compare the figures recomputed from per-question data with what the runner stored in condition_result.evidence."""
    cid, out = ce.condition_id, []
    m = ce.metrics
    pairs = {
        "selection_recall": m["selection_recall"].mean, "selection_all_gold_rate": m["all_gold_selected"].mean,
        "selection_n_questions": m["selection_recall"].n_defined, "retrieval_recall": m["retrieval_recall"].mean,
        "retrieval_all_gold_rate": m["all_gold_retrieved"].mean, "retrieval_n_questions": m["retrieval_recall"].n_defined,
    }
    if n_selected is not None and n_chunks_total:
        pairs["expected_random_selection_recall"] = n_selected / n_chunks_total
    if not stored:
        return [_issue("no_stored_evidence", "warning", cid,
                       "the runner stored no evidence figures for this condition: nothing to reconcile with")]
    for key, mine in pairs.items():
        if key not in stored:
            out.append(_issue("no_stored_evidence", "warning", cid, f"the runner did not store {key!r}"))
            continue
        theirs = stored[key]
        same = (mine is None and theirs is None) or (mine is not None and theirs is not None and abs(mine - theirs) <= TOL)
        if not same:
            out.append(_issue("evidence_mismatch", "error", cid,
                              f"{key}: recomputed {mine!r} but the runner stored {theirs!r}", expected=theirs, actual=mine))
    return out


def compute_evidence(analysis: AnalysisInput, *, gold_sources: dict[str, GoldSource] | None = None,
                     corpus_root: str | Path | None = None, allow_rebuild: bool = False,
                     corpus_builder: Callable | None = None) -> EvidenceReport:
    """Per-question and per-condition evidence metrics for every included condition of `analysis`.
    `gold_sources` ({dataset: GoldSource}) overrides the lookup (tests, or gold from elsewhere)."""
    gold_sources = dict(gold_sources or {})
    rec_by_id = {r.condition_id: r for r in analysis.records}
    obs_by_cond: dict[str, list[QuestionObservation]] = {}
    for o in analysis.observations:
        obs_by_cond.setdefault(o.condition_id, []).append(o)

    issues: list[Issue] = []
    origins: dict[str, str] = {}
    for dataset in sorted({c.dataset for c in analysis.conditions}):
        if dataset not in gold_sources:
            gold_sources[dataset] = find_gold_source(analysis, dataset, corpus_root=corpus_root,
                                                     allow_rebuild=allow_rebuild, corpus_builder=corpus_builder)
        origins[dataset] = gold_sources[dataset].origin

    rows: list[QuestionEvidence] = []
    conditions: list[ConditionEvidence] = []
    n_violations = 0
    for cond in analysis.conditions:
        gs = gold_sources[cond.dataset]
        rec = rec_by_id[cond.condition_id]
        cid = cond.condition_id
        # --- is this the corpus the run used?
        n_total = rec.counts.get("n_chunks_total")
        if n_total is not None and n_total != len(gs.chunk_ids):
            raise EvidenceError(f"{cid}: the gold source has {len(gs.chunk_ids)} chunks but the run used {n_total}: "
                                f"wrong corpus ({gs.origin})")
        stored_fp = None
        result = read_condition_result(Path(analysis.matrix_dir) / cond.directory)
        if result is not None and "corpus" in result.stages:
            stored_fp = result.stages["corpus"].details.get("all_chunks_fingerprint")
        if stored_fp is None:
            issues.append(_issue("no_corpus_fingerprint", "warning", cid,
                                 "the runner stored no corpus fingerprint: the gold source cannot be matched to the run"))
        elif stored_fp != _fingerprint(gs.chunk_ids):
            raise EvidenceError(f"{cid}: the gold source's chunk-id fingerprint {_fingerprint(gs.chunk_ids)} differs from "
                                f"the run's {stored_fp}: wrong corpus ({gs.origin})")
        corpus = gs.chunk_set
        selected = None if cond.selected_chunk_ids is None else frozenset(cond.selected_chunk_ids)
        if selected is not None and not selected <= corpus:
            issues.append(_issue("selected_not_in_corpus", "error", cid,
                                 f"{len(selected - corpus)} selected chunk id(s) are not in the corpus"))
        unknown_q = [q for q in cond.question_ids if q not in gs.gold]
        if unknown_q and len(unknown_q) == len(cond.question_ids):
            raise EvidenceError(f"{cid}: none of the condition's questions has gold chunks in the gold source: wrong corpus")

        cond_rows = []
        for o in sorted(obs_by_cond.get(cid, []), key=lambda x: x.question_id):
            retrieved = None if o.outcome == "no_prediction" else frozenset(o.retrieved_chunk_ids)
            if retrieved is not None and not retrieved <= corpus:
                issues.append(_issue("retrieved_not_in_corpus", "error", cid,
                                     f"question {o.question_id}: {len(retrieved - corpus)} retrieved chunk id(s) are not in the corpus"))
            q = question_evidence(gs.gold.get(o.question_id, ()), selected, retrieved)
            q.condition_id, q.dataset, q.strategy, q.budget_pct = cid, o.dataset, o.strategy, o.budget_pct
            q.seed, q.shuffle_seed, q.question_id = o.seed, o.shuffle_seed, o.question_id
            cond_rows.append(q)
        n_bad = sum(1 for q in cond_rows if q.retrieved_not_selected)
        n_violations += n_bad
        if n_bad:
            ex = next(q for q in cond_rows if q.retrieved_not_selected)
            issues.append(_issue("retrieved_not_selected", "error", cid,
                                 f"{n_bad} question(s) have retrieved chunks that were never selected (retrieval can only "
                                 f"return extracted chunks); e.g. {ex.question_id}: {list(ex.retrieved_not_selected)[:3]}"))
        ce = ConditionEvidence(
            condition_id=cid, dataset=cond.dataset, strategy=cond.strategy, budget_pct=cond.budget_pct,
            shuffle_seed=cond.shuffle_seed, n_questions=len(cond_rows),
            n_empty_gold=sum(1 for q in cond_rows if q.n_gold == 0),
            n_selected=None if selected is None else len(selected), n_chunks_total=len(gs.chunk_ids),
            n_questions_retrieved_not_selected=n_bad, metrics=_summarise(cond_rows))
        issues += reconcile(ce, rec.evidence, ce.n_selected, ce.n_chunks_total)
        rows += cond_rows
        conditions.append(ce)
    return EvidenceReport(status=analysis.status, gold_origin=origins, rows=rows, conditions=conditions, issues=issues,
                          reconciled=not any(i.severity == "error" for i in issues), n_subset_violations=n_violations)


# ============================================================================== arms (shuffle-aware)
def evidence_arm(rows: Iterable[QuestionEvidence], *, dataset: str, strategy: str, budget_pct: int, metric: str,
                 universe: Sequence[str], strict: bool = False, shuffles: Sequence[int] | None = None) -> ArmScores:
    """One value per question for an evidence metric, with the same shuffle rules as random_avg.build_arm: Random's
    value is the mean over the shuffles where the metric is defined; the sample size is the number of QUESTIONS.
    The result is an ArmScores, so analysis.stats (describe_arm, pair_arms, compare_paired) work on it unchanged.
    Questions with no gold are undefined, not zero. (The averaging mirrors random_avg.build_arm; it is not shared
    code, and both are tested against hand-calculated cases.)"""
    from analysis.random_avg import ArmError
    if metric not in EVIDENCE_METRICS:
        raise ArmError(f"unknown evidence metric {metric!r}; expected one of {EVIDENCE_METRICS}")
    universe = tuple(universe)
    uni = set(universe)
    cell = [r for r in rows if r.dataset == dataset and r.strategy == strategy and r.budget_pct == budget_pct]
    if not cell:
        raise ArmError(f"no evidence rows for {dataset}/{strategy}/{budget_pct}%")
    ids = sorted({r.shuffle_seed for r in cell}, key=lambda s: (s is not None, s))
    if strategy != "random" and ids != [None]:
        raise ArmError(f"{strategy} is deterministic but has shuffle seeds {ids}; only Random is repeated")
    if None in ids and len(ids) > 1:
        raise ArmError(f"{strategy} mixes runs with and without a shuffle seed ({ids})")
    if shuffles is not None:
        if strategy != "random":
            raise ArmError("`shuffles` applies to Random only")
        ids = [s for s in ids if s in set(shuffles)]
        if not ids:
            raise ArmError(f"none of the requested shuffles {sorted(set(shuffles))} exist")
        cell = [r for r in cell if r.shuffle_seed in set(ids)]
    by: dict[tuple, QuestionEvidence] = {}
    for r in cell:
        if r.question_id not in uni:
            raise ArmError(f"question {r.question_id!r} is not in the dataset's question set")
        if (r.shuffle_seed, r.question_id) in by:
            raise ArmError(f"question {r.question_id!r} appears twice for shuffle {r.shuffle_seed}")
        by[(r.shuffle_seed, r.question_id)] = r
    values, missing, k_avail = {}, {}, {}
    for q in universe:
        got, reasons = [], []
        for sid in ids:
            r = by.get((sid, q))
            if r is None:
                reasons.append("no_observation")
            elif getattr(r, metric) is None:
                reasons.append(r.undefined.get(metric, "undefined"))
            else:
                got.append(getattr(r, metric))
        k_avail[q] = len(got)
        if got and not (strict and reasons):
            values[q] = math.fsum(got) / len(got)
        else:
            missing[q] = "incomplete_shuffles" if got else "|".join(sorted(set(reasons)))
    return ArmScores(dataset=dataset, strategy=strategy, budget_pct=budget_pct, metric=metric, universe=universe,
                     values=values, missing=missing, replicates=tuple(ids), k_available=k_avail, strict=strict)
