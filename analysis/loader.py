"""
Load a finished Phase 10 matrix for analysis (Phase 12): the gate, then the per-question table.

    load_analysis_input(matrix_dir) -> AnalysisInput

THE GATE (nothing is analysed until it has passed; every refusal says why)
  1. The Phase 11 aggregation runs first (aggregation.aggregate.aggregate_matrices). Duplicate cells,
     mixed configurations and unreadable results stop here exactly as they do in Phase 11.
  2. MOCK vs REAL. A condition shows mock evidence if ANY of these holds: the runner flagged it is_mock;
     its data_source or any enabled backend in the saved config is "mock"; any prediction was answered by
     a mock backend; any score came from a mock judge. A matrix is
         all conditions mock    -> "mock": refused unless allow_mock_layout_test, and then it is only a
                                   layout test (status mock_layout_test, never findings)
         some mock, some not    -> refused ALWAYS (mock and real results are never combined)
         none mock              -> "real": must be reportable (see 3)
  3. REAL data must be reportable: the Phase 11 report is reportable, has no error issues, and every
     condition reconciled. The one exception is completeness: with allow_incomplete a matrix that is
     missing planned conditions (or has no matrix.json to prove completeness) is accepted as
     status real_partial. Any other reason still refuses.
  4. ALIGNMENT. Per dataset: one question seed, one corpus, and the SAME set of question ids in every
     included condition (a condition's universe is its predicted questions plus the questions that have
     no prediction because retrieval raised). Without that a paired comparison is meaningless.
  5. INTEGRITY. scored.jsonl and predictions.jsonl must agree with each other and with the runner's
     condition_result.json (ids, strategy, dataset, seed, budget, run id, counts).

WHAT THE GATE CANNOT DO: "real" means "labelled as non-mock throughout and internally consistent". The
saved files do not prove that a real API produced the answers (a test double that calls itself
"openai" would pass). Provenance hashes are recorded so a result can be traced to its exact input.

No gold answers are read here (scored.jsonl holds them, but only the scores are kept in the rows).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aggregation.aggregate import aggregate_matrices
from aggregation.loader import AggregationError, discover, resolve_duplicates
from aggregation.measures import read_jsonl_models
from aggregation.schemas import ConditionRecord, Issue
from analysis.schemas import (AnalysisInput, ConditionInput, DataKind, QuestionObservation, SourceFile)
from evaluation.schemas import PredictionRecord, ScoredRecord
from experiments.schemas import RESULT_FILE, read_condition_result

# report reasons that --allow-incomplete may waive (everything else still refuses)
_INCOMPLETE_REASON_PREFIXES = ("matrix incomplete", "the planned conditions are unknown")

SCORED_FILE, PREDICTIONS_FILE, SELECTION_FILE = "scored.jsonl", "predictions.jsonl", "extraction/selection.json"


class AnalysisRefused(RuntimeError):
    """The input must not be analysed (as it is / with these options). `kind` says which gate refused:
    mock | mixed | not_reportable | input."""

    def __init__(self, kind: str, reasons: list[str]):
        self.kind, self.reasons = kind, list(reasons)
        super().__init__(f"analysis refused ({kind}): " + "; ".join(self.reasons))


# ------------------------------------------------------------------------------------- helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _config_mock_reasons(rec: ConditionRecord) -> list[str]:
    cfg = rec.config or {}
    out = []
    if rec.is_mock:
        out.append("the runner flagged the condition is_mock")
    if rec.data_source == "mock" or cfg.get("data_source") == "mock":
        out.append("data_source is mock")
    for stage in ("extraction", "generation"):
        if cfg.get(f"{stage}_backend") == "mock":
            out.append(f"{stage} backend is mock")
    if cfg.get("judge_backend") == "mock" and rec.judge_enabled is not False:
        out.append("judge backend is mock")
    return out


def classify(records: list[ConditionRecord], per_item_mock: dict[str, list[str]]) -> tuple[DataKind, dict[str, list[str]]]:
    """('mock'|'real', {condition_id: mock reasons}); raises AnalysisRefused('mixed') for a mixture.
    Looks at EVERY loaded condition, included or not."""
    reasons = {r.condition_id: _config_mock_reasons(r) + per_item_mock.get(r.condition_id, []) for r in records}
    mock_ids = sorted(c for c, v in reasons.items() if v)
    if mock_ids and len(mock_ids) < len(records):
        real_ids = sorted(set(reasons) - set(mock_ids))
        raise AnalysisRefused("mixed", [
            f"mock and non-mock conditions are mixed in one matrix ({len(mock_ids)} mock, {len(real_ids)} not); "
            f"mock results are never combined with real ones. Mock: {mock_ids[:3]}...; not mock: {real_ids[:3]}..."])
    return ("mock" if mock_ids else "real"), reasons


def _refuse_mock_unless_allowed(kind: DataKind, allow_mock_layout_test: bool) -> None:
    if kind == "mock" and not allow_mock_layout_test:
        raise AnalysisRefused("mock", [
            "every condition is mock data (mock dataset / backends): it can never be reported as a finding. "
            "To check figure and table layouts only, pass --allow-mock-layout-test; the output then goes to a "
            "separate mock folder, is watermarked 'MOCK — NOT RESULTS', and contains no findings."])


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------- one condition
def _load_condition(matrix_dir: Path, rec: ConditionRecord, directory: Path, errors: list[str],
                    warnings: list[Issue], sources: list[SourceFile]):
    """(ConditionInput, [QuestionObservation], per-item mock reasons) or None if the files are unusable
    (the reasons are appended to `errors`)."""
    cid = rec.condition_id
    rel = directory.relative_to(matrix_dir).as_posix()

    result = read_condition_result(directory)
    if result is None:
        errors.append(f"{cid}: {RESULT_FILE} is missing or unreadable")
        return None
    scored, p1 = read_jsonl_models(directory / SCORED_FILE, ScoredRecord)
    preds, p2 = read_jsonl_models(directory / PREDICTIONS_FILE, PredictionRecord)
    if scored is None or preds is None:
        errors.append(f"{cid}: " + "; ".join(x for x in (f"{SCORED_FILE} {p1}" if scored is None else "",
                                                           f"{PREDICTIONS_FILE} {p2}" if preds is None else "") if x))
        return None

    local = []
    n_before = len(errors)

    def err(msg):
        errors.append(f"{cid}: {msg}")

    # --- ids: unique, scored == predictions, failures disjoint, universe == runner's n_questions
    scored_ids = [s.question_id for s in scored]
    pred_ids = [p.question_id for p in preds]
    if len(set(scored_ids)) != len(scored_ids):
        err("scored.jsonl has a question id more than once")
    if len(set(pred_ids)) != len(pred_ids):
        err("predictions.jsonl has a question id more than once")
    if set(scored_ids) != set(pred_ids):
        err(f"scored.jsonl and predictions.jsonl disagree on the questions "
            f"(only scored: {sorted(set(scored_ids) - set(pred_ids))[:3]}, only predicted: {sorted(set(pred_ids) - set(scored_ids))[:3]})")
    failed_ids = [f.question_id for f in result.question_failures]
    if set(failed_ids) & set(scored_ids):
        err("a question is both scored and listed as having no prediction")
    universe = sorted(set(scored_ids) | set(failed_ids))
    if result.n_questions is not None and len(universe) != result.n_questions:
        err(f"{len(universe)} questions found in the files but the runner recorded n_questions={result.n_questions}")
    if result.n_predictions != len(preds):
        err(f"runner recorded n_predictions={result.n_predictions} but predictions.jsonl has {len(preds)}")

    # --- every record belongs to this condition
    for s in scored:
        for field, expected in (("dataset", rec.dataset), ("strategy", rec.strategy), ("seed", rec.seed),
                                ("corpus_id", rec.corpus_id), ("run_id", rec.run_id)):
            got = getattr(s, field)
            if got is not None and expected is not None and got != expected:
                err(f"scored record {s.question_id}: {field}={got!r}, the condition has {expected!r}")
                break
        if s.budget is not None and abs(s.budget * 100 - rec.budget_pct) > 1e-6:
            err(f"scored record {s.question_id}: budget={s.budget}, the condition has {rec.budget_pct}%")

    # --- per-item mock evidence (catches a hand-edited is_mock flag)
    if any(p.answer.backend == "mock" for p in preds):
        local.append("some predictions were answered by a mock backend")
    if any(s.is_mock_generation for s in scored):
        local.append("some scores carry is_mock_generation")
    if any(s.is_mock_judge for s in scored) and rec.judge_enabled is not False:
        local.append("some scores come from a mock judge")

    # --- selected chunks (used later by the evidence metrics)
    selected = None
    sel = _read_json(directory / SELECTION_FILE)
    if sel is None or not isinstance(sel.get("selected_chunk_ids"), list):
        warnings.append(Issue(code="no_selection_file", severity="warning", condition_id=cid,
                              message=f"{SELECTION_FILE} missing or unreadable: selection-based evidence metrics "
                                      f"cannot be computed for this condition"))
    else:
        selected = [str(x) for x in sel["selected_chunk_ids"]]
        n_sel = rec.counts.get("n_selected")
        if n_sel is not None and len(selected) != n_sel:
            err(f"{SELECTION_FILE} lists {len(selected)} chunks but the runner recorded n_selected={n_sel}")
        if len(set(selected)) != len(selected):
            err(f"{SELECTION_FILE} lists a chunk more than once")

    if len(errors) > n_before:
        return None

    # --- provenance
    for name in (RESULT_FILE, SCORED_FILE, PREDICTIONS_FILE) + ((SELECTION_FILE,) if selected is not None else ()):
        sources.append(SourceFile(path=f"{rel}/{name}", sha256=_sha256(directory / name)))

    pred_by_id = {p.question_id: p for p in preds}
    obs = []
    for s in scored:
        p = pred_by_id[s.question_id]
        r = p.retrieval
        obs.append(QuestionObservation(
            condition_id=cid, dataset=rec.dataset, strategy=rec.strategy, budget_pct=rec.budget_pct, seed=rec.seed,
            shuffle_seed=rec.shuffle_seed, question_id=s.question_id, outcome=s.outcome,
            generation_status=s.generation_status, em=s.em, f1=s.f1, judge_correct=s.judge_correct,
            judge_status=s.judge_status,
            retrieved_chunk_ids=tuple(r.passage_chunk_ids) if r is not None else (),
            n_context_words=r.n_words if r is not None else None))
    for qid in sorted(failed_ids):
        obs.append(QuestionObservation(
            condition_id=cid, dataset=rec.dataset, strategy=rec.strategy, budget_pct=rec.budget_pct, seed=rec.seed,
            shuffle_seed=rec.shuffle_seed, question_id=qid, outcome="no_prediction"))
    cond = ConditionInput(
        condition_id=cid, directory=rel, dataset=rec.dataset, strategy=rec.strategy, budget_pct=rec.budget_pct,
        seed=rec.seed, shuffle_seed=rec.shuffle_seed, corpus_id=rec.corpus_id, status=rec.status,
        n_questions=len(universe), question_ids=universe, n_no_prediction=len(failed_ids), selected_chunk_ids=selected)
    return cond, obs, local


# ------------------------------------------------------------------------------------- alignment
def _check_alignment(conds: list[ConditionInput], errors: list[str]) -> tuple[dict, dict]:
    """One question seed, one corpus and one question universe per dataset."""
    seeds, universes = {}, {}
    for dataset in sorted({c.dataset for c in conds}):
        group = [c for c in conds if c.dataset == dataset]
        ds_seeds = sorted({c.seed for c in group})
        if len(ds_seeds) != 1:
            errors.append(f"{dataset}: {len(ds_seeds)} different question seeds {ds_seeds}; the analysis compares "
                          f"strategies on one question set per dataset (pooling seeds is not supported)")
        corpora = sorted({c.corpus_id for c in group if c.corpus_id})
        if len(corpora) > 1:
            errors.append(f"{dataset}: conditions use different corpora {corpora}")
        sets = {}
        for c in group:
            sets.setdefault(tuple(c.question_ids), []).append(c.condition_id)
        if len(sets) > 1:
            sizes = sorted((len(k), v[0]) for k, v in sets.items())
            errors.append(f"{dataset}: the conditions do not share one question set (found {len(sets)} different "
                          f"sets, sizes/first condition: {sizes[:4]}); paired comparisons would be meaningless")
        seeds[dataset] = ds_seeds[0] if ds_seeds else None
        universes[dataset] = list(group[0].question_ids)
    return seeds, universes


# ------------------------------------------------------------------------------------- main entry
def load_analysis_input(matrix_dir: str | Path, *, allow_mock_layout_test: bool = False,
                        allow_incomplete: bool = False) -> AnalysisInput:
    """Gate and load one matrix directory. Raises AnalysisRefused (kind: mock | mixed | not_reportable | input)."""
    matrix_dir = Path(matrix_dir)
    if not (matrix_dir / "conditions").is_dir():
        raise AnalysisRefused("input", [f"{matrix_dir} is not a matrix directory (no conditions/ folder)"])

    # 1. Phase 11 aggregation (its own refusals become ours)
    try:
        report, records = aggregate_matrices([matrix_dir])
        disc = discover([matrix_dir])
        kept, _dropped, _dups = resolve_duplicates(disc.conditions, "error")
    except (AggregationError, ValueError, FileNotFoundError) as e:
        raise AnalysisRefused("input", [f"aggregation failed: {e}"]) from e
    dir_of = {lc.result.condition_id: lc.directory for lc in kept}

    # 2a. early mock / mixed decision from the recorded flags and configs
    kind, _ = classify(records, {})
    _refuse_mock_unless_allowed(kind, allow_mock_layout_test)

    # 3 / 4 / 5. per-question files, integrity and alignment
    errors: list[str] = []
    warnings: list[Issue] = []
    sources: list[SourceFile] = []
    if (matrix_dir / "matrix.json").exists():
        sources.append(SourceFile(path="matrix.json", sha256=_sha256(matrix_dir / "matrix.json")))
    conds: list[ConditionInput] = []
    observations: list[QuestionObservation] = []
    per_item_mock: dict[str, list[str]] = {}
    included = [r for r in records if r.included]
    for rec in included:
        loaded = _load_condition(matrix_dir, rec, dir_of[rec.condition_id], errors, warnings, sources)
        if loaded is None:
            continue
        cond, obs, local_mock = loaded
        conds.append(cond)
        observations.extend(obs)
        if local_mock:
            per_item_mock[rec.condition_id] = local_mock

    # 2b. final mock decision, now including per-item evidence (a doctored is_mock flag is caught here)
    kind, _ = classify(records, per_item_mock)
    _refuse_mock_unless_allowed(kind, allow_mock_layout_test)

    if not conds and not errors:
        errors.append("no included conditions to analyse")
    seeds, universes = ({}, {}) if errors else _check_alignment(conds, errors)
    if errors:
        raise AnalysisRefused("input", errors)

    # 3. reportability of REAL data
    caveats: list[str] = []
    waived = False
    if kind == "real":
        reasons = [] if report.reportable else list(report.not_reportable_reasons)
        if report.n_errors:
            reasons.append(f"{report.n_errors} unresolved error issue(s) in the aggregation report")
        remaining = [x for x in reasons if not x.startswith(_INCOMPLETE_REASON_PREFIXES)]
        incomplete = [x for x in reasons if x.startswith(_INCOMPLETE_REASON_PREFIXES)]
        if remaining:
            raise AnalysisRefused("not_reportable", remaining + (incomplete if incomplete else []))
        if incomplete and not allow_incomplete:
            raise AnalysisRefused("not_reportable", incomplete + [
                "pass --allow-incomplete to analyse what exists (the output is then marked real_partial)"])
        waived = bool(incomplete)
        status = "real_partial" if waived else "real_complete"
        caveats.append("'real' means labelled non-mock throughout and internally consistent; the saved files cannot "
                       "prove that the real API produced the answers (see analysis/loader.py).")
        if waived:
            caveats.append("PARTIAL RUN: " + "; ".join(incomplete))
    else:
        status = "mock_layout_test"
        caveats.append("MOCK DATA: for checking layouts only. These are not research results and carry no findings.")

    judge_flags = {r.judge_enabled for r in included if r.judge_enabled is not None}
    judge_enabled = judge_flags.pop() if len(judge_flags) == 1 else None
    if judge_enabled is False:
        caveats.append("the judge was disabled: judge accuracy is not available")

    excluded = [e.model_dump(mode="json") for e in report.excluded]
    return AnalysisInput(
        matrix_dir=str(matrix_dir), matrix_ids=report.matrix_ids, status=status, data_kind=kind,
        judge_enabled=judge_enabled, question_seed=seeds, question_ids=universes, conditions=conds,
        observations=observations, records=records, excluded=excluded, report=report,
        issues=[i for i in report.issues if i.severity == "warning"] + warnings, allowed_incomplete=waived,
        source_files=sources, caveats=caveats)
