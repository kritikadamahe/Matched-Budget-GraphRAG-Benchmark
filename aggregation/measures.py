"""
Turn one Phase 10 condition into one flat ConditionRecord (Phase 11).

WHERE EACH FIGURE COMES FROM
  cost (if uncached)   condition_result.cost.<stage>.cost_if_uncached_usd   (cache hits at their original cost;
                       the comparable figure). Cross-checked against the sum over the per-item files.
  cost (spent)         the same block's cost_spent_usd. Depends on cache state and run order: labelled
                       cache_dependent and not for comparing conditions.
  tokens               REBUILT from the per-item files, which keep the original usage on cache hits:
                       extraction/extractions.jsonl, predictions.jsonl (answer.usage), scored.jsonl (judge.usage).
  runtime (intrinsic)  ranking: stages.ranking.details.ranking_seconds (original, kept when reused);
                       extraction / generation / judge: sum of the per-call runtime_seconds in those files
                       (original runtime kept on cache hits); selection / graph / retrieval: the stage's
                       elapsed time (the runner does not cache them; embedding-cache effects are not separable).
  runtime (this run)   wall_seconds and each stage's elapsed_seconds: cache_dependent. A stage that never
                       ran is `not_run`, NOT 0.0 (the schema's default elapsed of 0.0 is not a measurement).

TOTALS are "measured" only when every required component is measured. Required = extraction and
generation, plus the judge unless it was disabled; a stage that aborted or never ran makes the total a
`lower_bound` (or `unknown`), never a complete figure.

NORMALISED columns divide by a denominator that exists and covers what the numerator covers:
per selected chunk (n_selected) and per question (n_questions, only if every question has a prediction).

CROSS-CHECKS compare the per-item files with what the runner recorded (row counts, cache hits, status counts,
tokens, cost). A mismatch becomes an Issue; the recorded values are never silently corrected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from aggregation.loader import INCLUDED_STATUSES, LoadedCondition, judge_enabled_of, shared_config_fingerprint
from aggregation.schemas import ConditionRecord, Issue, Measure
from evaluation.schemas import PredictionRecord, ScoredRecord
from experiments.schemas import STAGES
from extraction.schemas import ExtractionResult

COST_TOL = 2e-6          # costs are rounded to 6 decimals by the earlier phases
TOKEN_STAGES = ("extraction", "generation", "judge")
PIPE_STAGE = {"extraction": "extraction", "generation": "generation", "judge": "evaluation"}


def _r(x: float) -> float:
    return round(x, 9)


def _close(a, b, tol=COST_TOL) -> bool:
    return abs(a - b) <= tol


# ------------------------------------------------------------------------------- file readers
def read_jsonl_models(path: Path, model):
    """(rows, problem). rows is None when the file is missing or unreadable."""
    if not path.exists():
        return None, "missing"
    try:
        with open(path, encoding="utf-8") as f:
            return [model.model_validate_json(line) for line in f if line.strip()], None
    except (ValueError, OSError) as e:
        return None, f"unreadable ({str(e).splitlines()[0][:100]})"


def read_json_file(path: Path):
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (ValueError, OSError) as e:
        return None, f"unreadable ({str(e)[:100]})"


# ------------------------------------------------------------------------------- the builder
class _Builder:
    def __init__(self, lc: LoadedCondition):
        self.lc = lc
        self.r = lc.result
        self.dir = lc.directory
        self.issues: list[Issue] = []
        self.judge_enabled = judge_enabled_of(self.r)
        self.measures: dict[str, Measure] = {}
        self.ext_rows, self.ext_err = read_jsonl_models(self.dir / "extraction" / "extractions.jsonl", ExtractionResult)
        self.preds, self.pred_err = read_jsonl_models(self.dir / "predictions.jsonl", PredictionRecord)
        self.scored, self.scored_err = read_jsonl_models(self.dir / "scored.jsonl", ScoredRecord)
        self.run_summary, _ = read_json_file(self.dir / "extraction" / "run_summary.json")
        self.eval_summary, self.eval_err = read_json_file(self.dir / "eval_summary.json")

    # -- small helpers --------------------------------------------------------------------
    def stage_status(self, name: str) -> str:
        st = self.r.stages.get(name)
        return st.status if st is not None else "not_run"

    def issue(self, code, severity, message, expected=None, actual=None):
        self.issues.append(Issue(code=code, severity=severity, condition_id=self.r.condition_id,
                                 message=message, expected=expected, actual=actual))

    def judge_required(self) -> bool:
        return self.judge_enabled is not False        # unknown counts as required: a total cannot be complete

    def required(self) -> list[str]:
        return ["extraction", "generation"] + (["judge"] if self.judge_required() else [])

    def _artifact_problem(self, what: str, err: str | None, stage: str):
        """A missing / unreadable per-item file is an error only if its stage claims to have run to the end."""
        if err and self.stage_status(stage) == "ok":
            self.issue("artifact_" + ("missing" if err == "missing" else "unreadable"), "error",
                       f"{what} is {err}, but the {stage} stage is recorded as ok")

    # -- recorded cost ----------------------------------------------------------------------
    def recorded_cost(self, stage: str, field: str, kind: str, cache_dependent: bool) -> Measure:
        src = f"condition_result.cost.{stage}.{field}"
        if stage == "judge" and self.judge_enabled is False:
            return Measure(kind=kind, status="disabled", reason="the judge was not run for this condition",
                           source=src, cache_dependent=cache_dependent)
        pipe = PIPE_STAGE[stage]
        st = self.stage_status(pipe)
        block = (self.r.cost or {}).get(stage)
        if not isinstance(block, dict):
            status = "not_run" if st in ("not_run", "skipped") else "unknown"
            return Measure(kind=kind, status=status, source=src, cache_dependent=cache_dependent,
                           reason=f"no {stage} cost was recorded (the {pipe} stage is {st})")
        value = block.get(field)
        if value is not None and st == "ok":
            return Measure(kind=kind, status="measured", value=value, source=src, cache_dependent=cache_dependent)
        if value is not None:
            return Measure(kind=kind, status="lower_bound", lower_bound=value, source=src, cache_dependent=cache_dependent,
                           reason=f"the {pipe} stage {st}: only the items processed before it stopped are counted")
        lb = block.get("known_cost_lower_bound_usd") if field == "cost_if_uncached_usd" else None
        if lb is not None:
            return Measure(kind=kind, status="lower_bound", lower_bound=lb, source=src, cache_dependent=cache_dependent,
                           reason="token usage is unknown for some calls; the known part is a lower bound")
        return Measure(kind=kind, status="unknown", source=src, cache_dependent=cache_dependent,
                       reason="token usage is unknown for some calls")

    # -- sums over per-item files --------------------------------------------------------------
    def sum_items(self, *, kind: str, pipe: str, values, err, expected, source: str, unknown_what: str,
                  as_int: bool = False, expected_unavailable: str | None = None) -> Measure:
        st = self.stage_status(pipe)
        if st in ("not_run", "skipped"):
            return Measure(kind=kind, status="not_run", source=source, reason=f"the {pipe} stage did not run")
        if values is None:
            return Measure(kind=kind, status="unknown", source=source,
                           reason=f"{source} is {err}: cannot be reconstructed")
        known = [v for v in values if v is not None]
        n_unknown = len(values) - len(known)
        reasons = []
        if st != "ok":
            reasons.append(f"the {pipe} stage {st}: only the items processed before it stopped are counted")
        if expected is not None and len(values) != expected:
            reasons.append(f"{len(values)} item(s) found, {expected} expected")
        if expected is None and expected_unavailable:
            reasons.append(expected_unavailable)
        if n_unknown:
            reasons.append(f"{n_unknown} item(s) have unknown {unknown_what}")
        total = sum(known) if as_int else _r(math.fsum(known))
        if reasons:
            return Measure(kind=kind, status="lower_bound", lower_bound=total, source=source, reason="; ".join(reasons))
        return Measure(kind=kind, status="measured", value=total, source=source)

    # -- totals ------------------------------------------------------------------------------
    def total(self, kind: str, parts: dict[str, Measure], required: list[str], source: str, *,
              cache_dependent: bool = False, as_int: bool = False) -> Measure:
        contributions, missing, statuses = [], [], []
        for name in required:
            m = parts[name]
            statuses.append(m.status)
            if m.status == "measured":
                contributions.append(m.value)
            else:
                missing.append(f"{name}: {m.status}")
                if m.status == "lower_bound" and m.lower_bound is not None:
                    contributions.append(m.lower_bound)
        base = dict(kind=kind, source=source, components=list(required), cache_dependent=cache_dependent)
        total = (sum(contributions) if as_int else _r(math.fsum(contributions)))
        if not missing:
            return Measure(status="measured", value=total, **base)
        reason = "incomplete: " + ", ".join(missing)
        if contributions:
            return Measure(status="lower_bound", lower_bound=total, reason=reason, **base)
        if all(s == "not_run" for s in statuses):
            return Measure(status="not_run", reason=reason, **base)
        return Measure(status="unknown", reason=reason, **base)

    # -- the figure families ----------------------------------------------------------------
    def build_costs(self):
        for kind, field, prefix, dep in (("cost", "cost_if_uncached_usd", "cost", False),
                                         ("spent", "cost_spent_usd", "spent", True)):
            parts = {s: self.recorded_cost(s, field, kind, dep) for s in TOKEN_STAGES}
            for s, m in parts.items():
                self.measures[f"{prefix}_{s}_usd"] = m
            self.measures[f"{prefix}_total_usd"] = self.total(
                kind, parts, self.required(), f"sum of {'+'.join(self.required())} ({field})", cache_dependent=dep)

    def build_tokens(self):
        ext_usage = None if self.ext_rows is None else [x.usage for x in self.ext_rows]
        gen_usage = None if self.preds is None else [p.answer.usage for p in self.preds]
        judge_calls = None if self.scored is None else [s.judge for s in self.scored if s.judge is not None]
        jud_usage = None if judge_calls is None else [j.usage for j in judge_calls]
        expected_judge = None
        if self.eval_summary and isinstance(self.eval_summary.get("cost", {}).get("judge"), dict):
            expected_judge = self.eval_summary["cost"]["judge"].get("n")
        cfg = {
            "extraction": (ext_usage, self.ext_err, self.r.n_selected, "extraction/extractions.jsonl", None),
            "generation": (gen_usage, self.pred_err, self.r.n_predictions, "predictions.jsonl", None),
            "judge": (jud_usage, self.scored_err or self.eval_err, expected_judge, "scored.jsonl",
                      "eval_summary.json does not give the expected number of judge calls"),
        }
        parts = {"input": {}, "output": {}}
        for stage, (usage, err, expected, src, unavailable) in cfg.items():
            for direction, attr in (("input", "input_tokens"), ("output", "output_tokens")):
                if stage == "judge" and self.judge_enabled is False:
                    m = Measure(kind="tokens", status="disabled", source=src, reason="the judge was not run for this condition")
                else:
                    vals = None if usage is None else [getattr(u, attr) for u in usage]
                    m = self.sum_items(kind="tokens", pipe=PIPE_STAGE[stage], values=vals, err=err, expected=expected,
                                       source=src, unknown_what="token usage", as_int=True, expected_unavailable=unavailable)
                parts[direction][stage] = m
                self.measures[f"tokens_{stage}_{direction}"] = m
        for direction in ("input", "output"):
            self.measures[f"tokens_total_{direction}"] = self.total(
                "tokens", parts[direction], self.required(), f"sum of {'+'.join(self.required())} ({direction} tokens)",
                as_int=True)
        self._judge_calls, self._ext_usage = judge_calls, ext_usage

    def stage_elapsed(self, name: str, kind: str, cache_dependent: bool) -> Measure:
        st = self.stage_status(name)
        rec = self.r.stages.get(name)
        src = f"stages.{name}.elapsed_seconds"
        if st in ("not_run",) or rec is None:
            return Measure(kind=kind, status="not_run", source=src, cache_dependent=cache_dependent,
                           reason=f"the {name} stage never ran (its recorded 0.0 is a default, not a measurement)")
        if st in ("ok", "skipped"):
            return Measure(kind=kind, status="measured", value=rec.elapsed_seconds, source=src, cache_dependent=cache_dependent)
        return Measure(kind=kind, status="lower_bound", lower_bound=rec.elapsed_seconds, source=src,
                       cache_dependent=cache_dependent, reason=f"the {name} stage {st}: time until it stopped")

    def build_runtimes(self):
        rt: dict[str, Measure] = {}
        # ranking: original computation time, kept even when the ranking was reused in memory
        rank = self.r.stages.get("ranking")
        rsec = None if rank is None else rank.details.get("ranking_seconds")
        if self.stage_status("ranking") == "ok" and rsec is not None:
            rt["ranking"] = Measure(kind="runtime", status="measured", value=rsec, source="stages.ranking.details.ranking_seconds")
        else:
            rt["ranking"] = Measure(kind="runtime", status="not_run" if self.stage_status("ranking") == "not_run" else "unknown",
                                    source="stages.ranking.details.ranking_seconds", reason="ranking time was not recorded")
        for name in ("selection", "graph", "retrieval"):
            m = self.stage_elapsed(name, "runtime", False)
            rt[name] = m
        ext_rt = None if self.ext_rows is None else [x.runtime_seconds for x in self.ext_rows]
        gen_rt = None if self.preds is None else [p.answer.runtime_seconds for p in self.preds]
        jud_rt = None if self._judge_calls is None else [j.runtime_seconds for j in self._judge_calls]
        expected_judge = None
        if self.eval_summary and isinstance(self.eval_summary.get("cost", {}).get("judge"), dict):
            expected_judge = self.eval_summary["cost"]["judge"].get("n")
        rt["extraction"] = self.sum_items(kind="runtime", pipe="extraction", values=ext_rt, err=self.ext_err,
                                          expected=self.r.n_selected, source="extraction/extractions.jsonl",
                                          unknown_what="runtime")
        rt["generation"] = self.sum_items(kind="runtime", pipe="generation", values=gen_rt, err=self.pred_err,
                                          expected=self.r.n_predictions, source="predictions.jsonl", unknown_what="runtime")
        if self.judge_enabled is False:
            rt["judge"] = Measure(kind="runtime", status="disabled", source="scored.jsonl",
                                  reason="the judge was not run for this condition")
        else:
            rt["judge"] = self.sum_items(kind="runtime", pipe="evaluation", values=jud_rt, err=self.scored_err or self.eval_err,
                                         expected=expected_judge, source="scored.jsonl", unknown_what="runtime",
                                         expected_unavailable="eval_summary.json does not give the expected number of judge calls")
        names = {"ranking": "ranking", "selection": "selection", "extraction": "extraction", "graph": "graph",
                 "retrieval": "retrieval", "generation": "generation", "judge": "judge"}
        for k, m in rt.items():
            self.measures[f"runtime_{names[k]}_s"] = m
        required = ["ranking", "selection", "extraction", "graph", "retrieval", "generation"] + (["judge"] if self.judge_required() else [])
        self.measures["runtime_total_s"] = self.total(
            "runtime", rt, required,
            "sum of ranking + selection + extraction calls + graph + retrieval + generation calls + judge calls; "
            "excludes corpus loading (shared, not recorded when reused), EM/F1 scoring and file writing")
        # this run: cache dependent, kept apart
        self.measures["wall_seconds"] = Measure(kind="elapsed", status="measured", value=self.r.wall_seconds,
                                               source="wall_seconds", cache_dependent=True)
        elapsed = {}
        for s in STAGES:
            m = self.stage_elapsed(s, "elapsed", True)
            elapsed[s] = m
            self.measures[f"elapsed_{s}_s"] = m
        if all(m.status == "measured" for m in elapsed.values()):
            self.measures["elapsed_unattributed_s"] = Measure(
                kind="elapsed", status="measured", cache_dependent=True,
                value=_r(self.r.wall_seconds - math.fsum(m.value for m in elapsed.values())),
                source="wall_seconds - sum of stage elapsed times")
        else:
            self.measures["elapsed_unattributed_s"] = Measure(
                kind="elapsed", status="not_applicable", cache_dependent=True,
                reason="only defined when every stage ran to the end")
        self._rt = rt

    def ratio(self, name: str, kind_src: str, den: int | None, den_name: str, *, coverage_ok: bool = True,
              coverage_reason: str | None = None):
        num = self.measures[kind_src]
        base = dict(kind="normalized", source=f"{kind_src} / {den_name}")
        if num.status in ("disabled", "not_run", "unknown"):
            self.measures[name] = Measure(status=num.status, reason=num.reason, **base)
        elif not den:
            self.measures[name] = Measure(status="not_applicable", reason=f"{den_name} is {den if den is not None else 'unknown'}: no denominator", **base)
        elif not coverage_ok:
            self.measures[name] = Measure(status="not_applicable", reason=coverage_reason, **base)
        elif num.status == "measured":
            self.measures[name] = Measure(status="measured", value=_r(num.value / den), **base)
        else:                                                              # lower_bound
            self.measures[name] = Measure(status="lower_bound", lower_bound=_r(num.lower_bound / den), reason=num.reason, **base)

    def build_normalised(self):
        r = self.r
        covered = r.n_questions is not None and r.n_predictions == r.n_questions
        why = (f"only {r.n_predictions} of {r.n_questions} questions have a prediction: a per-question figure "
               f"would mix covered and uncovered questions")
        self.ratio("cost_extraction_per_selected_chunk_usd", "cost_extraction_usd", r.n_selected, "n_selected")
        self.ratio("runtime_extraction_per_selected_chunk_s", "runtime_extraction_s", r.n_selected, "n_selected")
        for stage, key in (("total", "cost_total_usd"), ("generation", "cost_generation_usd"), ("judge", "cost_judge_usd")):
            self.ratio(f"cost_{stage}_per_question_usd", key, r.n_questions, "n_questions", coverage_ok=covered, coverage_reason=why)
        for stage, key in (("total", "runtime_total_s"), ("generation", "runtime_generation_s"), ("judge", "runtime_judge_s")):
            self.ratio(f"runtime_{stage}_per_question_s", key, r.n_questions, "n_questions", coverage_ok=covered, coverage_reason=why)

    # -- cross-checks ------------------------------------------------------------------------
    def check(self):
        r = self.r
        if r.n_selected is not None and r.n_selected_expected is not None and r.n_selected != r.n_selected_expected:
            self.issue("budget_mismatch", "error", "selected chunks differ from the budgeted count",
                       expected=r.n_selected_expected, actual=r.n_selected)
        self._check_extraction()
        self._check_generation()
        self._check_judge()
        if r.n_questions is not None and r.n_predictions + len(r.question_failures) != r.n_questions and self.stage_status("generation") == "ok":
            self.issue("question_accounting", "error", "predictions + question failures do not add up to the questions",
                       expected=r.n_questions, actual=r.n_predictions + len(r.question_failures))

    def _cost_check(self, code, stage_label, items_cost, recorded):
        if recorded is not None and items_cost is not None and not _close(items_cost, recorded):
            self.issue(code, "error", f"{stage_label}: cost summed over the per-item file differs from the recorded cost",
                       expected=recorded, actual=_r(items_cost))

    def _check_extraction(self):
        self._artifact_problem("extraction/extractions.jsonl", self.ext_err, "extraction")
        rows, rs = self.ext_rows, self.run_summary
        if rows is None or self.stage_status("extraction") != "ok":
            return
        if self.r.n_selected is not None and len(rows) != self.r.n_selected:
            self.issue("extraction_row_count", "error", "extraction rows differ from the selected chunk count",
                       expected=self.r.n_selected, actual=len(rows))
        if rs:
            hits = sum(1 for x in rows if x.cache_hit)
            if rs.get("n_cache_hits") is not None and hits != rs["n_cache_hits"]:
                self.issue("extraction_cache_hits", "error", "cache-hit rows differ from the run summary",
                           expected=rs["n_cache_hits"], actual=hits)
            n_ok = sum(1 for x in rows if x.status == "ok")
            if rs.get("n_ok") is not None and n_ok != rs["n_ok"]:
                self.issue("extraction_status_counts", "error", "rows with status ok differ from the run summary",
                           expected=rs["n_ok"], actual=n_ok)
            fresh = [x.usage for x in rows if not x.cache_hit]
            tokens = rs.get("tokens_spent_this_run")
            if isinstance(tokens, dict) and all(u.known for u in fresh):
                got = {"input": sum(u.input_tokens for u in fresh), "output": sum(u.output_tokens for u in fresh)}
                if got != {"input": tokens.get("input"), "output": tokens.get("output")}:
                    self.issue("extraction_tokens_this_run", "error",
                               "tokens of non-cached rows differ from tokens_spent_this_run", expected=tokens, actual=got)
        if all(x.usage.known for x in rows):
            block = (self.r.cost or {}).get("extraction") or {}
            self._cost_check("extraction_cost", "extraction", math.fsum(x.usage.cost_usd for x in rows), block.get("cost_if_uncached_usd"))
        self._zero_runtime("extraction", [x for x in rows if x.status == "ok" and x.extractor != "mock"], lambda x: x.runtime_seconds)

    def _check_generation(self):
        self._artifact_problem("predictions.jsonl", self.pred_err, "generation")
        preds = self.preds
        if preds is None or self.stage_status("generation") != "ok":
            return
        if len(preds) != self.r.n_predictions:
            self.issue("generation_row_count", "error", "prediction rows differ from n_predictions",
                       expected=self.r.n_predictions, actual=len(preds))
        detail = (self.r.cost or {}).get("generation_detail") or {}
        counts: dict[str, int] = {}
        for p in preds:
            counts[p.answer.status] = counts.get(p.answer.status, 0) + 1
        if detail.get("status_counts") is not None and counts != detail["status_counts"]:
            self.issue("generation_status_counts", "error", "answer statuses differ from the recorded counts",
                       expected=detail["status_counts"], actual=counts)
        if all(p.answer.usage.known for p in preds):
            for attr, key in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens")):
                got = sum(getattr(p.answer.usage, attr) for p in preds)
                if detail.get(key) is not None and got != detail[key]:
                    self.issue("generation_tokens", "error", f"summed {key} differ from the recorded total",
                               expected=detail[key], actual=got)
            block = (self.r.cost or {}).get("generation") or {}
            self._cost_check("generation_cost", "generation", math.fsum(p.answer.usage.cost_usd for p in preds), block.get("cost_if_uncached_usd"))
        self._zero_runtime("generation", [p for p in preds if p.answer.status != "skipped_empty_context" and p.answer.backend != "mock"],
                           lambda p: p.answer.runtime_seconds)

    def _check_judge(self):
        if self.judge_enabled is not True or self.stage_status("evaluation") != "ok":
            return
        self._artifact_problem("scored.jsonl", self.scored_err, "evaluation")
        self._artifact_problem("eval_summary.json", self.eval_err, "evaluation")
        if self.scored is None:
            return
        if self.preds is not None and len(self.scored) != len(self.preds):
            self.issue("scored_row_count", "error", "scored rows differ from prediction rows",
                       expected=len(self.preds), actual=len(self.scored))
        calls = [s.judge for s in self.scored if s.judge is not None]
        block = ((self.eval_summary or {}).get("cost") or {}).get("judge")
        if isinstance(block, dict):
            if block.get("n") is not None and len(calls) != block["n"]:
                self.issue("judge_call_count", "error", "judge results differ from the evaluation summary",
                           expected=block["n"], actual=len(calls))
            if all(j.usage.known for j in calls):
                for attr in ("input_tokens", "output_tokens"):
                    got = sum(getattr(j.usage, attr) for j in calls)
                    if block.get(attr) is not None and got != block[attr]:
                        self.issue("judge_tokens", "error", f"summed judge {attr} differ from the evaluation summary",
                                   expected=block[attr], actual=got)
                self._cost_check("judge_cost", "judge", math.fsum(j.usage.cost_usd for j in calls), block.get("cost_if_uncached_usd"))
                rec = ((self.r.cost or {}).get("judge") or {}).get("cost_if_uncached_usd")
                self._cost_check("judge_cost_recorded", "judge (condition_result)", math.fsum(j.usage.cost_usd for j in calls), rec)
        self._zero_runtime("judge", [j for j in calls if j.backend != "mock"], lambda j: j.runtime_seconds)

    def _zero_runtime(self, label, real_calls, getter: Callable):
        zero = sum(1 for x in real_calls if getter(x) == 0.0)
        if zero:
            self.issue("zero_runtime_calls", "warning",
                       f"{zero} real {label} call(s) record a runtime of exactly 0.0 (a real API call cannot take no time; "
                       f"the runtime may be missing)", actual=zero)

    # -- assembly ------------------------------------------------------------------------------
    def counts_and_failures(self):
        r = self.r
        ext_status: dict[str, int] = {}
        if self.ext_rows is not None:
            for x in self.ext_rows:
                ext_status[x.status] = ext_status.get(x.status, 0) + 1
        gen_status: dict[str, int] = {}
        if self.preds is not None:
            for p in self.preds:
                gen_status[p.answer.status] = gen_status.get(p.answer.status, 0) + 1
        oc = (r.metrics or {}).get("outcome_counts") or {}
        n_gen_failed = sum(v for k, v in gen_status.items() if k not in ("ok", "skipped_empty_context"))
        counts = {
            "n_chunks_total": r.n_chunks_total, "n_selected_expected": r.n_selected_expected, "n_selected": r.n_selected,
            "n_questions": r.n_questions, "n_predictions": r.n_predictions,
            "n_question_failures": len(r.question_failures),
            "extraction_n_rows": None if self.ext_rows is None else len(self.ext_rows),
            "extraction_n_ok": ext_status.get("ok") if self.ext_rows is not None else None,
            "extraction_n_failed": None if self.ext_rows is None else sum(v for k, v in ext_status.items() if k != "ok"),
            "extraction_n_cache_hits": None if self.ext_rows is None else sum(1 for x in self.ext_rows if x.cache_hit),
            "generation_n_failed": None if self.preds is None else n_gen_failed,
            "generation_n_skipped_empty_context": None if self.preds is None else gen_status.get("skipped_empty_context", 0),
            "n_answered": oc.get("answered"), "n_abstained": oc.get("abstained"), "n_empty": oc.get("empty"),
            "n_failed_outcomes": oc.get("failed"),
            "judge_n_calls": None if self._judge_calls is None else len(self._judge_calls),
            "judge_n_failed": (r.metrics or {}).get("judge_n_failed"),
        }
        failures = {
            "extraction_status_counts": ext_status if self.ext_rows is not None else None,
            "generation_status_counts": gen_status if self.preds is not None else None,
            "question_failures": [f.model_dump() for f in r.question_failures],
            "problems": list(r.problems),
        }
        return counts, failures


def build_record(lc: LoadedCondition) -> ConditionRecord:
    """Build one condition's flat record (never raises for a data problem: the problem is an Issue)."""
    b = _Builder(lc)
    b.build_costs()
    b.build_tokens()
    b.build_runtimes()
    b.build_normalised()
    b.check()
    r = lc.result
    counts, failures = b.counts_and_failures()
    included = r.status in INCLUDED_STATUSES
    return ConditionRecord(
        condition_id=r.condition_id, matrix_id=r.matrix_id, run_id=r.run_id, dataset=r.dataset, data_source=r.data_source,
        strategy=r.strategy, budget=r.budget, budget_pct=r.budget_pct, seed=r.seed, corpus_id=r.corpus_id,
        config_hash=r.config_hash, shared_config_fingerprint=shared_config_fingerprint(r.config), config=r.config,
        status=r.status, clean=r.clean, error=r.error, problems=list(r.problems), warnings=list(r.warnings),
        included=included,
        exclusion_reason=None if included else f"{r.status}: {(r.error or 'no error recorded')[:300]}",
        is_mock=r.is_mock, reportable=r.reportable, not_reportable_reasons=list(r.not_reportable_reasons),
        judge_enabled=b.judge_enabled, reconciled=not any(i.severity == "error" for i in b.issues), issues=b.issues,
        counts=counts, failures=failures, metrics=dict(r.metrics or {}), measures=b.measures,
        started_at_utc=r.started_at_utc, finished_at_utc=r.finished_at_utc,
    )
