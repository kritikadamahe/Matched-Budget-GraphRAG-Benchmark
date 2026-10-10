"""Phase 8 Evaluation: scoring records, the summary, mock/reportability handling, failures,
unknown cost, outputs, the CLI, and the integration AnswerResult + question/gold data -> scores."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pytest
import yaml

from evaluation.judge import Judge
from evaluation.schemas import PredictionRecord, read_predictions, write_predictions
from evaluation.score import (NotReportableError, assert_reportable, gold_from_questions, load_gold_file, main,
                              score_predictions, score_record, write_outputs)
from extraction.pricing import cost_usd
from extraction.schemas import Usage
from generation import NOT_FOUND, Answerer
from src.config import ExperimentConfig
from src.corpus import build_chunk_manifest
from tests.helpers_evaluation import (Q, judge_reply, make_answer, make_gold, make_prediction, openai_judge,
                                      unknown_usage)
from tests.helpers_extraction import FakeOpenAIClient, make_error, make_response


def preds_and_gold(rows):
    """rows: (qid, predicted answer, gold answer[, aliases]) -> (records, gold dict)."""
    records, gold = [], {}
    for i, row in enumerate(rows):
        qid, pred, ans = row[:3]
        aliases = row[3] if len(row) > 3 else []
        records.append(make_prediction(pred, qid=qid, question=f"Question {i}?"))
        gold[qid] = make_gold(qid, ans, aliases)
    return records, gold


# ------------------------------------------------------------------- gold data
def test_gold_from_the_real_question_format():
    _, questions, _ = build_chunk_manifest("hotpotqa", "mock", 5, 42, 250, 40)
    gold = gold_from_questions(questions)
    assert set(gold) == {q["question_id"] for q in questions}
    q = questions[0]
    assert gold[q["question_id"]].answer == q["answer"] and gold[q["question_id"]].aliases == q["answer_aliases"]


def test_gold_keeps_musique_aliases_and_rejects_duplicates_and_blank_answers(tmp_path):
    qs = [{"question_id": "a", "answer": "United States", "answer_aliases": ["USA", "US"]}]
    assert gold_from_questions(qs)["a"].all_answers == ["United States", "USA", "US"]
    with pytest.raises(ValueError, match="duplicate"):
        gold_from_questions(qs + qs)
    with pytest.raises(ValueError):
        gold_from_questions([{"question_id": "b", "answer": " "}])
    path = tmp_path / "questions.json"                          # the file src.prepare_data writes
    path.write_text(json.dumps(qs), encoding="utf-8")
    assert load_gold_file(path)["a"].aliases == ["USA", "US"]


# ------------------------------------------------------------------- one record
def test_answered_record_gets_em_and_f1():
    s = score_record(make_prediction("The Chicago."), make_gold(answer="Chicago"))
    assert (s.outcome, s.em, s.f1) == ("answered", 1.0, 1.0)
    assert s.judge_status == "disabled" and s.judge_correct is None and s.judge is None
    p = score_record(make_prediction("New York"), make_gold(answer="New York City"))
    assert p.em == 0.0 and 0.0 < p.f1 < 1.0


def test_aliases_are_scored_best_of():
    s = score_record(make_prediction("U.S."), make_gold(answer="United States of America", aliases=["USA", "US"]))
    assert (s.em, s.f1) == (1.0, 1.0)
    assert s.gold_answer == "United States of America" and s.gold_aliases == ["USA", "US"]


@pytest.mark.parametrize("pred,gold,em", [("yes", "yes", 1.0), ("Yes.", "yes", 1.0), ("no", "yes", 0.0)])
def test_yes_no_answers(pred, gold, em):
    s = score_record(make_prediction(pred), make_gold(answer=gold))
    assert s.outcome == "answered" and s.em == em and s.f1 == em


def test_empty_prediction_scores_zero_and_is_counted_as_empty_not_failed():
    s = score_record(make_prediction("   "), make_gold())
    assert (s.outcome, s.em, s.f1) == ("empty", 0.0, 0.0) and s.generation_status == "ok"


@pytest.mark.parametrize("text", [NOT_FOUND, "Not found.", "NOT FOUND"])
def test_not_found_is_an_abstention_never_a_match(text):
    s = score_record(make_prediction(text), make_gold(answer="Chicago"))
    assert (s.outcome, s.em, s.f1) == ("abstained", 0.0, 0.0)


def test_not_found_scores_zero_even_if_the_gold_text_were_the_same():
    s = score_record(make_prediction(NOT_FOUND), make_gold(answer="not found"))
    assert s.outcome == "abstained" and s.em == 0.0


def test_empty_context_skip_is_an_abstention_and_keeps_its_status():
    s = score_record(make_prediction(NOT_FOUND, status="skipped_empty_context"), make_gold())
    assert s.outcome == "abstained" and s.generation_status == "skipped_empty_context"


@pytest.mark.parametrize("status", ["api_error", "malformed", "refused", "truncated"])
def test_failed_generation_is_not_converted_into_a_wrong_answer(status):
    s = score_record(make_prediction("", status=status, error=f"{status} happened"), make_gold())
    assert s.outcome == "failed" and s.em is None and s.f1 is None and s.judge_correct is None
    assert s.generation_status == status and s.generation_error == f"{status} happened"


# ------------------------------------------------------------------- judge in scoring
def test_judge_is_called_only_for_answered_records(no_network):
    judge, fake = openai_judge(script=[judge_reply(correct=True)])
    records = [make_prediction("Chicago", qid="a", question="Q a?"),
               make_prediction(NOT_FOUND, qid="b", question="Q b?"),
               make_prediction("", qid="c", question="Q c?"),
               make_prediction("", qid="d", question="Q d?", status="truncated")]
    scored = [score_record(r, make_gold(r.question_id), judge) for r in records]
    assert len(fake.calls) == 1
    assert [(s.judge_status, s.judge_correct) for s in scored] == [
        ("ok", True), ("skipped_abstained", False), ("skipped_empty", False), ("skipped_failed_generation", None)]
    assert scored[0].judge.correct is True and scored[1].judge is None


def test_judge_receives_question_gold_and_answer_only(no_network):
    judge, fake = openai_judge(script=[judge_reply()])
    rec = PredictionRecord.build({"question_id": "q1", "question": Q}, make_answer("Chicago"),
                                 retrieval_extra={"relevance_tests": 3}, generation_settings={"x": 1})
    score_record(rec, make_gold(answer="Chicago", aliases=["Chi"]), judge)
    assert fake.calls[0]["messages"][1]["content"] == (
        f"Question: {Q}\nReference answer: Chicago\nOther acceptable answers: Chi\nCandidate answer: Chicago")


# ------------------------------------------------------------------- summary
def test_summary_counts_metrics_failures_and_abstentions():
    records = [make_prediction("Chicago", qid="a", question="Q a?"),
               make_prediction("Boston", qid="b", question="Q b?"),
               make_prediction(NOT_FOUND, qid="c", question="Q c?"),
               make_prediction("", qid="d", question="Q d?", status="malformed", error="bad json")]
    gold = {q: make_gold(q, "Chicago") for q in "abcd"}
    run = score_predictions(records, gold)
    s = run.summary
    assert s["n_records"] == 4 and s["n_scored"] == 3 and s["n_failed"] == 1
    assert s["outcome_counts"] == {"answered": 2, "abstained": 1, "empty": 0, "failed": 1}
    assert s["metrics"]["em"] == pytest.approx(1 / 3) and s["metrics"]["n"] == 3        # headline: failures excluded
    assert s["metrics"]["em_failures_as_wrong"] == pytest.approx(1 / 4)                 # reported side by side
    assert s["failures"] == {"n": 1, "status_counts": {"malformed": 1}, "question_ids": ["d"]}
    assert s["abstentions"]["n"] == 1 and s["abstentions"]["rate_of_scored"] == pytest.approx(1 / 3)
    assert s["judge"] == {"enabled": False}
    assert any("failed" in w for w in s["warnings"]) and any("no judge" in w for w in s["warnings"])
    assert [r.question_id for r in run.scored] == list("abcd")


def test_summary_with_no_scorable_record_has_none_not_zero():
    run = score_predictions([make_prediction("", status="api_error")], {"q1": make_gold()})
    assert run.summary["metrics"]["em"] is None and run.summary["metrics"]["f1"] is None
    assert run.summary["metrics"]["em_failures_as_wrong"] == 0.0


def test_judge_summary_and_agreement_bookkeeping(no_network):
    judge, _ = openai_judge(handler=lambda kw: judge_reply(correct="Chicago" in kw["messages"][1]["content"].split("Candidate answer:")[1]),
                            model="gpt-4o")
    records, gold = preds_and_gold([("a", "Chicago", "Chicago"), ("b", "Boston", "Chicago"), ("c", NOT_FOUND, "Chicago")])
    s = score_predictions(records, gold, judge).summary["judge"]
    assert (s["n_verdicts"], s["n_correct"], s["n_incorrect"], s["accuracy"]) == (3, 1, 2, pytest.approx(1 / 3))
    assert s["n_calls_made"] == 2 and s["status_counts"] == {"ok": 2, "skipped_abstained": 1}
    assert s["prompt_version"] == "judge-v1" and s["same_model_as_generator"] is False


def test_same_model_judge_is_flagged_as_a_limitation(no_network):
    judge, _ = openai_judge(handler=lambda kw: judge_reply())          # gpt-4o-mini, like the answers
    summary = score_predictions(*preds_and_gold([("a", "Chicago", "Chicago")]), judge).summary
    assert summary["judge"]["same_model_as_generator"] is True
    assert any("self-preference" in w for w in summary["warnings"])


def test_failed_judge_calls_are_reported_separately_and_excluded_from_accuracy(no_network):
    judge, _ = openai_judge(script=[judge_reply(correct=True), make_response(content="x"), make_response(content="y")],
                            max_retries=0)
    records, gold = preds_and_gold([("a", "Chicago", "Chicago"), ("b", "Chicago", "Chicago"), ("c", "Chicago", "Chicago")])
    run = score_predictions(records, gold, judge)
    j = run.summary["judge"]
    assert j["n_judge_failed"] == 2 and j["n_verdicts"] == 1 and j["accuracy"] == 1.0
    assert j["judge_failed_question_ids"] == ["b", "c"]
    assert run.scored[1].judge_status == "malformed" and run.scored[1].judge_correct is None
    assert run.scored[1].em == 1.0                                       # EM / F1 unaffected
    assert any("judge call" in w for w in run.summary["warnings"])


def test_judge_run_level_error_keeps_em_f1_and_marks_the_run_stopped(no_network):
    judge, _ = openai_judge(script=[judge_reply(), make_error("AuthenticationError", status=401)])
    records, gold = preds_and_gold([("a", "Chicago", "Chicago"), ("b", "Boston", "Chicago"), ("c", "Chicago", "Chicago")])
    run = score_predictions(records, gold, judge)
    assert [s.judge_status for s in run.scored] == ["ok", "not_run", "not_run"]
    assert [s.em for s in run.scored] == [1.0, 0.0, 1.0]
    assert run.summary["aborted"]["reason"] == "fatal_error" and run.summary["reportable"] is False
    assert any("stopped" in r for r in run.summary["not_reportable_reasons"])


# ------------------------------------------------------------------- mock / reportability
def test_mock_answers_are_flagged_and_not_reportable():
    mock = make_prediction("Chicago", backend="mock")
    run = score_predictions([mock], {"q1": make_gold()})
    s = run.summary
    assert s["is_mock"] and s["is_mock_generation"] and s["reportable"] is False
    assert "mock generation" in s["not_reportable_reasons"][0]
    assert run.scored[0].is_mock is True and run.scored[0].is_mock_generation is True
    with pytest.raises(NotReportableError, match="mock generation"):
        assert_reportable(s)


def test_a_mock_judge_makes_real_answers_not_reportable():
    run = score_predictions([make_prediction("Chicago")], {"q1": make_gold()}, Judge("mock"))
    assert run.summary["is_mock_judge"] and not run.summary["is_mock_generation"]
    assert run.scored[0].is_mock_judge is True and run.scored[0].is_mock is True
    assert run.summary["reportable"] is False and "mock judge" in run.summary["not_reportable_reasons"][0]
    with pytest.raises(NotReportableError):
        assert_reportable(run.summary)


def test_one_mock_record_among_real_ones_taints_the_whole_summary():
    records = [make_prediction("Chicago", qid="a", question="Q a?"),
               make_prediction("Chicago", qid="b", question="Q b?", backend="mock")]
    s = score_predictions(records, {q: make_gold(q) for q in "ab"}).summary
    assert s["is_mock"] and not s["reportable"]


def test_real_backends_are_reportable(no_network):
    judge, _ = openai_judge(handler=lambda kw: judge_reply(), model="gpt-4o")
    s = score_predictions([make_prediction("Chicago")], {"q1": make_gold()}, judge).summary
    assert s["is_mock"] is False and s["reportable"] is True and s["not_reportable_reasons"] == []
    assert_reportable(s)                                                  # does not raise


def test_a_judge_less_run_of_real_answers_is_reportable_for_em_f1():
    s = score_predictions([make_prediction("Chicago")], {"q1": make_gold()}).summary
    assert s["reportable"] is True


# ------------------------------------------------------------------- cost / unknown usage
def test_cost_totals_add_generation_and_judge_cost(no_network):
    judge, _ = openai_judge(handler=lambda kw: judge_reply(prompt_tokens=100, completion_tokens=50))
    records, gold = preds_and_gold([("a", "Chicago", "Chicago"), ("b", "Boston", "Chicago")])
    c = score_predictions(records, gold, judge).summary["cost"]
    per_judge = cost_usd(100, 50, 0.15, 0.60)
    assert c["usage_complete"] is True
    assert c["judge"]["cost_if_uncached_usd"] == pytest.approx(2 * per_judge, abs=1e-6)
    assert c["generation"]["cost_if_uncached_usd"] == pytest.approx(0.00004, abs=1e-6)
    assert c["total_cost_if_uncached_usd"] == pytest.approx(2 * per_judge + 0.00004, abs=1e-6)
    assert c["known_cost_lower_bound_usd"] is None


def test_unknown_generation_usage_makes_totals_unknown_not_zero():
    records = [make_prediction("Chicago", qid="a", question="Q a?"),
               make_prediction("Chicago", qid="b", question="Q b?", usage=Usage.unknown())]
    s = score_predictions(records, {q: make_gold(q) for q in "ab"}).summary
    c = s["cost"]
    assert c["usage_complete"] is False
    assert c["generation"]["cost_if_uncached_usd"] is None and c["total_cost_if_uncached_usd"] is None
    assert c["total_cost_spent_usd"] is None
    assert c["known_cost_lower_bound_usd"] == pytest.approx(0.00002, abs=1e-9)    # the known part only
    assert any("unknown" in w for w in s["warnings"])


def test_unknown_judge_usage_makes_totals_unknown_not_zero(no_network):
    judge, _ = openai_judge(script=[unknown_usage(judge_reply())])
    c = score_predictions([make_prediction("Chicago")], {"q1": make_gold()}, judge).summary["cost"]
    assert c["judge"]["usage_complete"] is False and c["judge"]["cost_if_uncached_usd"] is None
    assert c["usage_complete"] is False and c["total_cost_if_uncached_usd"] is None
    assert c["generation"]["cost_if_uncached_usd"] is not None            # the known side is still reported


def test_cache_hits_keep_their_original_cost_in_the_uncached_total(tmp_path, no_network):
    from generation.cache import QueryCache
    cache = QueryCache(tmp_path)
    j1, _ = openai_judge(script=[judge_reply()], cache=cache)
    first = score_predictions([make_prediction("Chicago")], {"q1": make_gold()}, j1).summary["cost"]["judge"]
    j2, fake = openai_judge(script=[], cache=cache)
    second = score_predictions([make_prediction("Chicago")], {"q1": make_gold()}, j2).summary["cost"]["judge"]
    assert fake.calls == [] and second["n_cache_hits"] == 1
    assert second["cost_spent_usd"] == 0.0 and second["cost_if_uncached_usd"] == first["cost_if_uncached_usd"] > 0


# ------------------------------------------------------------------- input checks
def test_duplicate_predictions_missing_gold_and_empty_input_are_errors():
    rec = make_prediction("Chicago")
    with pytest.raises(ValueError, match="duplicate prediction"):
        score_predictions([rec, rec], {"q1": make_gold()})
    with pytest.raises(ValueError, match="no gold answer"):
        score_predictions([rec], {})
    with pytest.raises(ValueError, match="no predictions"):
        score_predictions([], {"q1": make_gold()})


def test_same_question_in_different_conditions_is_allowed_and_listed():
    a = make_prediction("Chicago")
    b = a.model_copy(update={"strategy": "ketrag"})
    s = score_predictions([a, b], {"q1": make_gold()}).summary
    assert s["n_records"] == 2 and s["n_conditions"] == 2
    assert {c["strategy"] for c in s["conditions"]} == {"random", "ketrag"}


# ------------------------------------------------------------------- outputs
def test_outputs_are_machine_readable(tmp_path):
    records, gold = preds_and_gold([("a", "Chicago", "Chicago"), ("b", "", "Chicago")])
    run = score_predictions(records, gold, Judge("mock"))
    write_outputs(tmp_path / "out", run)
    rows = [json.loads(line) for line in (tmp_path / "out" / "scored.jsonl").read_text().splitlines()]
    assert [r["question_id"] for r in rows] == ["a", "b"] and all(r["is_mock"] is True for r in rows)
    assert rows[0]["em"] == 1.0 and rows[0]["judge_status"] == "ok" and rows[0]["judge"]["backend"] == "mock"
    assert rows[1]["outcome"] == "empty" and rows[1]["judge_correct"] is False
    summary = json.loads((tmp_path / "out" / "eval_summary.json").read_text())
    for key in ("n_records", "metrics", "failures", "abstentions", "judge", "cost", "is_mock", "reportable",
                "not_reportable_reasons", "warnings", "limitations"):
        assert key in summary
    assert any("Seed coupling" in x for x in summary["limitations"])
    assert any("Budget rounding" in x for x in summary["limitations"])
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_unknown_cost_serialises_as_null(tmp_path):
    run = score_predictions([make_prediction("Chicago", usage=Usage.unknown())], {"q1": make_gold()})
    write_outputs(tmp_path, run)
    summary = json.loads((tmp_path / "eval_summary.json").read_text())
    assert summary["cost"]["total_cost_if_uncached_usd"] is None and summary["cost"]["usage_complete"] is False


# ------------------------------------------------------------------- integration
def test_integration_answerer_result_and_question_gold_data_to_scores(tmp_path):
    """Real Answerer (mock backend) + real question dicts from the corpus loader -> record -> scores."""
    _, questions, _ = build_chunk_manifest("hotpotqa", "mock", 5, 42, 250, 40)
    cfg = ExperimentConfig(experiment_id="int", seed=42, dataset="hotpotqa", data_source="mock", num_questions=5,
                           strategy="random", budget=0.5, embedding_backend="tfidf",
                           cache_dir=str(tmp_path / "cache"))
    answerer = Answerer("mock")
    gold_context = "Passages:\n[1] " + questions[0]["answer"] + " Foo."
    records = []
    for i, q in enumerate(questions):
        ctx = gold_context if i == 0 else ""
        records.append(PredictionRecord.build(q, answerer.answer(q["question"], ctx), cfg=cfg,
                                              generation_settings=answerer.settings()))
    gold = gold_from_questions(questions)
    run = score_predictions(records, gold, Judge("mock"))
    assert run.summary["n_records"] == 5 and run.summary["is_mock"] is True and not run.summary["reportable"]
    by_id = {s.question_id: s for s in run.scored}
    assert by_id[questions[1]["question_id"]].outcome == "abstained"          # empty context -> "not found"
    assert by_id[questions[1]["question_id"]].generation_status == "skipped_empty_context"
    assert all(s.gold_answer == gold[s.question_id].answer for s in run.scored)
    assert {r.strategy for r in records} == {"random"} and records[0].generation_settings["backend"] == "mock"


def test_integration_openai_style_answers_and_judge_with_fake_clients_are_reportable(no_network):
    from generation.llm_client import ChatJSONClient
    answer_fake = FakeOpenAIClient(handler=lambda kw: make_response(payload={"reasoning": "r", "answer": "Chicago."}))
    answerer = Answerer("openai", client=ChatJSONClient(client=answer_fake, sleep=lambda s: None))
    question = {"question_id": "q1", "question": Q, "answer": "Chicago", "answer_aliases": []}
    rec = PredictionRecord.build(question, answerer.answer(Q, "Passages:\n[1] Chicago."),
                                 generation_settings=answerer.settings())
    assert rec.answer.answer == "Chicago" and rec.is_mock is False
    judge, _ = openai_judge(handler=lambda kw: judge_reply(), model="gpt-4o")
    run = score_predictions([rec], gold_from_questions([question]), judge)
    assert run.scored[0].em == 1.0 and run.scored[0].judge_correct is True
    assert_reportable(run.summary)


# ------------------------------------------------------------------- CLI
def _cli_setup(tmp_path, **cfg_extra):
    cfg_dict = dict(experiment_id="cli", seed=42, dataset="hotpotqa", data_source="mock", num_questions=5,
                    strategy="random", budget=0.5, embedding_backend="tfidf", cache_dir=str(tmp_path / "cache"),
                    **cfg_extra)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    cfg = ExperimentConfig(**cfg_dict)
    _, questions, _ = build_chunk_manifest("hotpotqa", "mock", 5, 42, 250, 40)
    answerer = Answerer("mock")
    records = [PredictionRecord.build(q, answerer.answer(q["question"], "Passages:\n[1] " + q["answer"]), cfg=cfg)
               for q in questions]
    pred_path = tmp_path / "run" / "predictions.jsonl"
    write_predictions(pred_path, records)
    return cfg_path, pred_path, questions


def test_cli_scores_and_writes_outputs_next_to_the_predictions(tmp_path, capsys):
    cfg_path, pred_path, questions = _cli_setup(tmp_path)
    code = main(["--predictions", str(pred_path), "--config", str(cfg_path)])
    out = capsys.readouterr()
    assert code == 0 and "MOCK RESULTS" in out.out and "NOT reportable" in out.out
    summary = json.loads((pred_path.parent / "eval_summary.json").read_text())
    assert summary["n_records"] == 5 and summary["is_mock"] is True and summary["reportable"] is False
    assert summary["judge"]["backend"] == "mock"
    assert len((pred_path.parent / "scored.jsonl").read_text().splitlines()) == 5


def test_cli_can_take_gold_from_a_questions_file_and_skip_the_judge(tmp_path):
    cfg_path, pred_path, questions = _cli_setup(tmp_path)
    qfile = tmp_path / "questions.json"
    qfile.write_text(json.dumps(questions), encoding="utf-8")
    out_dir = tmp_path / "elsewhere"
    assert main(["--predictions", str(pred_path), "--config", str(cfg_path), "--questions", str(qfile),
                 "--no-judge", "--out-dir", str(out_dir)]) == 0
    assert json.loads((out_dir / "eval_summary.json").read_text())["judge"] == {"enabled": False}


def test_cli_dry_run_writes_nothing_and_needs_no_key(tmp_path, capsys, no_api_key, no_network):
    cfg_path, pred_path, _ = _cli_setup(tmp_path, judge_backend="openai")
    assert main(["--predictions", str(pred_path), "--config", str(cfg_path), "--dry-run"]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert not (pred_path.parent / "scored.jsonl").exists() and not (pred_path.parent / "eval_summary.json").exists()


def test_cli_openai_judge_without_a_key_stops_with_exit_2(tmp_path, capsys, no_api_key, no_network, monkeypatch):
    import generation.llm_client as module
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: False)
    cfg_path, pred_path, _ = _cli_setup(tmp_path, judge_backend="openai")
    assert main(["--predictions", str(pred_path), "--config", str(cfg_path)]) == 2
    assert "STOPPED" in capsys.readouterr().err and not (pred_path.parent / "scored.jsonl").exists()


def test_cli_exit_codes_for_bad_input_and_for_failed_generations(tmp_path, capsys):
    cfg_path, pred_path, _ = _cli_setup(tmp_path)
    assert main(["--predictions", str(tmp_path / "missing.jsonl"), "--config", str(cfg_path)]) == 2
    recs = read_predictions(pred_path)
    recs[0] = recs[0].model_copy(update={"answer": recs[0].answer.model_copy(update={"status": "truncated", "answer": ""})})
    write_predictions(pred_path, recs)
    assert main(["--predictions", str(pred_path), "--config", str(cfg_path), "--no-judge"]) == 1
    assert "WARNING" in capsys.readouterr().err
