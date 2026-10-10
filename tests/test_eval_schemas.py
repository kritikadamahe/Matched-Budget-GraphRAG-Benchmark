"""Phase 8 Evaluation: PredictionRecord construction, gold data, JSONL round trip."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from pydantic import ValidationError

from evaluation.schemas import (GoldAnswer, PredictionRecord, read_predictions, summarize_retrieval,
                                write_predictions)
from generation.schemas import AnswerResult
from retrieval import RetrievalResult
from src.config import ExperimentConfig
from tests.helpers_evaluation import Q, make_answer, make_prediction
from tests.helpers_extraction import make_cfg

QUESTION = {"question_id": "q1", "question": Q, "answer": "Chicago"}


def test_answer_result_is_unchanged_and_has_no_question_id():
    # Decision: question_id must NOT be added to AnswerResult (cached results would carry a stale id).
    assert "question_id" not in AnswerResult.model_fields
    assert "question_id" in PredictionRecord.model_fields


def test_build_wraps_the_answer_and_adds_condition_metadata():
    cfg = ExperimentConfig(experiment_id="t", seed=7, dataset="hotpotqa", num_questions=5, strategy="ketrag",
                           budget=0.1, ketrag_mode="tfidf", embedding_backend="tfidf")
    answer = make_answer("Chicago")
    retrieval = RetrievalResult(question=Q, context="x y z", seeds=["a", "b"], top_entities=["a"],
                                facts=["f1", "f2", "f3"], chunk_ids=["c1", "c2"], n_words=3)
    rec = PredictionRecord.build(QUESTION, answer, cfg=cfg, retrieval=retrieval,
                                 generation_settings={"task": "answer", "model": "gpt-4o-mini", "temperature": 0.0})
    assert rec.question_id == "q1" and rec.answer == answer
    assert (rec.strategy, rec.budget, rec.seed, rec.dataset) == ("ketrag", 0.1, 7, "hotpotqa")
    assert rec.corpus_id == "hotpotqa_mock_5q_seed7_w250o40"
    assert rec.run_id and "ketrag" in rec.run_id
    assert rec.retrieval.n_facts == 3 and rec.retrieval.n_passages == 2 and rec.retrieval.n_seeds == 2
    assert rec.retrieval.passage_chunk_ids == ["c1", "c2"]
    assert rec.retrieval_settings["method"] == "personalised_pagerank" and rec.retrieval_settings["top_m"] == 20
    assert rec.generation_settings["temperature"] == 0.0


def test_retrieval_summary_never_holds_the_context_text():
    retrieval = RetrievalResult(question=Q, context="SECRET CONTEXT TEXT", chunk_ids=["c1"], n_words=3)
    rec = PredictionRecord.build(QUESTION, make_answer(), retrieval=retrieval)
    assert "SECRET CONTEXT TEXT" not in rec.model_dump_json()


def test_native_reference_point_has_no_budget_and_its_own_settings():
    cfg = make_cfg("lazygraphrag_native")
    rec = PredictionRecord.build(QUESTION, make_answer(), cfg=cfg, retrieval_extra={"relevance_tests": 9, "n_relevant": 2})
    assert rec.strategy == "lazygraphrag_native" and rec.budget is None
    assert rec.retrieval_settings["method"] == "native_lazygraphrag"
    assert rec.retrieval.extra == {"relevance_tests": 9, "n_relevant": 2}


def test_explicit_run_id_wins_and_cfg_is_optional():
    rec = PredictionRecord.build(QUESTION, make_answer(), run_id="my-run")
    assert rec.run_id == "my-run" and rec.strategy is None and rec.retrieval is None


def test_build_refuses_a_mismatched_question():
    with pytest.raises(ValueError, match="different question"):
        PredictionRecord.build({"question_id": "q9", "question": "Something else?"}, make_answer())


def test_is_mock_follows_the_answer_backend():
    assert PredictionRecord.build(QUESTION, make_answer(backend="mock")).is_mock is True
    assert PredictionRecord.build(QUESTION, make_answer(backend="openai")).is_mock is False
    assert '"is_mock":true' in PredictionRecord.build(QUESTION, make_answer(backend="mock")).model_dump_json().replace(" ", "")


def test_jsonl_round_trip_preserves_everything_including_failures_and_unknown_usage(tmp_path):
    from extraction.schemas import Usage
    ok = make_prediction("Chicago", qid="q1")
    failed = make_prediction("", qid="q2", question="Another question?", status="truncated",
                             error="reply truncated", usage=Usage.unknown())
    path = tmp_path / "p" / "predictions.jsonl"
    write_predictions(path, [ok, failed])
    back = read_predictions(path)
    assert back == [ok, failed]
    assert back[1].answer.status == "truncated" and back[1].answer.usage.known is False
    assert back[1].answer.usage.cost_usd is None                 # unknown stays unknown, not 0.0


def test_summarize_retrieval_none_cases():
    assert summarize_retrieval(None) is None
    assert summarize_retrieval(None, {"n_relevant": 1}, n_words=5).n_words == 5


def test_gold_answer_aliases_and_validation():
    g = GoldAnswer(question_id="q", answer="USA", aliases=["US", "USA", " ", "United States"])
    assert g.all_answers == ["USA", "US", "United States"]
    with pytest.raises(ValidationError):
        GoldAnswer(question_id="q", answer="   ")
    # no gold-like field exists on the prediction record
    assert not {"gold", "gold_answer", "answer_aliases"} & set(PredictionRecord.model_fields)


def test_no_earlier_phase_imports_evaluation():
    """Gold answers exist only on the evaluation side: nothing upstream may depend on it."""
    import re
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"^\s*(from|import)\s+evaluation\b", re.M)
    offenders = [str(p.relative_to(root)) for pkg in ("src", "extraction", "graph", "retrieval", "generation",
                                                      "strategies", "native")
                 for p in (root / pkg).rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
