"""Evaluation (Phase 8): Exact Match, token F1 and LLM-as-a-Judge over generated answers.

    PredictionRecord (wraps AnswerResult + question_id + condition metadata)
        + gold answers (evaluation side only)
        -> scored.jsonl + eval_summary.json

This package scores predictions. It does not generate answers, run strategies or loop over
conditions (that is the experiment runner, a later phase).
"""

from evaluation.judge import Judge, build_judge
from evaluation.metrics import best_scores, exact_match, f1_score
from evaluation.normalize import normalize_answer
from evaluation.schemas import (GoldAnswer, JudgeResult, PredictionRecord, RetrievalSummary, ScoredRecord,
                                read_predictions, write_predictions)
from evaluation.score import (EvaluationRun, NotReportableError, assert_reportable, gold_from_questions,
                              load_gold_file, score_predictions, score_record, write_outputs)

__all__ = ["EvaluationRun", "GoldAnswer", "Judge", "JudgeResult", "NotReportableError", "PredictionRecord",
           "RetrievalSummary", "ScoredRecord", "assert_reportable", "best_scores", "build_judge", "exact_match",
           "f1_score", "gold_from_questions", "load_gold_file", "normalize_answer", "read_predictions",
           "score_predictions", "score_record", "write_outputs", "write_predictions"]
