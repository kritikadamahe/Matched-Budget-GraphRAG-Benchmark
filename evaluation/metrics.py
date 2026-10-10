"""
Exact Match and token-level F1 (Phase 8, Evaluation). Pure functions: no I/O, no LLM.

- EM  = 1.0 if the normalised prediction equals the normalised gold answer, else 0.0.
- F1  = harmonic mean of token precision and recall (bag-of-tokens overlap).
- yes/no: as in the official HotpotQA scorer, a yes/no answer gets NO partial credit.
  If either side normalises to exactly "yes" or "no", F1 is 1.0 only when both are equal.
- aliases: a question may have several acceptable answers (MuSiQue ships aliases). The
  score is the BEST over the gold answer and every alias; EM and F1 are maximised
  independently.

Whether a reply is an abstention ("not found"), empty, or a failed call is decided one
level up (evaluation/score.py), because that depends on the AnswerResult, not on text.
"""

from __future__ import annotations

from collections import Counter

from evaluation.normalize import answer_tokens, normalize_answer

YES_NO = frozenset({"yes", "no"})


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    norm_pred, norm_gold = normalize_answer(prediction), normalize_answer(gold)
    if norm_pred in YES_NO or norm_gold in YES_NO:
        return 1.0 if norm_pred == norm_gold else 0.0
    pred, ref = norm_pred.split(), norm_gold.split()
    if not pred or not ref:
        return 1.0 if pred == ref else 0.0          # SQuAD convention (both empty = match)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_scores(prediction: str, golds: list[str]) -> tuple[float, float]:
    """(EM, F1), each the maximum over all acceptable gold answers."""
    golds = [g for g in golds if g and g.strip()]
    if not golds:
        raise ValueError("at least one non-blank gold answer is required")
    return (max(exact_match(prediction, g) for g in golds),
            max(f1_score(prediction, g) for g in golds))


__all__ = ["YES_NO", "answer_tokens", "best_scores", "exact_match", "f1_score", "normalize_answer"]
