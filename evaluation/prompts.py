"""
Judge prompt for LLM-as-a-Judge (Phase 8, Evaluation). One fixed template for every
condition. Bump JUDGE_PROMPT_VERSION whenever the prompt or schema changes: it is part
of the cache key, so old cached verdicts are never reused.

WHAT THE JUDGE SEES - and nothing else:
    the question, the reference answer (plus acceptable alternatives, if the dataset has
    them) and the generated answer.
It never sees the retrieved context, chunks, supporting documents, graph information or
supporting facts. build_judge_messages() has no parameter through which any of those
could reach it (a test checks this).
"""

from __future__ import annotations

JUDGE_PROMPT_VERSION = "judge-v1"

JUDGE_SYSTEM_PROMPT = (
    "You grade answers to factual questions. You are given a question, a reference answer "
    "(sometimes with other acceptable answers) and a candidate answer.\n"
    "Decide whether the candidate answer is correct: it must state the same fact as the "
    "reference answer. Ignore differences in capitalisation, punctuation, articles, "
    "abbreviations or harmless extra words. The candidate is incorrect if it contradicts the "
    "reference, gives a different entity, date or number, is only partly correct, or says the "
    "answer cannot be found. Grade against the reference only; do not use outside knowledge "
    "to second-guess it.\n"
    "Return JSON with two fields:\n"
    "  - reasoning: one short sentence comparing the candidate with the reference;\n"
    "  - correct: true or false."
)

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correct": {"type": "boolean"},
    },
    "required": ["reasoning", "correct"],
    "additionalProperties": False,
}
JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "judge", "strict": True, "schema": JUDGE_JSON_SCHEMA},
}


def build_judge_messages(question: str, reference: str, aliases: list[str], candidate: str) -> list[dict]:
    """The exact messages sent for one judgement."""
    lines = [f"Question: {question}", f"Reference answer: {reference}"]
    others = [a for a in dict.fromkeys(aliases) if a.strip() and a != reference]
    if others:
        lines.append("Other acceptable answers: " + "; ".join(others))
    lines.append(f"Candidate answer: {candidate}")
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
