"""
Prompts for query-time LLM calls (blueprint Phase 10, §K): answer generation and
the native LazyGraphRAG relevance check. One fixed template each, identical for
every strategy and budget. Bump the version string whenever a prompt or schema
changes - it is part of the cache key, so old cached replies are never reused.

Agreed for Phase 10:
- Strict grounding: answer ONLY from the context, else "not found". GPT-4o-mini has
  memorised much of Wikipedia; answering from memory would hide the effect of the
  extraction budget, which is the thing being measured.
- Brief reasoning first, then the answer (multi-hop questions chain 2-4 facts).
  Only `answer` is scored; `reasoning` is kept for inspection.
- Exact-match friendly format: shortest exact span; "yes"/"no" for yes/no questions.
"""

from __future__ import annotations

ANSWER_PROMPT_VERSION = "answer-v1"
RELEVANCE_PROMPT_VERSION = "relevance-v1"
NOT_FOUND = "not found"

ANSWER_SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context (facts and passages). "
    "Do not use any outside knowledge.\n"
    "Return JSON with two fields:\n"
    "  - reasoning: one or two short sentences linking the facts from the context that lead to the answer;\n"
    "  - answer: the shortest exact answer, copied from the context where possible (a name, date, "
    "number or short phrase, not a sentence). For yes/no questions answer exactly \"yes\" or \"no\".\n"
    f"If the context does not contain the answer, set answer to exactly \"{NOT_FOUND}\"."
)

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["reasoning", "answer"],
    "additionalProperties": False,
}
ANSWER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "strict": True, "schema": ANSWER_JSON_SCHEMA},
}

RELEVANCE_SYSTEM_PROMPT = (
    "You decide whether a text passage contains information that helps answer a question, "
    "even if it answers only part of it. Use only the passage. "
    "Return JSON with one field, relevant: true or false."
)
RELEVANCE_JSON_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
    "additionalProperties": False,
}
RELEVANCE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "relevance", "strict": True, "schema": RELEVANCE_JSON_SCHEMA},
}


def build_answer_messages(question: str, context: str) -> list[dict]:
    """The exact messages sent for one question (blueprint §K template)."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
    ]


def build_relevance_messages(question: str, passage: str) -> list[dict]:
    return [
        {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nPassage:\n{passage}"},
    ]
