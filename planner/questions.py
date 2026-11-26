"""
Test questions for a pilot, generated from the user's own chunks.

Each question is written from one chunk, or from two chunks of different documents that
mention the same name (a two-hop question), so its source chunks are known: they play the
role of HotpotQA's gold paragraphs (evidence recall) and the generated short answer is the
gold answer (EM / F1). Questions are generated from the whole pilot sample BEFORE any
selection, so no selector is favoured. Replies are cached like answers (cache/answers/...).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from generation.cache import QueryCache
from generation.llm_client import ChatJSONClient

QGEN_PROMPT_VERSION = "qgen-v1"
MAX_ANSWER_WORDS = 8

QGEN_SYSTEM_PROMPT = (
    "You write evaluation questions for a question-answering system over a document collection.\n"
    "Given one or two passages, write ONE question whose answer is a short span (a name, date, number "
    "or short phrase of at most 8 words) stated in the passages.\n"
    "If two passages are given, the question must need BOTH: combine a fact from each (a two-hop "
    "question), and do not mention the passage titles.\n"
    "The question must make sense on its own, without seeing the passages (no 'this passage', 'the "
    "author', 'the document').\n"
    "Return JSON: question, answer, answerable (false if no good question is possible)."
)
QGEN_JSON_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}, "answerable": {"type": "boolean"}},
    "required": ["question", "answer", "answerable"],
    "additionalProperties": False,
}
QGEN_RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "qgen", "strict": True, "schema": QGEN_JSON_SCHEMA}}

_NAME = re.compile(r"\b[A-Z][a-zA-Z'’-]+(?:\s+[A-Z][a-zA-Z'’-]+)+\b")


class QGenOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str
    answerable: bool


def plan_sources(chunks, n_questions: int, seed: int = 42) -> list[list]:
    """Which chunk(s) each question is written from: about half two-hop pairs (two documents sharing
    a multi-word name), the rest single chunks. Over-provisioned by 50% for rejected questions."""
    rng = random.Random(seed)
    want = int(n_questions * 1.5) + 2
    by_name = defaultdict(list)
    for c in chunks:
        for name in set(_NAME.findall(c.text.split(": ", 1)[-1])):
            by_name[name].append(c)
    pairs = []
    for name, cs in by_name.items():
        docs = {}
        for c in cs:
            docs.setdefault(c.source_doc_id, c)
        if 2 <= len(docs) <= 6:                     # shared, but not a name that is everywhere
            a, b = rng.sample(list(docs.values()), 2)
            pairs.append([a, b])
    rng.shuffle(pairs)
    pairs = pairs[: want // 2]
    used = {c.chunk_id for p in pairs for c in p}
    singles = [c for c in chunks if c.chunk_id not in used and len(c.text.split()) >= 40]
    rng.shuffle(singles)
    plan = pairs + [[c] for c in singles[: want - len(pairs)]]
    rng.shuffle(plan)
    return plan


def _messages(sources) -> list[dict]:
    body = "\n\n".join(f"Passage {i + 1}:\n{c.text}" for i, c in enumerate(sources))
    return [{"role": "system", "content": QGEN_SYSTEM_PROMPT}, {"role": "user", "content": body}]


def _mock(sources) -> QGenOutput:
    names = _NAME.findall(sources[-1].text.split(": ", 1)[-1]) or [sources[-1].source_title]
    return QGenOutput(question=f"Which name appears in the text about {sources[0].source_title}?",
                      answer=names[0], answerable=True)


class QuestionGenerator:
    def __init__(self, backend: str, client: ChatJSONClient | None, cache: QueryCache, model: str):
        self.backend, self.client, self.cache, self.model = backend, client, cache, model
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}

    def settings(self) -> dict:
        return {"task": "qgen", "backend": self.backend, "model": self.model, "prompt_version": QGEN_PROMPT_VERSION}

    def one(self, sources) -> dict | None:
        inputs = {"chunks": [c.text for c in sources]}
        cached = self.cache.get(self.settings(), inputs)
        if cached is not None:
            self.usage["cache_hits"] += 1
            out = QGenOutput(**cached["parsed"])
        elif self.backend == "mock":
            out = _mock(sources)
        else:
            call = self.client.call(_messages(sources), QGEN_RESPONSE_FORMAT, QGenOutput, label="question generation")
            self.usage["calls"] += 1
            if call.usage is not None:
                self.usage["input_tokens"] += call.usage.input_tokens or 0
                self.usage["output_tokens"] += call.usage.output_tokens or 0
                self.usage["usd"] += call.usage.cost_usd or 0.0
            if call.status != "ok":
                return None
            out = call.parsed
            self.cache.put(self.settings(), inputs, {"status": "ok", "parsed": out.model_dump()}, True)
        answer = out.answer.strip().strip(".")
        if not out.answerable or not answer or len(answer.split()) > MAX_ANSWER_WORDS or answer.lower() == "not found":
            return None
        return {"question": out.question.strip(), "answer": answer,
                "gold_chunks": [c.chunk_id for c in sources], "gold_titles": [c.source_title for c in sources],
                "hops": len(sources)}


def generate(chunks, n_questions: int, gen: QuestionGenerator, progress=None) -> list[dict]:
    plan = plan_sources(chunks, n_questions)
    out, seen = [], set()
    for i, sources in enumerate(plan):
        if len(out) >= n_questions:
            break
        q = gen.one(sources)
        if q and q["question"].lower() not in seen:
            seen.add(q["question"].lower())
            q["id"] = f"q{len(out) + 1:03d}"
            out.append(q)
        if progress:
            progress(min(len(out), n_questions), n_questions)
    return out
