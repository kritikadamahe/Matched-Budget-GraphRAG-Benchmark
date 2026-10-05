"""Result shapes for query-time LLM calls (blueprint Phase 10). Token usage reuses
the extraction Usage model, so "unknown is never free" holds here too."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from extraction.schemas import Usage
from generation.prompts import NOT_FOUND

# "ok"                    - the model answered with valid JSON
# "skipped_empty_context" - retrieval returned nothing, so no call was made (answer "not found", $0)
# "api_error" / "malformed" / "refused" / "truncated" - as in extraction (see extraction/schemas.py)
GenerationStatus = Literal["ok", "skipped_empty_context", "api_error", "malformed", "refused", "truncated"]


class AnswerOutput(BaseModel):
    """What the model must return for answer generation."""
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    answer: str


class RelevanceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevant: bool


class _CallInfo(BaseModel):
    status: GenerationStatus
    error: str | None = None
    backend: str                                   # "mock" or "openai"
    model: str
    prompt_version: str
    prompt_sha256: str                             # hash of the exact messages sent (reproducibility)
    usage: Usage = Field(default_factory=Usage)    # every attempt; None fields = unknown
    attempts: int = 1
    runtime_seconds: float = 0.0
    cache_hit: bool = False


class AnswerResult(_CallInfo):
    question: str
    answer: str = ""                               # "" when the call failed (scored as wrong)
    reasoning: str = ""
    context_words: int = 0


class RelevanceResult(_CallInfo):
    question: str
    relevant: bool = False                          # a failed check counts as "not relevant"


_NOT_FOUND_VARIANTS = {"not found", "notfound", "unknown", "not mentioned", "cannot be determined",
                       "not enough information", "no answer", "n/a"}


def clean_answer(text: str) -> str:
    """Tidy a model answer for scoring: collapse whitespace, strip wrapping quotes and
    one trailing period, and map "no answer" variants to the canonical "not found"."""
    s = " ".join(text.split()).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if s.endswith(".") and not re.search(r"\b[A-Za-z]\.$", s):   # keep "Washington D.C."
        s = s[:-1].rstrip()
    if s.lower().rstrip(".") in _NOT_FOUND_VARIANTS:
        return NOT_FOUND
    return s
