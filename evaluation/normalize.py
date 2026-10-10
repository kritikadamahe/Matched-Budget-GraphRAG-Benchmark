"""
Answer normalisation for Exact Match and token F1 (Phase 8, Evaluation).

SQuAD-style, as used by the official HotpotQA / MuSiQue scorers: lowercase,
remove punctuation, remove the articles "a", "an", "the", collapse whitespace.
Two small, documented extensions so typographic variants do not count as errors:
  - Unicode NFKC first (full-width characters, ligatures);
  - ALL Unicode punctuation is removed, not only ASCII (so a curly apostrophe or
    an en dash behaves like its ASCII twin).
Punctuation is deleted, not replaced by a space: "U.S." -> "us", "O'Neil" -> "oneil".

This is deliberately NOT generation.schemas.clean_answer(). That function tidies a
model reply for display and keeps case and articles; scoring uses this one.
"""

from __future__ import annotations

import re
import string
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_ASCII_PUNCTUATION = set(string.punctuation)


def _is_punctuation(ch: str) -> bool:
    return ch in _ASCII_PUNCTUATION or unicodedata.category(ch).startswith("P")


def normalize_answer(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = "".join(ch for ch in s if not _is_punctuation(ch))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def answer_tokens(text: str) -> list[str]:
    return normalize_answer(text).split()
