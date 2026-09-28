"""
Small, dependency-free text helpers shared by the FastGraphRAG strategy and the
native LazyGraphRAG module.

capitalized_phrases() is a cheap, LLM-free entity guesser used as a FALLBACK
when spaCy (or its en_core_web_sm model) is not installed. It is deliberately
simple: runs of Capitalised Words joined by spaces, never across a line break.
"""

from __future__ import annotations
import re

# "Christopher Nolan", "Bank of America", "University of Alabama"
_CAP_PHRASE = re.compile(r"\b[A-Z][\w'\-]*(?:[ \t]+(?:(?:of|the|de|van|von)[ \t]+)?[A-Z][\w'\-]*)*")
# Capitalised only because they start a sentence - not entities.
_NOT_ENTITIES = {"The", "A", "An", "He", "She", "It", "They", "His", "Her", "In", "On",
                 "What", "Which", "Who", "When", "Where", "How", "This", "That"}


def capitalized_phrases(text: str) -> list[str]:
    out = []
    for m in _CAP_PHRASE.finditer(text):
        words = m.group(0).split()
        if words and words[0] in _NOT_ENTITIES:
            words = words[1:]
        phrase = " ".join(words)
        if len(phrase) > 1:
            out.append(phrase)
    return out


def normalize_name(name: str) -> str:
    """Lowercase + collapse whitespace, so 'Tom  Hanks' and 'tom hanks' match."""
    return " ".join(name.lower().split())


def load_spacy(model: str = "en_core_web_sm", disable: list[str] | None = None):
    """Return a spaCy pipeline, or None if spaCy/the model is not installed."""
    try:
        import spacy
        return spacy.load(model, disable=disable or [])
    except (ImportError, OSError):
        return None
