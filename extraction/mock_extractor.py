"""
Deterministic mock extractor (Phase 4) - for TESTING ONLY.

HEURISTIC, NOT A MODEL OF LLM QUALITY. It exists so the complete Phase 4
pipeline (selection -> budget gate -> cache -> extraction -> saved results) can
be exercised with no API key, no network and no cost.

What it does, using only the chunk text:
  1. split the text into sentences;
  2. entities  = capitalised phrases (the same cheap rule FastGraphRAG's regex
                 fallback uses, src/text_utils.py);
  3. entity type = picked from a hash of the name. This is ARBITRARY but
                 deterministic; mock types carry no meaning. It just makes sure
                 all six types show up in tests;
  4. relationships = consecutive entities inside one sentence are linked
                 ("mentioned together in one sentence").
Same text in -> identical output, always.

Every result is stamped extractor="mock". Mock output must NEVER appear in a
reported benchmark result.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

from extraction.base_extractor import Extractor
from extraction.pricing import estimate_tokens
from extraction.schemas import (ENTITY_TYPES, SCHEMA_VERSION, Entity, ExtractionInput,
                                ExtractionResult, Relationship, Usage, clean_extraction)
from src.text_utils import capitalized_phrases, normalize_name

# Bump if the mock's behaviour changes, so stale mock cache entries are not reused.
MOCK_VERSION = "mock-v1"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _mock_type(name: str) -> str:
    digest = hashlib.sha1(normalize_name(name).encode("utf-8")).hexdigest()
    return ENTITY_TYPES[int(digest[:8], 16) % len(ENTITY_TYPES)]


class MockExtractor(Extractor):
    name = "mock"
    model = MOCK_VERSION

    def cache_settings(self) -> dict:
        return {"extractor": self.name, "model": self.model, "prompt_version": "n/a",
                "schema_version": SCHEMA_VERSION, "temperature": 0.0, "max_output_tokens": 0}

    def extract(self, item: ExtractionInput) -> ExtractionResult:
        started = time.perf_counter()
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        for sentence in _SENTENCE_SPLIT.split(item.text):
            names: list[str] = []
            for phrase in capitalized_phrases(sentence):
                if normalize_name(phrase) not in {normalize_name(n) for n in names}:
                    names.append(phrase)
            entities.extend(Entity(name=n, type=_mock_type(n), description="") for n in names)
            relationships.extend(
                Relationship(source=a, target=b, description="mentioned together in one sentence")
                for a, b in zip(names, names[1:])
            )

        entities, relationships, issues, notes = clean_extraction(entities, relationships)
        payload = json.dumps({"entities": [e.model_dump() for e in entities],
                              "relationships": [r.model_dump() for r in relationships]})
        return ExtractionResult(
            chunk_id=item.chunk_id, status="ok",
            entities=entities, relationships=relationships,
            validation_issues=issues, validation_notes=notes,
            extractor=self.name, model=self.model, prompt_version="n/a",
            # Fake token counts (so the accounting code is exercised); no money is spent.
            usage=Usage(input_tokens=estimate_tokens(item.text),
                        output_tokens=estimate_tokens(payload), cost_usd=0.0),
            runtime_seconds=time.perf_counter() - started,
            attempts=1,
        )
