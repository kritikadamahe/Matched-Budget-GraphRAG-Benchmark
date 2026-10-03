"""
Structured schemas for LLM extraction (Phase 4).

WHY THIS EXISTS:
Everything the extractor takes in and hands back is described here, with
pydantic validating it. If the LLM returns something that does not fit these
shapes, validation fails loudly instead of letting bad data flow into the
knowledge graph that later phases will build.

THE FOUR SHAPES
- ExtractionInput  : what the extractor is ALLOWED to see - chunk_id and text,
                     nothing else. Gold labels, questions and answers cannot be
                     passed in (extra fields are rejected). This is a leakage
                     guard (blueprint section S).
- Entity           : one named thing found in a chunk. `type` is one of six
                     fixed values (see ENTITY_TYPES).
- Relationship     : a link between two entities, described in words.
- ExtractionResult : everything about one chunk's extraction attempt: the
                     entities and relationships, status, token usage, cost,
                     model, runtime and number of attempts.

RELATIONSHIP CLEAN-UP (decision approved for Phase 4)
A relationship whose source or target is not in the extracted entity list is
DROPPED and counted in `validation_issues`. It does NOT trigger a retry - the
rest of the extraction is still good, and retrying would only cost money.
Self-relationships (source == target) are dropped and counted the same way.
Duplicate entities (same name ignoring case/spacing) are merged silently; that
is not an error.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.text_utils import normalize_name

# Bump this if the shape of Entity/Relationship/ExtractionResult changes, so old
# cache entries are not reused with a new meaning (it is part of the cache key).
# "2": Usage can now be UNKNOWN (None) instead of silently 0, and "truncated" is a status.
#      Bumping it means no cache entry written under the old meaning is ever reused.
SCHEMA_VERSION = "2"

EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "WORK", "OTHER"]
ENTITY_TYPES: tuple[str, ...] = get_args(EntityType)

# "ok"         - extraction succeeded and validated
# "api_error"  - the API kept failing (rate limit, timeout, 5xx) after all retries
# "malformed"  - the model answered but not with valid data after all retries
# "refused"    - the model refused to answer after all retries
# "truncated"  - the reply hit the output-token limit. NOT retried: the same input
#                would be cut off again, so a retry only costs money. Fix = raise
#                extraction_max_output_tokens.
ExtractionStatus = Literal["ok", "api_error", "malformed", "refused", "truncated"]

MAX_VALIDATION_NOTES = 20


class ExtractionInput(BaseModel):
    """The ONLY thing an extractor receives. Deliberately minimal."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    text: str


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: EntityType
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("entity name must not be blank")
        return v


class Relationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    description: str

    @field_validator("source", "target")
    @classmethod
    def _endpoint_not_blank(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("relationship endpoint must not be blank")
        return v


class RawExtraction(BaseModel):
    """What the model must return (and what the JSON schema sent to the API describes)."""
    model_config = ConfigDict(extra="forbid")

    entities: list[Entity]
    relationships: list[Relationship]


class Usage(BaseModel):
    """Token usage and dollar cost of one extraction (every attempt included).

    None means UNKNOWN, never "zero". If the API response carries no usage
    information, all three fields are None: we do not know what was spent, and we
    must not record that as a free extraction. Token counts are never estimated
    here - only values the API actually reported (or None) are stored.
    """
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    cost_usd: float | None = 0.0

    @model_validator(mode="after")
    def _fully_known_or_fully_unknown(self):
        values = (self.input_tokens, self.output_tokens, self.cost_usd)
        if any(v is None for v in values) and not all(v is None for v in values):
            raise ValueError("usage must be fully known or fully unknown (all None), never half-known")
        return self

    @property
    def known(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None and self.cost_usd is not None

    @classmethod
    def unknown(cls) -> "Usage":
        return cls(input_tokens=None, output_tokens=None, cost_usd=None)


class ExtractionResult(BaseModel):
    chunk_id: str
    status: ExtractionStatus
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    validation_issues: int = 0                       # relationships dropped by clean_extraction
    validation_notes: list[str] = Field(default_factory=list)
    error: str | None = None
    extractor: str                                   # "mock" or "openai"
    model: str
    prompt_version: str
    schema_version: str = SCHEMA_VERSION
    usage: Usage = Field(default_factory=Usage)      # every attempt, failed ones too; None fields = unknown
    runtime_seconds: float = 0.0
    attempts: int = 1
    cache_hit: bool = False                          # set when loaded from the cache


def clean_extraction(
    entities: list[Entity], relationships: list[Relationship]
) -> tuple[list[Entity], list[Relationship], int, list[str]]:
    """Merge duplicate entities and drop relationships that point at unknown
    entities (or at themselves). Returns (entities, relationships, issues, notes).

    Relationship endpoints are rewritten to the exact entity name, so a later
    graph builder can match them by simple string equality.
    """
    canonical: dict[str, str] = {}
    kept_entities: list[Entity] = []
    for e in entities:
        key = normalize_name(e.name)
        if key in canonical:
            continue                                  # duplicate entity: merged silently
        canonical[key] = e.name
        kept_entities.append(e)

    issues = 0
    notes: list[str] = []
    kept: list[Relationship] = []
    seen: set[tuple[str, str, str]] = set()

    def note(msg: str) -> None:
        if len(notes) < MAX_VALIDATION_NOTES:
            notes.append(msg)

    for r in relationships:
        s, t = normalize_name(r.source), normalize_name(r.target)
        if s not in canonical or t not in canonical:
            missing = r.source if s not in canonical else r.target
            issues += 1
            note(f"dropped relationship {r.source!r} -> {r.target!r}: unknown entity {missing!r}")
            continue
        if s == t:
            issues += 1
            note(f"dropped self-relationship on {r.source!r}")
            continue
        signature = (s, t, normalize_name(r.description))
        if signature in seen:
            continue                                  # exact duplicate: merged silently
        seen.add(signature)
        kept.append(Relationship(source=canonical[s], target=canonical[t], description=r.description))

    return kept_entities, kept, issues, notes
