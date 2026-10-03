"""
Extraction prompt and the JSON schema the API must follow (Phase 4).

CATEGORY B - ADAPTATION, not a reproduction. This is "GraphRAG-style"
entity/relationship extraction, but the wording, the six fixed entity types and
the simplified output are OUR OWN. It is not Microsoft GraphRAG's extraction
prompt and not KET-RAG's code; it also has no "gleaning" (repeat passes) and no
claim extraction. In the report, call it "an LLM entity/relationship extraction
step in the style of GraphRAG", never "GraphRAG's extractor".

What IS faithful to the benchmark design: every selection strategy is extracted
with this identical prompt, model and settings.

If you edit SYSTEM_PROMPT or the schema, bump PROMPT_VERSION. It is part of the
cache key, so old cached extractions are not reused for a changed prompt.
"""

from __future__ import annotations

from extraction.schemas import ENTITY_TYPES

PROMPT_VERSION = "extract-v1"

SYSTEM_PROMPT = (
    "You extract a small knowledge graph from a passage of text.\n"
    "Return JSON with two lists.\n"
    "\n"
    "entities: every distinct named entity in the passage. Each entity has:\n"
    "  - name: the entity's name as written in the passage (use its most complete form);\n"
    f"  - type: exactly one of {', '.join(ENTITY_TYPES)}. "
    "WORK means a film, book, song, album, artwork or similar creation. "
    "Use OTHER for anything else worth recording;\n"
    "  - description: one short sentence about the entity, using only the passage.\n"
    "\n"
    "relationships: facts in the passage that connect two entities from your entity list. Each has:\n"
    "  - source and target: copied EXACTLY from the names in your entity list;\n"
    "  - description: a short phrase saying how they are related.\n"
    "\n"
    "Use only information stated in the passage. Do not invent entities or facts. "
    "If the passage has none, return empty lists."
)

EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "target", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": False,
}

# Passed to the API as response_format: the model is constrained to this schema.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "extraction", "strict": True, "schema": EXTRACTION_JSON_SCHEMA},
}


def build_messages(text: str) -> list[dict]:
    """The exact chat messages sent for one chunk. Only the chunk text varies."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage:\n{text}"},
    ]
