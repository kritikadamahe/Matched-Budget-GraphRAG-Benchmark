"""Phase 4: schemas, the clean-up step, prompt/schema consistency, and the new config fields."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import get_args

import pytest
from pydantic import ValidationError

from extraction.prompts import (EXTRACTION_JSON_SCHEMA, PROMPT_VERSION, RESPONSE_FORMAT,
                                SYSTEM_PROMPT, build_messages)
from extraction.schemas import (ENTITY_TYPES, SCHEMA_VERSION, Entity, ExtractionInput,
                                ExtractionResult, ExtractionStatus, RawExtraction, Relationship,
                                Usage, clean_extraction)
from src.config import ExperimentConfig, load_config

pytestmark = pytest.mark.usefixtures("no_network", "no_api_key")
PROJECT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ schemas
def test_entity_types_are_exactly_the_approved_six():
    assert ENTITY_TYPES == ("PERSON", "ORGANIZATION", "LOCATION", "EVENT", "WORK", "OTHER")


@pytest.mark.parametrize("t", ENTITY_TYPES)
def test_every_approved_type_is_accepted(t):
    assert Entity(name="X", type=t).type == t


def test_unknown_entity_type_rejected():
    with pytest.raises(ValidationError):
        Entity(name="X", type="ANIMAL")


def test_missing_and_blank_fields_rejected():
    with pytest.raises(ValidationError):
        Entity(type="PERSON")                                   # no name
    with pytest.raises(ValidationError):
        Entity(name="   ", type="PERSON")                       # blank name
    with pytest.raises(ValidationError):
        Relationship(source="A", target="B")                    # no description
    with pytest.raises(ValidationError):
        RawExtraction(entities=[])                              # no relationships list


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        Entity(name="X", type="PERSON", confidence=0.9)


def test_extraction_input_has_only_chunk_id_and_text_and_rejects_gold_labels():
    # Leakage guard: the extractor cannot even be handed gold labels or questions.
    assert set(ExtractionInput.model_fields) == {"chunk_id", "text"}
    with pytest.raises(ValidationError):
        ExtractionInput(chunk_id="c", text="t", is_gold_for_question_ids=["q1"])
    with pytest.raises(ValidationError):
        ExtractionInput(chunk_id="c", text="t", question="what?")


def test_result_round_trips_through_json():
    r = ExtractionResult(
        chunk_id="c1", status="ok", entities=[Entity(name="A", type="PERSON")],
        relationships=[], extractor="mock", model="m", prompt_version="p",
        usage=Usage(input_tokens=5, output_tokens=7, cost_usd=0.001),
    )
    assert ExtractionResult.model_validate_json(r.model_dump_json()) == r


def test_unknown_status_rejected():
    with pytest.raises(ValidationError):
        ExtractionResult(chunk_id="c", status="great", extractor="x", model="m", prompt_version="p")


# ------------------------------------------- truncated status + unknown usage
def test_truncated_is_a_valid_status_next_to_the_others():
    assert set(get_args(ExtractionStatus)) == {"ok", "api_error", "malformed", "refused", "truncated"}
    r = ExtractionResult(chunk_id="c", status="truncated", extractor="openai", model="m",
                         prompt_version="p", usage=Usage.unknown())
    assert r.status == "truncated"


def test_usage_defaults_to_known_zero_for_a_result_that_never_called_an_api():
    u = Usage()
    assert u.known is True and (u.input_tokens, u.output_tokens, u.cost_usd) == (0, 0, 0.0)


def test_unknown_usage_is_none_never_zero():
    u = Usage.unknown()
    assert u.known is False
    assert u.input_tokens is None and u.output_tokens is None and u.cost_usd is None
    assert u.cost_usd != 0.0                       # None is not "free"


def test_adding_an_unknown_cost_to_a_number_fails_loudly_instead_of_counting_as_zero():
    with pytest.raises(TypeError):
        0.5 + Usage.unknown().cost_usd


@pytest.mark.parametrize("half_known", [
    dict(input_tokens=None, output_tokens=5, cost_usd=0.1),
    dict(input_tokens=5, output_tokens=None, cost_usd=0.1),
    dict(input_tokens=5, output_tokens=7, cost_usd=None),
])
def test_half_known_usage_is_rejected(half_known):
    with pytest.raises(ValidationError):
        Usage(**half_known)


def test_unknown_usage_survives_a_json_round_trip_as_null_not_zero():
    r = ExtractionResult(chunk_id="c", status="ok", extractor="openai", model="m",
                         prompt_version="p", usage=Usage.unknown())
    dumped = r.model_dump_json()
    assert '"cost_usd":null' in dumped.replace(" ", "")
    back = ExtractionResult.model_validate_json(dumped)
    assert back == r and back.usage.known is False and back.usage.cost_usd is None


def test_schema_version_was_bumped_so_pre_fix_cache_entries_are_never_reused():
    assert SCHEMA_VERSION == "2"


# ---------------------------------------------------- clean_extraction rules
def _e(*names):
    return [Entity(name=n, type="OTHER") for n in names]


def test_relationship_with_unknown_entity_is_dropped_and_counted():
    ents, rels, issues, notes = clean_extraction(
        _e("Alpha", "Beta"),
        [Relationship(source="Alpha", target="Beta", description="knows"),
         Relationship(source="Alpha", target="Ghost", description="haunts")],
    )
    assert [(r.source, r.target) for r in rels] == [("Alpha", "Beta")]
    assert issues == 1 and "Ghost" in notes[0]
    assert len(ents) == 2


def test_unknown_source_also_dropped():
    _, rels, issues, _ = clean_extraction(
        _e("Beta"), [Relationship(source="Ghost", target="Beta", description="x")])
    assert rels == [] and issues == 1


def test_endpoint_matching_ignores_case_and_spacing_and_uses_canonical_name():
    _, rels, issues, _ = clean_extraction(
        _e("Tom Hanks", "Forrest Gump"),
        [Relationship(source="tom  hanks", target="FORREST GUMP", description="starred in")])
    assert issues == 0
    assert (rels[0].source, rels[0].target) == ("Tom Hanks", "Forrest Gump")


def test_self_relationship_dropped_and_counted():
    _, rels, issues, _ = clean_extraction(
        _e("Alpha"), [Relationship(source="Alpha", target="alpha", description="is")])
    assert rels == [] and issues == 1


def test_duplicate_entities_and_relationships_merged_without_counting_issues():
    ents, rels, issues, _ = clean_extraction(
        _e("Alpha", "alpha", "Beta"),
        [Relationship(source="Alpha", target="Beta", description="knows"),
         Relationship(source="alpha", target="beta", description="Knows")])
    assert len(ents) == 2 and len(rels) == 1 and issues == 0


# --------------------------------------------- prompt / JSON-schema consistency
def test_json_schema_matches_the_pydantic_models():
    props = EXTRACTION_JSON_SCHEMA["properties"]
    ent = props["entities"]["items"]
    rel = props["relationships"]["items"]
    assert set(ent["properties"]) == set(Entity.model_fields) == set(ent["required"])
    assert set(rel["properties"]) == set(Relationship.model_fields) == set(rel["required"])
    assert ent["properties"]["type"]["enum"] == list(ENTITY_TYPES)
    assert set(EXTRACTION_JSON_SCHEMA["required"]) == set(RawExtraction.model_fields)
    assert EXTRACTION_JSON_SCHEMA["additionalProperties"] is False


def test_response_format_is_strict_structured_output():
    assert RESPONSE_FORMAT["type"] == "json_schema"
    assert RESPONSE_FORMAT["json_schema"]["strict"] is True


def test_prompt_lists_every_entity_type_and_messages_carry_only_the_chunk_text():
    for t in ENTITY_TYPES:
        assert t in SYSTEM_PROMPT
    msgs = build_messages("some chunk text")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "some chunk text" in msgs[1]["content"]
    assert PROMPT_VERSION


# ------------------------------------------------------------------- config
def test_extraction_defaults_are_safe():
    cfg = ExperimentConfig(experiment_id="t", seed=1, dataset="hotpotqa", num_questions=5,
                           strategy="random", budget=0.1)
    assert cfg.extraction_backend == "mock"          # no accidental spending
    assert cfg.extraction_model == "gpt-4o-mini"
    assert cfg.extraction_temperature == 0.0
    assert cfg.extraction_max_cost_usd is None
    assert cfg.extraction_cache_enabled is True


def test_new_example_configs_load():
    dev = load_config(PROJECT / "configs" / "extraction_dev.yaml")
    assert dev.extraction_backend == "mock" and dev.data_source == "mock"
    real = load_config(PROJECT / "configs" / "extraction_openai_example.yaml")
    assert real.extraction_backend == "openai" and real.extraction_max_cost_usd == 0.5


@pytest.mark.parametrize("bad", [
    {"extraction_backend": "anthropic"},
    {"extraction_temperature": 3.0},
    {"extraction_max_cost_usd": -1.0},
    {"extraction_max_cost_usd": 0.0},
    {"extraction_max_retries": -1},
    {"extraction_request_timeout_s": 0},
    {"extraction_price_input_per_1m": -0.1},
])
def test_bad_extraction_settings_rejected(bad):
    with pytest.raises(ValidationError):
        ExperimentConfig(experiment_id="t", seed=1, dataset="hotpotqa", num_questions=5,
                         strategy="random", budget=0.1, **bad)


def test_zero_retries_is_allowed():
    cfg = ExperimentConfig(experiment_id="t", seed=1, dataset="hotpotqa", num_questions=5,
                           strategy="random", budget=0.1, extraction_max_retries=0)
    assert cfg.extraction_max_retries == 0
