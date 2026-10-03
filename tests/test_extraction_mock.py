"""Phase 4: the deterministic MockExtractor - the whole pipeline must run on it with no key."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from extraction.mock_extractor import MOCK_VERSION, MockExtractor
from extraction.schemas import ENTITY_TYPES, ExtractionInput, ExtractionResult
from src.text_utils import normalize_name

pytestmark = pytest.mark.usefixtures("no_network", "no_api_key")

TEXT = ("Forrest Gump: Forrest Gump is a 1994 film starring Tom Hanks. "
        "Robert Zemeckis directed it. It won the Academy Award for Best Picture.")


def _extract(text=TEXT, cid="c1"):
    return MockExtractor().extract(ExtractionInput(chunk_id=cid, text=text))


def _stable(result):
    return result.model_dump(exclude={"runtime_seconds"})


def test_mock_needs_no_api_key_and_returns_a_valid_ok_result():
    r = _extract()
    assert isinstance(r, ExtractionResult)
    assert r.status == "ok" and r.error is None
    assert r.chunk_id == "c1"


def test_mock_is_deterministic():
    assert _stable(_extract()) == _stable(_extract())


def test_different_text_gives_different_output():
    assert _stable(_extract(TEXT)) != _stable(_extract("Winston Groom wrote the novel Forrest Gump."))


def test_mock_finds_the_obvious_entities():
    names = {e.name for e in _extract().entities}
    assert {"Tom Hanks", "Robert Zemeckis"} <= names


def test_mock_types_come_from_the_fixed_list_and_are_stable():
    r = _extract()
    assert r.entities and all(e.type in ENTITY_TYPES for e in r.entities)
    assert [e.type for e in r.entities] == [e.type for e in _extract().entities]


def test_mock_relationships_only_reference_listed_entities():
    r = _extract()
    names = {normalize_name(e.name) for e in r.entities}
    assert r.relationships and r.validation_issues == 0
    assert all(normalize_name(x.source) in names and normalize_name(x.target) in names
               for x in r.relationships)


def test_text_without_entities_is_still_a_valid_ok_result():
    r = _extract("nothing to see here at all.")
    assert r.status == "ok" and r.entities == [] and r.relationships == []


def test_mock_is_labelled_mock_costs_nothing_and_reports_fake_tokens():
    r = _extract()
    assert r.extractor == "mock" and r.model == MOCK_VERSION
    assert r.usage.cost_usd == 0.0
    assert r.usage.input_tokens > 0 and r.usage.output_tokens > 0
    assert r.attempts == 1 and r.cache_hit is False


def test_mock_cache_settings_name_the_mock_and_its_version():
    s = MockExtractor().cache_settings()
    assert s["extractor"] == "mock" and s["model"] == MOCK_VERSION
