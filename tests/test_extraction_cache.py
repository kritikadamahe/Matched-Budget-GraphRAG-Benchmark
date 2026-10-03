"""Phase 4: the extraction cache - deterministic keys that include model/prompt/settings."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pytest

from extraction.cache import ExtractionCache
from extraction.mock_extractor import MockExtractor
from extraction.schemas import Entity, ExtractionInput, ExtractionResult, Usage

pytestmark = pytest.mark.usefixtures("no_network", "no_api_key")

SETTINGS = {"extractor": "openai", "model": "gpt-4o-mini", "prompt_version": "extract-v1",
            "schema_version": "1", "temperature": 0.0, "max_output_tokens": 1500, "seed": 1234}
ITEM = ExtractionInput(chunk_id="chunk-A", text="Tom Hanks starred in Forrest Gump.")


def _result(status="ok", chunk_id="chunk-A"):
    return ExtractionResult(
        chunk_id=chunk_id, status=status, entities=[Entity(name="Tom Hanks", type="PERSON")],
        extractor="openai", model="gpt-4o-mini", prompt_version="extract-v1", attempts=2)


@pytest.fixture
def cache(tmp_path):
    return ExtractionCache(tmp_path / "cache")


def test_put_then_get_returns_the_result_marked_as_cache_hit(cache):
    assert cache.put(SETTINGS, ITEM, _result()) is True
    hit = cache.get(SETTINGS, ITEM)
    assert hit is not None and hit.cache_hit is True
    assert hit.entities == _result().entities and hit.attempts == 2


def test_get_before_put_is_a_miss(cache):
    assert cache.get(SETTINGS, ITEM) is None


def test_hit_is_restamped_with_the_requesting_chunk_id(cache):
    # Content-addressed: the same text under another chunk_id reuses the entry.
    cache.put(SETTINGS, ITEM, _result(chunk_id="chunk-A"))
    other = ExtractionInput(chunk_id="chunk-B", text=ITEM.text)
    assert cache.get(SETTINGS, other).chunk_id == "chunk-B"


@pytest.mark.parametrize("field,new_value", [
    ("model", "gpt-4o"),
    ("prompt_version", "extract-v2"),
    ("schema_version", "2"),
    ("temperature", 0.7),
    ("max_output_tokens", 800),
    ("seed", 99),
    ("extractor", "mock"),
])
def test_changing_any_setting_is_a_cache_miss(cache, field, new_value):
    cache.put(SETTINGS, ITEM, _result())
    changed = {**SETTINGS, field: new_value}
    assert cache.get(changed, ITEM) is None


def test_changing_the_chunk_text_is_a_cache_miss(cache):
    cache.put(SETTINGS, ITEM, _result())
    assert cache.get(SETTINGS, ExtractionInput(chunk_id="chunk-A", text=ITEM.text + " ")) is None


def test_key_is_deterministic_and_ignores_dict_order():
    k1 = ExtractionCache.make_key(SETTINGS, ITEM.text)
    k2 = ExtractionCache.make_key(dict(reversed(list(SETTINGS.items()))), ITEM.text)
    assert k1 == k2 and len(k1) == 64 and int(k1, 16) >= 0
    assert ExtractionCache(Path("/x")).make_key(SETTINGS, ITEM.text) == k1


@pytest.mark.parametrize("status", ["api_error", "malformed", "refused", "truncated"])
def test_failed_results_are_never_cached(cache, status):
    assert cache.put(SETTINGS, ITEM, _result(status)) is False
    assert cache.get(SETTINGS, ITEM) is None
    assert not list(cache.root.rglob("*.json")) if cache.root.exists() else True


def test_corrupt_cache_file_counts_as_a_miss(cache):
    cache.put(SETTINGS, ITEM, _result())
    path = cache.path_for(SETTINGS, ITEM.text)
    path.write_text("{ this is not json", encoding="utf-8")
    assert cache.get(SETTINGS, ITEM) is None


def test_cache_file_for_a_different_key_is_not_trusted(cache):
    cache.put(SETTINGS, ITEM, _result())
    path = cache.path_for(SETTINGS, ITEM.text)
    data = json.loads(path.read_text())
    data["key"] = "0" * 64
    path.write_text(json.dumps(data))
    assert cache.get(SETTINGS, ITEM) is None


def test_disabled_cache_never_hits_and_writes_nothing(tmp_path):
    cache = ExtractionCache(tmp_path / "cache", enabled=False)
    assert cache.put(SETTINGS, ITEM, _result()) is False
    assert cache.get(SETTINGS, ITEM) is None
    assert not (tmp_path / "cache").exists()


def test_writes_are_atomic_no_temp_files_left_behind(cache):
    cache.put(SETTINGS, ITEM, _result())
    assert not list(cache.root.rglob("*.tmp"))
    assert len(list(cache.root.rglob("*.json"))) == 1


def test_mock_and_openai_entries_live_in_separate_folders(cache):
    mock_settings = MockExtractor().cache_settings()
    cache.put(SETTINGS, ITEM, _result())
    cache.put(mock_settings, ITEM, _result())
    assert cache.path_for(SETTINGS, ITEM.text).parent != cache.path_for(mock_settings, ITEM.text).parent
    assert (cache.root / "openai").is_dir() and (cache.root / "mock").is_dir()


def test_chunk_text_is_not_stored_only_its_hash(cache):
    cache.put(SETTINGS, ITEM, _result())
    raw = cache.path_for(SETTINGS, ITEM.text).read_text()
    assert "Forrest Gump" not in raw and "text_sha256" in raw


# ---------------------------------------------------- unknown usage is never cached
def _ok_but_usage_unknown():
    return _result().model_copy(update={"usage": Usage.unknown()})


def test_an_ok_result_whose_usage_is_unknown_is_not_cached(cache):
    # Its content is fine, but caching it would make this chunk's cost unknown in every future run.
    assert cache.put(SETTINGS, ITEM, _ok_but_usage_unknown()) is False
    assert cache.get(SETTINGS, ITEM) is None
    assert not (cache.root.exists() and list(cache.root.rglob("*.json")))


def test_an_entry_with_unknown_usage_found_on_disk_is_never_served(cache):
    cache.put(SETTINGS, ITEM, _result())                        # a normal, known entry...
    path = cache.path_for(SETTINGS, ITEM.text)
    data = json.loads(path.read_text())
    data["result"]["usage"] = {"input_tokens": None, "output_tokens": None, "cost_usd": None}
    path.write_text(json.dumps(data))                           # ...tampered/legacy: usage now unknown
    assert cache.get(SETTINGS, ITEM) is None


def test_known_usage_including_a_real_zero_is_still_cached(cache):
    zero_cost = _result().model_copy(update={"usage": Usage(input_tokens=10, output_tokens=0, cost_usd=0.0)})
    assert cache.put(SETTINGS, ITEM, zero_cost) is True
    assert cache.get(SETTINGS, ITEM).usage.known is True
