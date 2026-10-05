"""Blueprint Phase 4: cached embedding pipeline. Uses a fake encoder, so no torch
or model download is needed; one test runs the real model when it is installed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3

import numpy as np
import pytest

from native import build_native
from src.config import ExperimentConfig
from src.embed_corpus import embed_corpus
from src.embeddings import CachedEmbedder, TfidfEmbedder, embedder_from_config, make_embedder
from strategies import build_strategy


class FakeEncoder:
    """Deterministic stand-in for a sentence-transformers model; counts encode calls."""
    def __init__(self):
        self.calls, self.texts = 0, []

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        self.calls += 1
        self.texts.extend(texts)
        return np.array([[len(t), sum(map(ord, t)) % 97 + 1, 1.0] for t in texts], dtype=np.float32)


def make(tmp_path, encoder=None, model="fake-model", **kw):
    encoder = encoder or FakeEncoder()
    return CachedEmbedder(model, cache_dir=tmp_path, encoder_factory=lambda name: encoder, **kw), encoder


def test_second_run_is_all_cache_hits_and_never_loads_the_model(tmp_path):
    texts = ["Forrest Gump: a 1994 film", "Robert Zemeckis: director", "Tuscaloosa: a city"]
    first, enc1 = make(tmp_path)
    v1 = first.embed(texts)
    assert first.stats == {"hits": 0, "misses": 3, "model_loads": 1} and enc1.calls == 1

    second, enc2 = make(tmp_path)                       # fresh process, same cache dir
    v2 = second.embed(texts)
    assert second.stats == {"hits": 3, "misses": 0, "model_loads": 0} and enc2.calls == 0   # Phase 4 checkpoint
    np.testing.assert_array_equal(v1, v2)


def test_vectors_are_float32_unit_length_and_keep_input_order(tmp_path):
    emb, _ = make(tmp_path)
    emb.embed(["b"])                                    # "b" cached, "a" not: order must still hold
    out = emb.embed(["a", "b", "a"])
    assert out.dtype == np.float32 and out.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(out[0], out[2])


def test_duplicates_are_encoded_once(tmp_path):
    emb, enc = make(tmp_path)
    emb.embed(["same text", "same text", "other"])
    assert enc.texts == ["same text", "other"] and emb.stats["misses"] == 2


def test_model_name_is_part_of_the_key(tmp_path):
    a, _ = make(tmp_path, model="model-a")
    a.embed(["x"])
    b, enc_b = make(tmp_path, model="model-b")
    b.embed(["x"])
    assert enc_b.calls == 1                             # model-a's vector is never served for model-b
    assert {p.name for p in tmp_path.iterdir()} == {"model-a.sqlite", "model-b.sqlite"}


def test_cache_disabled_always_encodes(tmp_path):
    emb, enc = make(tmp_path, cache_enabled=False)
    emb.embed(["x"]); emb.embed(["x"])
    assert enc.calls == 2 and not list(tmp_path.iterdir())


def test_corrupt_row_is_treated_as_a_miss(tmp_path):
    emb, _ = make(tmp_path)
    emb.embed(["x"])
    with sqlite3.connect(emb.cache.path) as con:
        con.execute("UPDATE vectors SET dim = 999")
    again, enc = make(tmp_path)
    again.embed(["x"])
    assert enc.calls == 1


def test_fit_transform_and_transform_agree(tmp_path):
    emb, _ = make(tmp_path)
    np.testing.assert_array_equal(emb.fit_transform(["q"]), emb.transform(["q"]))


# ------------------------------------------------------------- config wiring
def _cfg(**kw):
    base = dict(experiment_id="t", seed=0, dataset="hotpotqa", num_questions=5, strategy="lazygraphrag",
                budget=0.1, fast_use_spacy=False)
    return ExperimentConfig(**{**base, **kw})


def test_backend_selection_and_cache_location(tmp_path):
    assert isinstance(make_embedder("tfidf"), TfidfEmbedder)
    emb = embedder_from_config(_cfg(cache_dir=str(tmp_path)))
    assert isinstance(emb, CachedEmbedder) and emb.name == "all-MiniLM-L6-v2"
    assert emb.cache.path == tmp_path / "embeddings" / "all-MiniLM-L6-v2.sqlite"


def test_strategies_and_native_share_the_configured_embedder(tmp_path):
    cfg = _cfg(cache_dir=str(tmp_path))
    assert isinstance(build_strategy(cfg).embedder, CachedEmbedder)
    assert isinstance(build_native(cfg).embedder, CachedEmbedder)
    assert isinstance(build_strategy(_cfg(embedding_backend="tfidf")).embedder, TfidfEmbedder)


def test_faithful_ketrag_rejects_tfidf_backend():
    with pytest.raises(Exception, match="faithful needs embedding_backend"):
        _cfg(strategy="ketrag", ketrag_mode="faithful", embedding_backend="tfidf")
    _cfg(strategy="ketrag", ketrag_mode="tfidf", embedding_backend="tfidf")           # allowed


def test_embed_corpus_checkpoint_with_fake_model(tmp_path):
    cfg = _cfg(cache_dir=str(tmp_path), data_source="mock")
    first = embed_corpus(cfg, make(tmp_path / "embeddings", model="all-MiniLM-L6-v2")[0])
    assert first["misses"] > 0 and first["model_loads"] == 1 and first["_shapes_ok"]
    second = embed_corpus(cfg, make(tmp_path / "embeddings", model="all-MiniLM-L6-v2")[0])
    assert second["checkpoint_fully_cached"] and second["hits"] == first["misses"]


def test_embed_corpus_refuses_tfidf():
    with pytest.raises(ValueError, match="not cacheable"):
        embed_corpus(_cfg(embedding_backend="tfidf", data_source="mock"))


def test_real_model_round_trip(tmp_path):
    pytest.importorskip("sentence_transformers")
    first = CachedEmbedder(cache_dir=tmp_path)
    v1 = first.embed(["Robert Zemeckis directed Forrest Gump", "Tuscaloosa is in Alabama"])
    second = CachedEmbedder(cache_dir=tmp_path)
    v2 = second.embed(["Robert Zemeckis directed Forrest Gump", "Tuscaloosa is in Alabama"])
    assert v1.shape == (2, 384) and second.stats["model_loads"] == 0
    np.testing.assert_array_equal(v1, v2)
