"""
Embeddings (blueprint Phase 4, §N, §P).

ONE shared embedding model is used by every component that needs text vectors:
KET-RAG (semantic neighbours), LazyGraphRAG L3 (topic clustering), the native
LazyGraphRAG L4 (question-chunk relevance) and, later, retrieval (§J). Using the
same model everywhere keeps it a constant across all conditions (§N).

BACKENDS
- "sentence-transformers" (default): local model all-MiniLM-L6-v2, free, runs on
  CPU. Wrapped in CachedEmbedder, so every text is embedded at most once, ever.
- "tfidf": TF-IDF vectors (scikit-learn). Free and instant, used by the tests and
  for ablations. NOT cached: TF-IDF vectors depend on the whole corpus (the
  vocabulary and IDF weights), so a chunk has no fixed vector of its own.

THE CACHE (CachedEmbedder)
- One SQLite file per model: cache/embeddings/<model>.sqlite (git-ignored).
- Key = SHA-256 of (model name + text). Content-addressed, like the extraction
  cache: identical text is never embedded twice, whichever corpus, strategy or
  question it comes from. chunk_id is not in the key (the blueprint suggested
  chunk_id + model; the text hash is stricter - if a chunk's text ever changed,
  its old vector could not be served by mistake).
- The model is loaded lazily, only if some text is missing from the cache. A fully
  cached run never loads it: that is the Phase 4 checkpoint ("cache hit on the
  2nd run, 0 model/API calls").
- Writes are SQLite transactions, so a crash never leaves a half-written vector.

All backends return float32, L2-normalised rows, so a dot product is a cosine.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"


def _normalise(vectors) -> np.ndarray:
    return normalize(np.asarray(vectors, dtype=np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# TF-IDF (corpus-fitted, not cacheable)
# ---------------------------------------------------------------------------
class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self._vectorizer: TfidfVectorizer | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit the vocabulary on the corpus and return dense, L2-normalised vectors."""
        self._vectorizer = TfidfVectorizer(stop_words="english")
        return normalize(self._vectorizer.fit_transform(texts)).toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        """Embed new texts (e.g. questions) in the already-fitted vocabulary."""
        if self._vectorizer is None:
            raise RuntimeError("call fit_transform on the corpus first")
        return normalize(self._vectorizer.transform(texts)).toarray()


# ---------------------------------------------------------------------------
# Sentence-transformers, uncached (kept for direct use; prefer CachedEmbedder)
# ---------------------------------------------------------------------------
def load_sentence_transformer(model: str):
    """Load a local sentence-transformers model (optional dependency)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "The semantic embedding model needs: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu && pip install sentence-transformers. "
            "Or set embedding_backend: tfidf (and ketrag_mode: tfidf)."
        ) from e
    return SentenceTransformer(model)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.name = model
        self._model = load_sentence_transformer(model)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=64, show_progress_bar=False)
        return _normalise(vecs)


# ---------------------------------------------------------------------------
# Cached semantic embeddings
# ---------------------------------------------------------------------------
class EmbeddingCache:
    """SQLite store: key -> float32 vector. One file per model."""

    def __init__(self, root: str | Path, model: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.sqlite"
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self._connect() as con:
            for start in range(0, len(keys), 500):          # SQLite parameter limit
                batch = keys[start:start + 500]
                marks = ",".join("?" * len(batch))
                for key, dim, blob in con.execute(
                        f"SELECT key, dim, vec FROM vectors WHERE key IN ({marks})", batch):
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if vec.shape == (dim,):                  # corrupt row -> treated as a miss
                        found[key] = vec
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self._connect() as con:                          # one transaction: all or nothing
            con.executemany(
                "INSERT OR REPLACE INTO vectors (key, dim, vec) VALUES (?, ?, ?)",
                [(k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
            )

    def __len__(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]


class CachedEmbedder:
    """Shared semantic embedder with a persistent cache. Same interface as the
    others: fit_transform(texts) and transform(texts) both return embeddings
    (a neural model needs no fitting)."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | Path | None = None,
                 cache_enabled: bool = True, encoder_factory=None, batch_size: int = 64):
        self.name = model
        self.cache = EmbeddingCache(cache_dir or DEFAULT_CACHE_DIR, model) if cache_enabled else None
        self._encoder_factory = encoder_factory or load_sentence_transformer   # injectable for tests
        self._encoder = None
        self.batch_size = batch_size
        self.stats = {"hits": 0, "misses": 0, "model_loads": 0}

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.name}\x1f{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [self.key(t) for t in texts]
        found = self.cache.get_many(list(dict.fromkeys(keys))) if self.cache is not None else {}

        missing: dict[str, str] = {}                         # key -> text, unique, in first-seen order
        for k, t in zip(keys, texts):
            if k not in found:
                missing.setdefault(k, t)
        self.stats["hits"] += sum(1 for k in keys if k in found)
        self.stats["misses"] += len(missing)

        if missing:
            if self._encoder is None:
                self._encoder = self._encoder_factory(self.name)
                self.stats["model_loads"] += 1
            vecs = _normalise(self._encoder.encode(list(missing.values()), batch_size=self.batch_size,
                                                   show_progress_bar=False))
            new = dict(zip(missing.keys(), vecs))
            if self.cache is not None:          # not `if self.cache`: an EMPTY cache has len 0
                self.cache.put_many(new)
            found.update(new)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)


# ---------------------------------------------------------------------------
def make_embedder(backend: str = "sentence-transformers", model: str = DEFAULT_MODEL,
                  cache_dir: str | Path | None = None, cache_enabled: bool = True):
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "sentence-transformers":
        return CachedEmbedder(model, cache_dir=cache_dir, cache_enabled=cache_enabled)
    raise ValueError(f"Unknown embedding backend: {backend!r}")


def embedder_from_config(cfg):
    """The ONE shared embedder for a run, built from the ExperimentConfig."""
    cache_dir = Path(cfg.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    return make_embedder(cfg.embedding_backend, cfg.embedding_model,
                         cache_dir=cache_dir / "embeddings", cache_enabled=cfg.embedding_cache_enabled)
