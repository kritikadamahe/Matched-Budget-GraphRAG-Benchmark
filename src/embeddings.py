"""
Chunk/question vectors for strategies that need text similarity.

Default: TF-IDF (scikit-learn, already a dependency for KET-RAG) - free, offline,
deterministic, no model download. Rows are L2-normalised, so a dot product is a
cosine similarity.

Optional: sentence-transformers (pip install sentence-transformers) for
semantic embeddings, e.g. model="all-MiniLM-L6-v2". Still $0 (runs locally).
Whichever is used must be the SAME for every condition (blueprint §N).
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


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


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.name = model
        self._model = SentenceTransformer(model)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=64, show_progress_bar=False)
        return normalize(np.asarray(vecs, dtype=np.float32))


def make_embedder(backend: str = "tfidf", model: str = "all-MiniLM-L6-v2"):
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model)
    raise ValueError(f"Unknown embedding backend: {backend!r}")
