"""
Extractor interface (Phase 4).

WHY THIS EXISTS:
The pipeline (extraction/pipeline.py) talks to every extractor through this one
contract, exactly like strategies/base_strategy.py does for selection
strategies. That is what lets us test the whole pipeline with the free
MockExtractor and later swap in the real OpenAI one without touching anything
else - and what guarantees that every strategy is extracted by the SAME
extractor, model and prompt (blueprint section N: the strategy is the only
variable).

An extractor's job is small: given ONE chunk's text, return an ExtractionResult.
It never decides WHICH chunks to extract - the pipeline decides that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from extraction.schemas import ExtractionInput, ExtractionResult


class ExtractionAbort(RuntimeError):
    """The whole run must stop (bad API key, no quota, unknown model, cost cap...).

    Used for problems that would hit EVERY chunk identically - continuing would
    only waste time and money. Per-chunk problems (a malformed answer) are NOT
    aborts; they become a failed ExtractionResult and the run carries on.
    """


class Extractor(ABC):
    name: str    # "mock" or "openai" - stored in every result and in the cache path
    model: str   # model identifier - part of the cache key

    @abstractmethod
    def extract(self, item: ExtractionInput) -> ExtractionResult:
        """Extract entities and relationships from one chunk.

        Must return a result with item.chunk_id. Per-chunk failures are returned
        as results with a non-"ok" status; only run-level problems raise
        ExtractionAbort.
        """
        raise NotImplementedError

    @abstractmethod
    def cache_settings(self) -> dict:
        """Everything that can change the output for the same text: extractor,
        model, prompt version, schema version, temperature, max output tokens...
        The cache key is a hash of this plus the chunk text, so changing ANY of
        these automatically invalidates old cache entries."""
        raise NotImplementedError

    def estimate_cost_usd(self, item: ExtractionInput) -> float:
        """Rough cost of extracting this chunk, used by --dry-run and the spending
        cap BEFORE any call is made. Free extractors return 0."""
        return 0.0
