"""
Phase 4: LLM extraction layer.

    corpus -> chunks -> strategy -> budget -> budget-selected chunks
           -> [this package] LLM entity/relationship extraction -> cache + results
           -> (later phases) knowledge graph construction

This package only extracts. It builds no graph, does no retrieval or answer
generation, and is not the full benchmark runner.
"""

from extraction.base_extractor import ExtractionAbort, Extractor
from extraction.mock_extractor import MockExtractor
from extraction.openai_extractor import DisabledClient, MissingAPIKeyError, OpenAIExtractor


def build_extractor(cfg, dry_run: bool = False) -> Extractor:
    """Create the extractor named in an ExperimentConfig (extraction_backend).

    The backend is always the one the config says. "openai" NEVER turns into the
    mock: a missing API key raises MissingAPIKeyError. With dry_run=True the real
    extractor is built with a client that refuses to send anything, so cost
    estimates work without a key and without any possibility of an API call.
    """
    if cfg.extraction_backend == "mock":
        return MockExtractor()
    if cfg.extraction_backend == "openai":
        return OpenAIExtractor(
            model=cfg.extraction_model,
            temperature=cfg.extraction_temperature,
            max_output_tokens=cfg.extraction_max_output_tokens,
            max_retries=cfg.extraction_max_retries,
            timeout_s=cfg.extraction_request_timeout_s,
            price_input_per_1m=cfg.extraction_price_input_per_1m,
            price_output_per_1m=cfg.extraction_price_output_per_1m,
            client=DisabledClient() if dry_run else None,
        )
    raise ValueError(f"unknown extraction_backend: {cfg.extraction_backend!r}")
