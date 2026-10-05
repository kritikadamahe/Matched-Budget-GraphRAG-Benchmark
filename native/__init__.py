"""Native LazyGraphRAG reference point (option L4) - not a budgeted strategy."""

from native.lazygraphrag_native import NativeLazyGraphRAG


def build_native(cfg) -> NativeLazyGraphRAG:
    """Create the L4 system from an ExperimentConfig, with the ONE shared embedder."""
    from src.embeddings import embedder_from_config
    return NativeLazyGraphRAG(
        relevance_budget=cfg.native_relevance_budget,
        per_community=cfg.native_per_community,
        max_relevant=cfg.native_max_relevant,
        use_spacy=cfg.fast_use_spacy,
        seed=cfg.seed,
        embedder=embedder_from_config(cfg),
    )
