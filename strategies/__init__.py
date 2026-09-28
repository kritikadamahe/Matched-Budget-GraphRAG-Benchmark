"""
Strategy registry. The experiment runner (Phase 13) builds strategies through
build_strategy() so that adding a strategy never touches the pipeline code.

Note: the native LazyGraphRAG reference point (option L4) is NOT a
SelectionStrategy - it has no pre-extraction budget. See native/lazygraphrag_native.py.
"""

from strategies.fastgraphrag_strategy import FastGraphRAGStrategy
from strategies.ketrag_strategy import KETRAGStrategy
from strategies.lazygraphrag_strategy import LazyGraphRAGStrategy
from strategies.random_strategy import RandomStrategy

STRATEGIES = {
    "random": RandomStrategy,
    "ketrag": KETRAGStrategy,
    "lazygraphrag": LazyGraphRAGStrategy,
    "fastgraphrag": FastGraphRAGStrategy,
}


def build_strategy(cfg):
    """Create the strategy named in an ExperimentConfig, with its own knobs."""
    if cfg.strategy == "random":
        return RandomStrategy(seed=cfg.seed)
    if cfg.strategy == "ketrag":
        return KETRAGStrategy(knn_k=cfg.ketrag_knn_k, seed=cfg.seed)
    if cfg.strategy == "lazygraphrag":
        return LazyGraphRAGStrategy(n_clusters=cfg.lazy_n_clusters, seed=cfg.seed)
    if cfg.strategy == "fastgraphrag":
        return FastGraphRAGStrategy(use_spacy=cfg.fast_use_spacy)
    raise ValueError(f"{cfg.strategy!r} is not a budgeted selection strategy")
