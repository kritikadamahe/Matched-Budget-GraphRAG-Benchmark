"""
Random baseline (blueprint §H).

No logic - a fixed-seed shuffle of all chunk_ids. This is the strategy that
needs multiple seeds at evaluation time (blueprint §O), because its
quality at a given budget depends entirely on which chunks its particular
shuffle happened to include.
"""

from __future__ import annotations
import random

from src.chunking import Chunk
from strategies.base_strategy import SelectionStrategy


class RandomStrategy(SelectionStrategy):
    name = "random"

    def __init__(self, seed: int):
        self.seed = seed

    def rank(self, chunks: list[Chunk]) -> list[str]:
        chunk_ids = [c.chunk_id for c in chunks]
        rng = random.Random(self.seed)
        rng.shuffle(chunk_ids)
        return chunk_ids
