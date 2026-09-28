"""
Config system.

WHY THIS EXISTS:
Every one of our 48+ experiment runs must be traceable to an exact set of
settings (dataset, budget, seed, chunk size, ...). If those settings live
as scattered hardcoded numbers in scripts, we can't reproduce a result or
prove to an examiner that two runs were actually comparable.

Instead: every run is described by ONE YAML file. This module loads that
YAML into a validated Python object (ExperimentConfig). "Validated" means
pydantic checks the types and ranges immediately and fails loudly with a
clear error, instead of failing silently 20 minutes into a run.
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    # --- identity ---
    experiment_id: str = Field(..., description="Unique name for this run, used in file names and logs.")
    seed: int = Field(..., description="Random seed. Fixed per run for reproducibility.")

    # --- dataset / corpus (blueprint §E) ---
    dataset: Literal["hotpotqa", "musique"] = Field(..., description="Which dataset this run uses.")
    data_source: Literal["huggingface", "mock"] = Field(
        "mock",
        description=(
            "'huggingface' pulls the real dataset (needs internet access to "
            "huggingface.co - not available in this sandbox). 'mock' uses the "
            "small bundled fixture with the same schema, for pipeline testing."
        ),
    )
    num_questions: int = Field(
        ..., gt=0,
        description="How many questions' contexts get pooled into the shared corpus (blueprint §E).",
    )

    # --- chunking (blueprint §F) ---
    chunk_size_words: int = Field(250, gt=0, description="Approx. words per chunk.")
    chunk_overlap_words: int = Field(40, ge=0, description="Words of overlap between consecutive chunks.")

    # --- strategy / budget (blueprint §G, §H) ---
    strategy: Literal["random", "ketrag", "lazygraphrag", "fastgraphrag", "lazygraphrag_native"] = Field(
        ...,
        description=(
            "Chunk-selection strategy. 'lazygraphrag_native' is the L4 reference point: "
            "it has no pre-extraction budget, so `budget` is ignored for it."
        ),
    )
    budget: float = Field(..., gt=0, le=1.0, description="Fraction of chunks to select, e.g. 0.10 for 10%.")

    # --- strategy-specific knobs ---
    ketrag_knn_k: int = Field(10, gt=0, description="KET-RAG: neighbors per chunk in the similarity graph.")
    lazy_n_clusters: int | None = Field(
        None, gt=0, description="LazyGraphRAG (L3): number of topics; None = sqrt(N/2)."
    )
    fast_use_spacy: bool = Field(
        True, description="FastGraphRAG (F1): use spaCy NER if installed, else a capitalised-phrase rule."
    )
    native_relevance_budget: int = Field(
        20, gt=0, description="Native LazyGraphRAG (L4): max LLM relevance checks per question."
    )
    native_per_community: int = Field(3, gt=0, description="Native LazyGraphRAG (L4): checks per community.")
    native_max_relevant: int = Field(5, gt=0, description="Native LazyGraphRAG (L4): stop after this many relevant chunks.")

    # --- paths (defaults are relative to project root) ---
    cache_dir: str = "cache"
    output_dir: str = "results"

    @field_validator("chunk_overlap_words")
    @classmethod
    def overlap_smaller_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size_words")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError(
                f"chunk_overlap_words ({v}) must be smaller than chunk_size_words ({chunk_size})."
            )
        return v


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML config file into an ExperimentConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)
