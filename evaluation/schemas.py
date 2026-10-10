"""
Data shapes for Evaluation (Phase 8).

THE JOIN PROBLEM THIS SOLVES
generation.schemas.AnswerResult knows the question TEXT but not which dataset question
it answers, and nothing in it says which strategy / budget / seed produced it. It must
stay that way: answers are cached by (question, context), so a question_id stored inside
a cached AnswerResult would be the first writer's id, not necessarily the current one.

So Evaluation wraps instead of changing it:

    PredictionRecord = AnswerResult (unchanged, nested as `answer`)
                     + question_id
                     + condition metadata (run_id, dataset, corpus_id, strategy, budget, seed)
                     + a retrieval summary and the retrieval / generation settings

PredictionRecord.build() is a pure constructor: it makes no calls and runs no loops (the
future experiment runner will call it once per question). It refuses to wrap an answer
whose question text differs from the question it is paired with, so a wrong join fails
loudly instead of scoring answers against the wrong gold.

GOLD DATA lives only on the evaluation side (GoldAnswer). A PredictionRecord never holds a
gold answer, and retrieval context is not stored in it at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from extraction.schemas import Usage
from generation.schemas import AnswerResult

EVAL_SCHEMA_VERSION = "1"

# A judge call: "ok", or the same failure vocabulary as extraction / generation.
JudgeCallStatus = Literal["ok", "api_error", "malformed", "refused", "truncated"]

# ScoredRecord.judge_status explains why a record has (or lacks) a judge verdict:
#   ok / api_error / malformed / refused / truncated : a judge call was made (JudgeCallStatus)
#   skipped_abstained / skipped_empty : a non-answer cannot be correct, so no call; verdict False
#   skipped_failed_generation         : the answer call failed, nothing to judge; verdict None
#   disabled / not_run                : judge switched off / run stopped before this record; verdict None
# ScoredRecord.outcome: answered | abstained ("not found") | empty (blank reply) | failed (the call failed)
Outcome = Literal["answered", "abstained", "empty", "failed"]


# --------------------------------------------------------------------------- gold
class GoldAnswer(BaseModel):
    """Reference answer(s) for one question. Evaluation-side only."""
    model_config = ConfigDict(frozen=True)

    question_id: str
    answer: str
    aliases: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def _answer_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("gold answer must not be blank")
        return v

    @property
    def all_answers(self) -> list[str]:
        """The gold answer first, then each non-blank alias, without duplicates."""
        return [a for a in dict.fromkeys([self.answer, *self.aliases]) if a.strip()]


# --------------------------------------------------------------------------- prediction
class RetrievalSummary(BaseModel):
    """What retrieval returned for one question - sizes and ids only, never the context text."""
    n_words: int = 0
    n_facts: int = 0
    n_passages: int = 0
    passage_chunk_ids: list[str] = Field(default_factory=list)
    n_seeds: int = 0
    empty_graph: bool = False
    extra: dict = Field(default_factory=dict)       # e.g. native L4: relevance_tests, n_relevant


def summarize_retrieval(retrieval=None, extra: dict | None = None, n_words: int = 0) -> RetrievalSummary | None:
    """Summary of a retrieval.RetrievalResult (read by attribute, so evaluation does not
    import the retrieval package). With no result but some `extra` (native L4), only that."""
    if retrieval is None:
        return RetrievalSummary(n_words=n_words, extra=dict(extra)) if extra else None
    return RetrievalSummary(
        n_words=int(retrieval.n_words), n_facts=len(retrieval.facts),
        n_passages=len(retrieval.chunk_ids), passage_chunk_ids=list(retrieval.chunk_ids),
        n_seeds=len(retrieval.seeds), empty_graph=bool(retrieval.empty_graph), extra=dict(extra or {}),
    )


def _retrieval_settings(cfg) -> dict:
    if cfg.strategy == "lazygraphrag_native":
        return {"method": "native_lazygraphrag", "relevance_budget": cfg.native_relevance_budget,
                "per_community": cfg.native_per_community, "max_relevant": cfg.native_max_relevant,
                "max_context_words": cfg.retrieval_max_context_words,
                "embedding_backend": cfg.embedding_backend, "embedding_model": cfg.embedding_model}
    return {"method": "personalised_pagerank", "k_seeds": cfg.retrieval_k_seeds,
            "damping": cfg.retrieval_damping, "top_m": cfg.retrieval_top_m,
            "max_context_words": cfg.retrieval_max_context_words, "max_facts": cfg.retrieval_max_facts,
            "embedding_backend": cfg.embedding_backend, "embedding_model": cfg.embedding_model}


class PredictionRecord(BaseModel):
    schema_version: str = EVAL_SCHEMA_VERSION
    question_id: str
    question: str

    # --- condition metadata (None where unknown; the runner supplies what it has) ---
    run_id: str | None = None
    experiment_id: str | None = None
    dataset: str | None = None
    data_source: str | None = None
    corpus_id: str | None = None
    strategy: str | None = None
    budget: float | None = None              # None for the native L4 reference point (no budget)
    seed: int | None = None

    retrieval: RetrievalSummary | None = None
    retrieval_settings: dict | None = None
    generation_settings: dict | None = None  # Answerer.settings(): model, prompt version, temperature, ...

    answer: AnswerResult                     # the existing result, unchanged

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_mock(self) -> bool:
        """True if a mock backend produced the answer. Mock output is never reportable."""
        return self.answer.backend == "mock"

    @classmethod
    def build(cls, question: dict, answer: AnswerResult, *, cfg=None, retrieval=None,
              retrieval_extra: dict | None = None, generation_settings: dict | None = None,
              run_id: str | None = None) -> "PredictionRecord":
        """Wrap `answer` for the dataset question `question` (a dict from src.corpus,
        with question_id / question). `cfg` is the ExperimentConfig of the run, if known."""
        if answer.question != question["question"]:
            raise ValueError(
                f"answer is for a different question than {question['question_id']!r}: "
                f"{answer.question!r} != {question['question']!r}"
            )
        meta: dict = {}
        retrieval_settings = None
        if cfg is not None:
            from src.prepare_data import corpus_id
            native = cfg.strategy == "lazygraphrag_native"
            meta = dict(experiment_id=cfg.experiment_id, dataset=cfg.dataset, data_source=cfg.data_source,
                        corpus_id=corpus_id(cfg), strategy=cfg.strategy,
                        budget=None if native else cfg.budget, seed=cfg.seed)
            retrieval_settings = _retrieval_settings(cfg)
            if run_id is None:
                from extraction.run import make_run_id
                run_id = make_run_id(cfg)
        return cls(
            question_id=str(question["question_id"]), question=question["question"], run_id=run_id,
            retrieval=summarize_retrieval(retrieval, retrieval_extra, n_words=answer.context_words),
            retrieval_settings=retrieval_settings, generation_settings=generation_settings,
            answer=answer, **meta,
        )

    def condition(self) -> dict:
        return {"run_id": self.run_id, "experiment_id": self.experiment_id, "dataset": self.dataset,
                "data_source": self.data_source, "corpus_id": self.corpus_id, "strategy": self.strategy,
                "budget": self.budget, "seed": self.seed}


# --------------------------------------------------------------------------- judge
class JudgeOutput(BaseModel):
    """What the judge model must return."""
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    correct: bool


class JudgeResult(BaseModel):
    """One judge call (same bookkeeping as generation's AnswerResult)."""
    status: JudgeCallStatus
    error: str | None = None
    backend: str                                   # "mock" or "openai"
    model: str
    prompt_version: str
    prompt_sha256: str                             # hash of the exact messages sent
    usage: Usage = Field(default_factory=Usage)    # every attempt; None fields = unknown
    attempts: int = 1
    runtime_seconds: float = 0.0
    cache_hit: bool = False
    correct: bool | None = None                    # None when the call failed: NOT "incorrect"
    reasoning: str = ""


# --------------------------------------------------------------------------- scored
class ScoredRecord(BaseModel):
    """One scored prediction: flat, so a runner can group it by condition directly."""
    schema_version: str = EVAL_SCHEMA_VERSION
    question_id: str
    run_id: str | None = None
    experiment_id: str | None = None
    dataset: str | None = None
    corpus_id: str | None = None
    strategy: str | None = None
    budget: float | None = None
    seed: int | None = None

    gold_answer: str
    gold_aliases: list[str] = Field(default_factory=list)
    predicted_answer: str
    generation_status: str                         # AnswerResult.status, preserved as is
    generation_error: str | None = None

    # answered | abstained ("not found") | empty (blank reply) | failed (the call failed)
    outcome: Outcome
    em: float | None = None                        # None only for failed generations
    f1: float | None = None
    judge_correct: bool | None = None              # None = no verdict (see judge_status)
    judge_status: str
    judge: JudgeResult | None = None               # the call, when one was made

    is_mock_generation: bool
    is_mock_judge: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_mock(self) -> bool:
        return self.is_mock_generation or self.is_mock_judge


# --------------------------------------------------------------------------- JSONL I/O
def write_jsonl(path: str | Path, models: Iterable[BaseModel]) -> None:
    """Atomic write (temp file, then rename): a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for m in models:
            f.write(json.dumps(m.model_dump(mode="json"), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_predictions(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    write_jsonl(path, records)


def read_predictions(path: str | Path) -> list[PredictionRecord]:
    with open(path, encoding="utf-8") as f:
        return [PredictionRecord.model_validate_json(line) for line in f if line.strip()]
