"""Test doubles for the experiment-runner tests. Spies WRAP the real components (so the real pipeline
runs and is observed); failure doubles subclass the free MockExtractor / use the fake OpenAI client.
Nothing here can make a real API call."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.judge import Judge
from experiments.runner import Components, RunnerContext
from extraction.base_extractor import ExtractionAbort
from extraction.mock_extractor import MockExtractor
from extraction.schemas import ExtractionResult, Usage
from generation.answerer import Answerer
from generation.llm_client import ChatJSONClient
from src.config import ExperimentConfig
from tests.helpers_extraction import FakeOpenAIClient, make_response


def base_cfg(tmp_path, **kw) -> ExperimentConfig:
    """A free, mock-backed, TF-IDF config (3 questions: the MuSiQue mock fixture has only 3)."""
    d = dict(experiment_id="t", seed=42, dataset="hotpotqa", data_source="mock", num_questions=3,
             strategy="random", budget=0.1, embedding_backend="tfidf", ketrag_mode="tfidf", fast_use_spacy=False,
             cache_dir=str(Path(tmp_path) / "cache"), output_dir=str(Path(tmp_path) / "out"))
    d.update(kw)
    return ExperimentConfig(**d)


def make_ctx(tmp_path, comps=None, cfg=None, no_judge=False, **cfg_kw) -> RunnerContext:
    cfg = cfg or base_cfg(tmp_path, **cfg_kw)
    return RunnerContext(base_cfg=cfg, matrix_id=cfg.experiment_id, matrix_dir=Path(tmp_path) / "matrix",
                         comps=comps or Components(), no_judge=no_judge)


# ------------------------------------------------------------------------------- spies
def labels(chunks):
    return tuple(tuple(c.is_gold_for_question_ids) for c in chunks)


class SpyStrategy:
    def __init__(self, inner, events):
        self.inner, self.events, self.name = inner, events, inner.name

    def rank(self, chunks):
        self.events.append(("rank", self.name, labels(chunks)))
        return self.inner.rank(chunks)


class SpyExtractor:
    def __init__(self, inner, events):
        self.inner, self.events = inner, events
        self.name, self.model = inner.name, inner.model

    def extract(self, item):
        self.events.append(("extract", type(item).__name__, item.chunk_id))
        return self.inner.extract(item)

    def cache_settings(self):
        return self.inner.cache_settings()

    def estimate_cost_usd(self, item):
        return self.inner.estimate_cost_usd(item)


class SpyAnswerer:
    def __init__(self, inner, events):
        self.inner, self.events = inner, events

    def answer(self, question, context):
        self.events.append(("answer", type(question).__name__, type(context).__name__, question))
        return self.inner.answer(question, context)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SpyRetriever:
    def __init__(self, inner, events, fail_on=None):
        self.inner, self.events, self.fail_on = inner, events, fail_on

    def retrieve(self, question):
        self.events.append(("retrieve", question))
        if self.fail_on is not None and question == self.fail_on:
            raise RuntimeError("PageRank did not converge")
        return self.inner.retrieve(question)


def spy_components(events, *, extractor_wrap=lambda e: e, answerer=None, judge=None, fail_retrieval_on=None,
                   strategy_wrap=lambda s: s, **overrides) -> Components:
    """Real components, observed. `answerer` / `judge` replace the real builders (e.g. with fake-client ones)."""
    real = Components()

    def load_corpus(cfg):
        events.append(("load_corpus", cfg.dataset))
        return real.load_corpus(cfg)

    def build_strategy(cfg):
        return strategy_wrap(SpyStrategy(real.build_strategy(cfg), events))

    def build_extractor(cfg, dry_run=False):
        events.append(("build_extractor",))
        return SpyExtractor(extractor_wrap(real.build_extractor(cfg, dry_run=dry_run)), events)

    def build_answerer(cfg, dry_run=False):
        events.append(("build_answerer",))
        inner = answerer(cfg) if answerer is not None else real.build_answerer(cfg, dry_run=dry_run)
        return SpyAnswerer(inner, events)

    def build_judge(cfg, dry_run=False):
        events.append(("build_judge",))
        return judge(cfg) if judge is not None else real.build_judge(cfg, dry_run=dry_run)

    def make_retriever(cfg, graph, chunks, embedder=None):
        events.append(("make_retriever", labels(chunks)))
        return SpyRetriever(real.make_retriever(cfg, graph, chunks, embedder), events, fail_on=fail_retrieval_on)

    c = Components(load_corpus=load_corpus, build_strategy=build_strategy, build_extractor=build_extractor,
                   build_answerer=build_answerer, build_judge=build_judge, make_retriever=make_retriever)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


# ------------------------------------------------------------------------------- extractor doubles
class FakeOpenAIExtractor(MockExtractor):
    """Free, but reports itself as a non-mock backend ('openai') so reportability can be tested."""
    name = "openai"


class FailingChunksExtractor(MockExtractor):
    """Per-chunk failure: every chunk (or every other chunk) comes back malformed."""
    def __init__(self, every=1):
        self.every, self.n = every, 0

    def extract(self, item):
        self.n += 1
        if self.n % self.every == 0:
            return ExtractionResult(chunk_id=item.chunk_id, status="malformed", error="invalid JSON",
                                    extractor=self.name, model=self.model, prompt_version="n/a",
                                    usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.0))
        return super().extract(item)


class AbortingExtractor(MockExtractor):
    """Run-level failure after `after` successful chunks (bad key / no quota)."""
    def __init__(self, after=0):
        self.after, self.n = after, 0

    def extract(self, item):
        self.n += 1
        if self.n > self.after:
            raise ExtractionAbort("insufficient_quota: you exceeded your current quota")
        return super().extract(item)


class FakeOpenAIAbortingExtractor(AbortingExtractor):
    """Aborts like AbortingExtractor but reports itself as a non-mock ('openai') backend."""
    name = "openai"


class UnknownUsageExtractor(MockExtractor):
    def extract(self, item):
        return super().extract(item).model_copy(update={"usage": Usage.unknown()})


class CostlyExtractor(MockExtractor):
    """Every call costs $0.01 (known); the pre-run estimate is configurable (for the spending-cap test)."""
    def __init__(self, estimate=0.0):
        self.estimate = estimate

    def extract(self, item):
        return super().extract(item).model_copy(update={"usage": Usage(input_tokens=100, output_tokens=50, cost_usd=0.01)})

    def estimate_cost_usd(self, item):
        return self.estimate


# ------------------------------------------------------------------------------- LLM doubles
def answer_reply(text="x"):
    return make_response(payload={"reasoning": "r", "answer": text})


def judge_verdict(correct=True):
    return make_response(payload={"reasoning": "ok", "correct": correct})


def fake_answerer(script=None, handler=None):
    """An Answerer on the fake OpenAI client (backend 'openai', so it is not a mock)."""
    fake = FakeOpenAIClient(script=script, handler=handler)
    return Answerer("openai", client=ChatJSONClient(client=fake, sleep=lambda s: None))


def fake_judge(script=None, handler=None, model="gpt-4o"):
    fake = FakeOpenAIClient(script=script, handler=handler)
    return Judge("openai", client=ChatJSONClient(model=model, client=fake, sleep=lambda s: None))
