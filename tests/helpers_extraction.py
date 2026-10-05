"""Test doubles for the extraction tests: a fake OpenAI client, fake API errors, and
an extractor that records every chunk it is asked to process."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extraction.mock_extractor import MockExtractor
from src.config import ExperimentConfig
from src.corpus import build_chunk_manifest
from strategies import build_strategy

GOOD_PAYLOAD = {
    "entities": [
        {"name": "Alpha", "type": "PERSON", "description": "a person"},
        {"name": "Beta", "type": "WORK", "description": "a film"},
    ],
    "relationships": [{"source": "Alpha", "target": "Beta", "description": "made"}],
}


def make_response(payload=None, content=None, finish_reason="stop", refusal=None,
                  prompt_tokens=100, completion_tokens=50):
    """Looks like an OpenAI chat-completion response, as far as the extractor reads it."""
    if content is None:
        content = json.dumps(GOOD_PAYLOAD if payload is None else payload)
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def make_error(class_name, status=None, code=None, message="boom"):
    """An exception whose class NAME and attributes match the real OpenAI SDK errors."""
    err = type(class_name, (Exception,), {})(message)
    err.status_code = status
    err.code = code
    return err


class FakeOpenAIClient:
    """Stands in for openai.OpenAI. `script` = responses/exceptions returned in order;
    `handler(kwargs)` is used once the script is empty. Every request is recorded."""

    def __init__(self, script=None, handler=None):
        self.script = list(script or [])
        self.handler = handler
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.script:
            step = self.script.pop(0)
        elif self.handler is not None:
            step = self.handler(kwargs)
        else:
            raise AssertionError("fake client ran out of scripted responses")
        if isinstance(step, BaseException):
            raise step
        return step


def good_handler(prompt_tokens=100, completion_tokens=50):
    return lambda kwargs: make_response(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class RecordingExtractor(MockExtractor):
    """Free extractor that remembers the id of every chunk it was asked to process."""

    def __init__(self):
        self.seen = []

    def extract(self, item):
        self.seen.append(item.chunk_id)
        return super().extract(item)


def make_cfg(strategy="random", **extra):
    return ExperimentConfig(experiment_id="t", seed=0, dataset="hotpotqa", num_questions=5,
                            strategy=strategy, budget=0.1, fast_use_spacy=False,
                            ketrag_mode="tfidf", embedding_backend="tfidf", **extra)


def corpus():
    """37 chunks from the bundled mock HotpotQA fixture (same setup as the strategy tests)."""
    chunks, questions, _ = build_chunk_manifest("hotpotqa", "mock", 5, 42, 8, 2)
    return chunks, questions


def rank_with(strategy_name, chunks):
    """Rank through the real strategy registry, exactly as the pipeline CLI does."""
    return build_strategy(make_cfg(strategy_name)).rank(chunks)
