"""Optional reasoning_effort: sent only when set, and part of every cache key when set."""
import json
from types import SimpleNamespace

from extraction.openai_extractor import OpenAIExtractor
from generation.answerer import Answerer
from generation.llm_client import ChatJSONClient
from generation.prompts import ANSWER_RESPONSE_FORMAT
from generation.schemas import AnswerOutput


class FakeCompletions:
    def __init__(self, content):
        self.content, self.calls = content, []

    def create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content=self.content, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def fake(content):
    comp = FakeCompletions(content)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


def test_chat_client_sends_reasoning_effort_only_when_set():
    reply = json.dumps({"reasoning": "r", "answer": "a"})
    for effort, expected in ((None, False), ("none", True)):
        client, comp = fake(reply)
        c = ChatJSONClient(client=client, reasoning_effort=effort)
        assert c.call([{"role": "user", "content": "q"}], ANSWER_RESPONSE_FORMAT, AnswerOutput).status == "ok"
        assert ("reasoning_effort" in comp.calls[0]) is expected
        if expected:
            assert comp.calls[0]["reasoning_effort"] == "none"


def test_reasoning_effort_changes_answer_cache_key_only_when_set():
    reply = json.dumps({"reasoning": "r", "answer": "a"})
    plain = Answerer("openai", client=ChatJSONClient(client=fake(reply)[0])).settings()
    tuned = Answerer("openai", client=ChatJSONClient(client=fake(reply)[0], reasoning_effort="none")).settings()
    assert "reasoning_effort" not in plain
    assert tuned["reasoning_effort"] == "none"


def test_extractor_cache_settings():
    client = fake("{}")[0]
    assert "reasoning_effort" not in OpenAIExtractor(client=client).cache_settings()
    assert OpenAIExtractor(client=client, reasoning_effort="low").cache_settings()["reasoning_effort"] == "low"
