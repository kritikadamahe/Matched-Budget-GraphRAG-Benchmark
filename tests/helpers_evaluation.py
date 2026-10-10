"""Test doubles for the Evaluation tests: builders for AnswerResult / PredictionRecord /
gold data, and a scripted judge on top of the extraction tests' fake OpenAI client."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.judge import Judge
from evaluation.schemas import GoldAnswer, PredictionRecord
from extraction.schemas import Usage
from generation.cache import QueryCache
from generation.llm_client import ChatJSONClient
from generation.schemas import AnswerResult
from tests.helpers_extraction import FakeOpenAIClient, make_response

Q = "Where was the director of Forrest Gump born?"


def make_answer(text="Chicago", status="ok", backend="openai", question=Q, usage=None, error=None, **kw):
    return AnswerResult(
        question=question, status=status, backend=backend, model="gpt-4o-mini", prompt_version="answer-v1",
        prompt_sha256="0" * 64, answer=text, error=error,
        usage=usage if usage is not None else Usage(input_tokens=100, output_tokens=20, cost_usd=0.00002), **kw)


def make_prediction(text="Chicago", qid="q1", question=Q, **answer_kw):
    return PredictionRecord(question_id=qid, question=question, strategy="random", budget=0.1, seed=0,
                            answer=make_answer(text, question=question, **answer_kw))


def make_gold(qid="q1", answer="Chicago", aliases=None):
    return GoldAnswer(question_id=qid, answer=answer, aliases=aliases or [])


def judge_reply(correct=True, reasoning="Same city.", **kw):
    return make_response(payload={"reasoning": reasoning, "correct": correct}, **kw)


def openai_judge(script=None, handler=None, cache=None, **client_kw):
    fake = FakeOpenAIClient(script=script, handler=handler)
    client = ChatJSONClient(client=fake, sleep=lambda s: None, **client_kw)
    return Judge("openai", client=client, cache=cache), fake


def unknown_usage(response):
    """Same response, but the API sent no usage information."""
    response.usage = None
    return response
