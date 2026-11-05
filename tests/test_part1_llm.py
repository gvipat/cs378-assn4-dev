"""Part 1: LLM backends, offline. No test here calls Bedrock."""

import asyncio
import json

import pytest

from research_app import prompts
from research_app.corpus import Corpus
from research_app.llm import BedrockLLM, FakeLLM, PermanentLLMError, TransientLLMError
from research_app.prompts import Msg


class StubBedrock:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def converse(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return {
            "output": {"message": {"content": [{"text": '{"subtopics": []}'}]}},
            "usage": {"inputTokens": 11, "outputTokens": 3},
        }


def client_error(code):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": code}}, "Converse")


def test_bedrock_merges_roles_and_reports_usage():
    stub = StubBedrock()
    llm = BedrockLLM("test-model", client=stub)
    msgs = prompts.extract_messages(
        "Q?", prompts.Subtopic("f1", "t", "q"), [Corpus().doc("raft")]
    )
    resp = asyncio.run(llm.complete(msgs, max_tokens=50, stage="extract"))
    (call,) = stub.calls
    assert call["modelId"] == "test-model" and call["system"] == [
        {"text": prompts.SYSTEM_PROMPT}
    ]
    assert [m["role"] for m in call["messages"]] == [
        "user"
    ]  # consecutive user turns merged
    assert len(call["messages"][0]["content"]) == len(msgs) - 1
    assert call["inferenceConfig"] == {"maxTokens": 50}  # no temperature unless asked
    assert (resp.input_tokens, resp.output_tokens) == (11, 3)


def test_bedrock_sends_temperature_only_when_given():
    stub = StubBedrock()
    asyncio.run(
        BedrockLLM("m", client=stub, temperature=0.2).complete(
            [Msg("user", "hi")], max_tokens=5, stage="plan"
        )
    )
    assert stub.calls[0]["inferenceConfig"] == {"maxTokens": 5, "temperature": 0.2}


def test_bedrock_alternates_with_assistant_history():
    system, convo = BedrockLLM.to_converse(
        [
            Msg("system", "s"),
            Msg("user", "a"),
            Msg("user", "b"),
            Msg("assistant", "c"),
            Msg("user", "d"),
        ]
    )
    assert [m["role"] for m in convo] == ["user", "assistant", "user"]


@pytest.mark.parametrize(
    "code,exc",
    [
        ("ThrottlingException", TransientLLMError),
        ("ServiceUnavailableException", TransientLLMError),
        ("ValidationException", PermanentLLMError),
    ],
)
def test_bedrock_error_mapping(code, exc):
    llm = BedrockLLM("m", client=StubBedrock(error=client_error(code)))
    with pytest.raises(exc):
        asyncio.run(llm.complete([Msg("user", "x")], max_tokens=5, stage="plan"))


def test_fake_llm_is_deterministic_and_quotes_verbatim():
    corpus = Corpus()
    docs = [corpus.doc("raft"), corpus.doc("paxos")]
    msgs = prompts.extract_messages(
        "How does Raft elect a leader?",
        prompts.Subtopic(
            "f1", "Raft leader election", "raft leader election heartbeat"
        ),
        docs,
    )
    a = asyncio.run(FakeLLM().complete(msgs, max_tokens=700, stage="extract"))
    b = asyncio.run(FakeLLM().complete(msgs, max_tokens=700, stage="extract"))
    assert a == b
    ev = json.loads(a.text)["evidence"]
    assert 1 <= len(ev) <= 3
    texts = {d.doc_id: d.text for d in docs}
    assert all(e["quote"] in texts[e["doc_id"]] for e in ev)
    assert a.input_tokens > 1000 and a.output_tokens > 10


def test_fake_plan_uses_known_subtopics():
    from research_app.questions import load_questions

    q = load_questions(public=True)[0]
    resp = asyncio.run(
        FakeLLM().complete(
            prompts.plan_messages(q["question"]), max_tokens=400, stage="plan"
        )
    )
    assert json.loads(resp.text)["subtopics"] == [
        {"title": f["title"], "query": f["query"]} for f in q["subtopics"]
    ]


def test_parse_json_tolerates_fences_and_braces_in_strings():
    assert prompts.parse_json('```json\n{"a": "x}y", "b": [1]}\n```') == {
        "a": "x}y",
        "b": [1],
    }
    assert prompts.parse_json('Sure! {"a": "say \\"hi\\""} done') == {"a": 'say "hi"'}
    assert prompts.parse_json("no json") is None
