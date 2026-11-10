"""Part 1: LLM backends, offline. No test here calls Bedrock or the course proxy."""

import asyncio
import json

import httpx
import pytest

from research_app import llm, prompts
from research_app.corpus import Corpus
from research_app.llm import BedrockLLM, FakeLLM, PermanentLLMError, TransientLLMError
from research_app.prompts import Msg


class StubProxy:
    """Stands in for the course LLM proxy's /v1/chat/completions."""

    def __init__(self, status=200, error=None):
        self.status, self.error = status, error
        self.calls = []

    def handler(self, request):
        self.calls.append(json.loads(request.content))
        if self.error:
            raise self.error
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "stub"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"subtopics": []}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    def client(self):
        return httpx.Client(
            base_url="http://proxy.test", transport=httpx.MockTransport(self.handler)
        )


def test_bedrock_merges_roles_and_reports_usage():
    stub = StubProxy()
    llm = BedrockLLM("test-model", client=stub.client())
    msgs = prompts.extract_messages(
        "Q?", prompts.Subtopic("f1", "t", "q"), [Corpus().doc("raft")]
    )
    resp = asyncio.run(llm.complete(msgs, max_tokens=50, stage="extract"))
    (call,) = stub.calls
    assert call["model"] == "test-model"
    system, *convo = call["messages"]
    assert system == {
        "role": "system",
        "content": [{"type": "text", "text": prompts.SYSTEM_PROMPT}],
    }
    assert [m["role"] for m in convo] == ["user"]  # consecutive user turns merged
    assert len(convo[0]["content"]) == len(msgs) - 1
    assert call["max_tokens"] == 50 and "temperature" not in call  # only when asked
    assert (resp.input_tokens, resp.output_tokens) == (11, 3)


def test_bedrock_sends_temperature_only_when_given():
    stub = StubProxy()
    asyncio.run(
        BedrockLLM("m", client=stub.client(), temperature=0.2).complete(
            [Msg("user", "hi")], max_tokens=5, stage="plan"
        )
    )
    assert stub.calls[0]["max_tokens"] == 5 and stub.calls[0]["temperature"] == 0.2


def test_bedrock_alternates_with_assistant_history():
    convo = BedrockLLM.to_chat(
        [
            Msg("system", "s"),
            Msg("user", "a"),
            Msg("user", "b"),
            Msg("assistant", "c"),
            Msg("user", "d"),
        ]
    )
    assert [m["role"] for m in convo] == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize(
    "stub,exc",
    [
        (StubProxy(status=429), TransientLLMError),  # rate limited
        (StubProxy(status=503), TransientLLMError),
        (StubProxy(error=httpx.ReadTimeout("slow")), TransientLLMError),
        (StubProxy(status=422), PermanentLLMError),  # group budget spent
        (StubProxy(status=401), PermanentLLMError),  # bad key
    ],
)
def test_bedrock_error_mapping(stub, exc):
    llm = BedrockLLM("m", client=stub.client())
    with pytest.raises(exc):
        asyncio.run(llm.complete([Msg("user", "x")], max_tokens=5, stage="plan"))


def test_proxy_client_needs_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTKV_LLM_KEY", raising=False)
    monkeypatch.setattr(llm, "KEY_FILE", tmp_path / "missing")
    with pytest.raises(ValueError, match="no LLM key"):
        llm.proxy_client()


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
