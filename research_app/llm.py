"""LLM backends (supplied).

    FakeLLM     deterministic and rule-based; recognises the plan / extract /
                synthesize / repair steps and returns valid structured output.
                Used by the tests, offline runs, and staff trace generation.
    BedrockLLM  Amazon Bedrock through the Converse API. Model and region come
                from AGENTKV_BEDROCK_MODEL and AWS_REGION.

Both implement `await llm.complete(messages, max_tokens=..., stage=...)`.
Transient failures (throttling, timeouts) raise TransientLLMError; the tracing
client retries them inside the call's concurrency slot.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

from agentkv import tokenizer
from agentkv.trace import output_text

from .corpus import terms
from .prompts import Msg, parse_json, render

QUESTIONS_PATH = Path(__file__).parent / "data" / "questions.json"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class TransientLLMError(Exception):
    pass


class PermanentLLMError(Exception):
    pass


class LLM(Protocol):
    name: str

    async def complete(
        self, messages: list[Msg], *, max_tokens: int, stage: str
    ) -> LLMResponse: ...


def prompt_tokens(messages: list[Msg]) -> int:
    return sum(len(ids) for ids in tokenizer.encode_batch(render(messages)))


# ---------------------------------------------------------------- fake


@dataclass(frozen=True)
class FakeLatency:
    """Assign simulated service times to timed FakeLLM calls."""

    base: float = 0.35
    per_input_token: float = 0.00004
    per_output_token: float = 0.012
    scale: float = 1.0

    def seconds(self, input_tokens: int, output_tokens: int) -> float:
        return self.scale * (
            self.base
            + self.per_input_token * input_tokens
            + self.per_output_token * output_tokens
        )


@lru_cache(maxsize=1)
def _known_subtopics() -> dict[str, list[dict]]:
    qs = json.loads(QUESTIONS_PATH.read_text())["questions"]
    return {_norm(q["question"]): q["subtopics"] for q in qs}


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _digest(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:12], 16)


DOC_RE = re.compile(
    r'<document id="([^"]+)" snapshot="([^"]+)" title="([^"]*)">\n(.*)\n</document>',
    re.S,
)
SENT_RE = re.compile(r"[^\n.!?]+(?:[.!?]+(?=\s|$))?")


class FakeLLM:
    """Generate repeatable replies for testing the agent without a model API.

    Rules for each stage produce plans, quoted passages, and reports in the
    expected JSON format. Set bad_citation_rate to introduce invalid evidence
    ids in a fraction of reports and exercise the repair step. If the optional
    transient(stage, attempt) callback returns True, raise TransientLLMError
    so the tracing client can exercise its retry logic.
    """

    name = "fake-rule-based-1"

    def __init__(
        self,
        *,
        latency: FakeLatency | None = None,
        bad_citation_rate: float = 0.2,
        transient: Callable[[str, int], bool] | None = None,
    ):
        self.latency = latency
        self.bad_citation_rate = bad_citation_rate
        self.transient = transient
        self._attempts: dict[int, int] = {}
        self.calls = 0

    async def complete(
        self, messages: list[Msg], *, max_tokens: int, stage: str
    ) -> LLMResponse:
        key = _digest(stage, *(m.text for m in messages))
        attempt = self._attempts[key] = self._attempts.get(key, 0) + 1
        in_tok = prompt_tokens(messages)
        if self.transient and self.transient(stage, attempt):
            if self.latency:
                await asyncio.sleep(self.latency.scale * self.latency.base)
            raise TransientLLMError(f"fake throttle on {stage} attempt {attempt}")
        handler = {
            "plan": self._plan,
            "extract": self._extract,
            "synthesize": self._synthesize,
            "repair": self._repair,
        }.get(stage)
        if handler is None:
            raise PermanentLLMError(f"FakeLLM has no rule for stage {stage!r}")
        text = handler(messages)
        out_tok = tokenizer.count(output_text(text))
        if self.latency:
            await asyncio.sleep(self.latency.seconds(in_tok, out_tok))
        self.calls += 1
        return LLMResponse(text, in_tok, out_tok, self.name)

    # -- steps

    def _plan(self, messages: list[Msg]) -> str:
        question = messages[-1].text.removeprefix("Question:").strip()
        subtopics = _known_subtopics().get(_norm(question))
        if subtopics is None:
            subtopics = _heuristic_subtopics(question)
        return json.dumps(
            {
                "subtopics": [
                    {"title": f["title"], "query": f["query"]} for f in subtopics
                ]
            }
        )

    def _extract(self, messages: list[Msg]) -> str:
        docs = [
            (m.group(1), m.group(4))
            for msg in messages
            for m in [DOC_RE.search(msg.text)]
            if m
        ]
        task = messages[-1].text
        subtopic = re.search(r"Subtopic: (.*)", task)
        query = re.search(r"Search query: (.*)", task)
        want = set(
            terms(
                f"{subtopic.group(1) if subtopic else ''} {query.group(1) if query else ''}"
            )
        )
        need = (
            1 if len(want) <= 3 else 2
        )  # minimum number of subtopic terms that a sentence must contain
        scored = []
        for d_index, (doc_id, text) in enumerate(docs):
            for m in SENT_RE.finditer(text):
                sent = m.group(0).strip()
                if len(sent) < 40:
                    continue
                if len(sent) > 320:
                    sent = sent[:320].rsplit(" ", 1)[0]
                score = len(want & set(terms(sent)))
                if score >= need:
                    scored.append((-score, d_index, m.start(), doc_id, sent))
        scored.sort()
        picked, per_doc = [], {}
        for _, _, _, doc_id, sent in scored:
            if per_doc.get(doc_id, 0) >= 2:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            picked.append({"doc_id": doc_id, "quote": sent})
            if len(picked) == 3:
                break
        return json.dumps({"evidence": picked}, ensure_ascii=False)

    def _synthesize(self, messages: list[Msg]) -> str:
        head = messages[-2].text
        question = re.search(r"Question: (.*)", head).group(1)
        plan = re.findall(r"^(f\d+)\. (.*)$", head, re.M)
        ev = re.findall(
            r'^\[(E\d+)\] \(([^;]+); subtopic (f\d+): [^)]*\) "(.*)"$',
            messages[-1].text,
            re.M,
        )
        claims = []
        for fid, title in plan:
            mine = [(eid, quote) for eid, _, f, quote in ev if f == fid]
            for i in range(0, min(len(mine), 4), 2):
                group = mine[i : i + 2]
                claims.append(
                    {
                        "text": f"{title}: {_condense(group[0][1])}",
                        "evidence_ids": [e for e, _ in group],
                    }
                )
        if claims and _digest("flaw", question) % 1000 < self.bad_citation_rate * 1000:
            claims.append(
                {
                    "text": "Further sources discuss related trade-offs.",
                    "evidence_ids": [f"E{len(ev) + 7}"],
                }
            )
        return json.dumps(
            {"title": question.rstrip("?"), "claims": claims}, ensure_ascii=False
        )

    def _repair(self, messages: list[Msg]) -> str:
        draft = parse_json(messages[-2].text) or {"title": "", "claims": []}
        problems = messages[-1].text
        bad_ids = set(re.findall(r"cites (E\d+)", problems))
        bad_claims = {
            int(i) for i in re.findall(r"uncited_claim \(claim (\d+)\)", problems)
        }
        claims = []
        for i, c in enumerate(draft.get("claims", [])):
            ids = [e for e in c.get("evidence_ids", []) if e not in bad_ids]
            if ids and i not in bad_claims:
                claims.append({"text": c["text"], "evidence_ids": ids})
        return json.dumps(
            {"title": draft.get("title", ""), "claims": claims}, ensure_ascii=False
        )


def _condense(quote: str, words: int = 28) -> str:
    w = quote.split()
    s = " ".join(w[:words])
    return s if s.endswith((".", "!", "?")) else s.rstrip(",;:") + "."


def _heuristic_subtopics(question: str) -> list[dict]:
    q = question.rstrip("?")
    parts = [
        p.strip()
        for p in re.split(r",| and | versus | vs\.? ", q)
        if len(terms(p)) >= 2
    ][:3]
    if len(parts) < 2:
        key = " ".join(terms(q)[:6])
        parts = [q, f"{key} trade-offs"]
    return [{"title": p[:60], "query": " ".join(terms(p)[:6])} for p in parts]


# ---------------------------------------------------------------- bedrock

THROTTLE_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "TooManyRequestsException",
    "InternalServerException",
    "ModelTimeoutException",
}


class BedrockLLM:
    """Send model requests through Amazon Bedrock's Converse API.

    Disable botocore retries so the tracing client records and counts every
    attempt. Send temperature only when it is explicitly configured, since
    support for that parameter varies by model.
    """

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        *,
        temperature: float | None = None,
        client=None,
    ):
        self.model_id = model_id or os.environ.get("AGENTKV_BEDROCK_MODEL", "")
        if not self.model_id:
            raise ValueError(
                "set AGENTKV_BEDROCK_MODEL to a Bedrock model id or inference profile"
            )
        self.name = self.model_id
        self.temperature = temperature
        if client is None:
            import boto3
            from botocore.config import Config

            region = (
                region
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-west-2"
            )
            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(
                    retries={"max_attempts": 1, "mode": "standard"}, read_timeout=120
                ),
            )
        self.client = client

    @staticmethod
    def to_converse(messages: list[Msg]) -> tuple[list[dict], list[dict]]:
        """Convert the prompt to Bedrock's system and conversation fields.

        Merge consecutive messages with the same role into one message containing
        multiple content blocks, so the conversation alternates between roles.
        """
        system = [{"text": m.text} for m in messages if m.role == "system"]
        convo: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            if convo and convo[-1]["role"] == m.role:
                convo[-1]["content"].append({"text": m.text})
            else:
                convo.append({"role": m.role, "content": [{"text": m.text}]})
        return system, convo

    async def complete(
        self, messages: list[Msg], *, max_tokens: int, stage: str
    ) -> LLMResponse:
        from botocore.exceptions import (
            ClientError,
            ConnectionError as BotoConnectionError,
            ReadTimeoutError,
        )

        system, convo = self.to_converse(messages)
        config = {"maxTokens": max_tokens}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        try:
            resp = await asyncio.to_thread(
                self.client.converse,
                modelId=self.model_id,
                messages=convo,
                system=system,
                inferenceConfig=config,
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in THROTTLE_CODES:
                raise TransientLLMError(code) from e
            raise PermanentLLMError(f"{code}: {e}") from e
        except (BotoConnectionError, ReadTimeoutError) as e:
            raise TransientLLMError(type(e).__name__) from e
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        usage = resp.get("usage", {})
        return LLMResponse(
            text,
            usage.get("inputTokens", 0),
            usage.get("outputTokens", 0),
            self.model_id,
        )


def make_llm(backend: str, **kw) -> LLM:
    if backend == "fake":
        return FakeLLM(**kw)
    if backend == "bedrock":
        return BedrockLLM(**kw)
    raise ValueError(f"unknown backend {backend!r}")
