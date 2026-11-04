"""The topic interface uses the same graph with local or asynchronous web retrieval."""

import asyncio
import hashlib
import json

import httpx
import pytest

from research_app import ResearchAgent
from research_app.artifacts import Claim, Evidence, Report
from research_app.citations import check_report
from research_app.corpus import Doc, PermanentToolError, ToolLatency
from research_app.llm import FakeLLM
from research_app.search import ExaSearch, LocalSearch, SourceArchive

TOPIC = "How does Raft consensus handle leader failures?"
TEXT = "Raft consensus handles leader failures by electing a new leader and preserving committed log entries."


def test_topic_api_local_and_saved_sources_can_be_checked_offline(tmp_path):
    async def run():
        agent = ResearchAgent(
            llm=FakeLLM(bad_citation_rate=0),
            search=LocalSearch(latency=ToolLatency(scale=0)),
        )
        result = await agent.research(TOPIC, out=tmp_path / "local")
        assert result.valid
        assert result.report.question == TOPIC
        saved = SourceArchive.load(tmp_path / "local/sources.json")
        assert not check_report(result.report, saved)
        assert "## Sources" in (tmp_path / "local/report.md").read_text()
        assert (tmp_path / "local/replay.json").exists()
        before = agent.llm.calls
        with pytest.raises(FileExistsError):
            await agent.research(TOPIC, out=tmp_path / "local")
        assert (
            agent.llm.calls == before
        )  # Do not spend tokens before noticing an existing output directory.

    asyncio.run(run())


def test_web_backend_retries_runs_fetches_in_parallel_and_saves_exact_sources(tmp_path):
    async def run():
        searches, fetching, peak = 0, 0, 0
        content_requests = []

        async def provider(request):
            nonlocal searches, fetching, peak
            assert request.headers["x-api-key"] == "test-only"
            body = json.loads(request.content)
            if request.url.path == "/search":
                searches += 1
                if searches == 1:
                    return httpx.Response(429)
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "url": f"https://example.org/paper-{i}",
                                "title": f"Raft paper {i}",
                            }
                            for i in range(3)
                        ]
                    },
                )
            assert request.url.path == "/contents" and body["text"] is True
            url = body["urls"][0]
            content_requests.append(url)
            fetching += 1
            peak = max(peak, fetching)
            await asyncio.sleep(0.02)
            fetching -= 1
            return httpx.Response(
                200,
                json={
                    "requestId": url,
                    "costDollars": {"total": 0.001},
                    "results": [{"url": url, "title": 'Raft "paper"', "text": TEXT}],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            search = ExaSearch("test-only", client=client)
            agent = ResearchAgent(llm=FakeLLM(bad_citation_rate=0), search=search)
            result = await agent.research(TOPIC, out=tmp_path / "web")
            assert result.valid and peak >= 2
            assert len(content_requests) == len(set(content_requests)) == 3
            ops = result.recording.ops
            assert any(op.kind == "search" and op.retries == 1 for op in ops)
            assert any(op.kind == "fetch" and op.delay >= 0.02 for op in ops)
            assert not check_report(
                result.report, SourceArchive.load(tmp_path / "web/sources.json")
            )
            assert all(
                e.uri.startswith("https://example.org/") for e in result.report.evidence
            )
            first_ids = {r["op_id"] for r in result.search_receipts}
            again = await agent.research(TOPIC)
            assert first_ids.isdisjoint(r["op_id"] for r in again.search_receipts)
            assert result.report.question == again.report.question

    asyncio.run(run())


def test_source_archive_preserves_both_versions_and_detects_tampering(tmp_path):
    def doc(text):
        return Doc(
            "url-id",
            "web",
            "Paper",
            "https://example.org/paper",
            "retrieved",
            hashlib.sha256(text.encode()).hexdigest(),
            0,
            "https://example.org/paper",
            text,
        )

    old, new = doc(TEXT), doc("Changed content.")
    sources = SourceArchive([old, new])
    evidence = Evidence(
        "E1", old.doc_id, old.sha256, 0, len(TEXT), TEXT, "f1", old.uri, "extract"
    )
    report = Report(
        "R00",
        "topic",
        TOPIC,
        "Raft",
        [Claim("Raft elects a leader.", ["E1"])],
        [evidence],
    )
    assert not check_report(report, sources)
    path = tmp_path / "sources.json"
    sources.save(path)
    data = json.loads(path.read_text())
    data["documents"][0]["text"] += " changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        SourceArchive.load(path)


@pytest.mark.parametrize(
    "response", [httpx.Response(401), httpx.Response(200, json={"results": []})]
)
def test_api_failures_do_not_become_fake_source_text(response):
    async def run():
        def provider(request):
            if request.url.path == "/search":
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {"url": "https://example.org/paper", "title": "Paper"}
                        ]
                    },
                )
            return response

        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            search = ExaSearch("test-only", client=client)
            (hit,) = await search.search("Raft", op_id="R00/search")
            with pytest.raises(PermanentToolError) as error:
                await search.fetch(hit.doc_id, op_id="R00/fetch")
            assert "test-only" not in str(error.value)
            assert not search.corpus.has(hit.doc_id)

    asyncio.run(run())
