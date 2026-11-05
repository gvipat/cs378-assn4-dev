"""Provider responses become saved source text, not generated evidence."""

import asyncio
import json

import httpx
import pytest

from research_app import ResearchAgent
from research_app.corpus import PermanentToolError, TransientToolError
from research_app.llm import FakeLLM
from research_app.search import make_search, SourceArchive
from research_app.citations import check_report
from research_app import web_pages

TEXT = "Raft consensus handles leader failures by electing a new leader and preserving committed log entries."
URL = "https://example.org/raft"


@pytest.mark.parametrize("name", ["tavily", "perplexity", "openai"])
def test_provider_runs_same_graph_and_preserves_source_text(
    name, tmp_path, monkeypatch
):
    async def public(url):
        assert url == URL

    monkeypatch.setattr(web_pages, "_check_public_url", public)

    async def run():
        requests = []

        def provider(request):
            requests.append(request)
            if request.method == "GET":
                assert name == "openai" and "authorization" not in request.headers
                return httpx.Response(
                    200,
                    text=f"<html><script>not evidence</script><p>{TEXT}</p></html>",
                    headers={"content-type": "text/html"},
                )
            assert request.headers["authorization"] == "Bearer test-only"
            body = json.loads(request.content)
            if name == "tavily":
                if request.url.path == "/search":
                    assert (
                        body["include_answer"] is False
                        and body["include_usage"] is True
                    )
                    return httpx.Response(
                        200,
                        json={
                            "results": [{"url": URL, "title": "Raft"}],
                            "usage": {"credits": 1},
                        },
                    )
                assert request.url.path == "/extract" and body["urls"] == [URL]
                return httpx.Response(
                    200,
                    json={
                        "results": [{"url": URL, "raw_content": TEXT}],
                        "usage": {"credits": 1},
                    },
                )
            if name == "perplexity":
                assert request.url.path == "/search" and "model" not in body
                return httpx.Response(
                    200,
                    json={"results": [{"url": URL, "title": "Raft", "snippet": TEXT}]},
                )
            assert request.url.path == "/v1/responses"
            assert body["max_tool_calls"] == 1 and body["tool_choice"] == "required"
            assert body["include"] == ["web_search_call.action.sources"]
            return httpx.Response(
                200,
                json={
                    "id": "response-id",
                    "usage": {"input_tokens": 8, "output_tokens": 5},
                    "output": [
                        {
                            "type": "web_search_call",
                            "action": {"sources": [{"url": URL}]},
                        },
                        {
                            "type": "message",
                            "content": [
                                {
                                    "text": "GENERATED TEXT IS NOT EVIDENCE",
                                    "annotations": [
                                        {
                                            "type": "url_citation",
                                            "url": URL,
                                            "title": "Raft",
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            extra = {"model": "test-model"} if name == "openai" else {}
            search = make_search(name, api_key="test-only", client=client, **extra)
            agent = ResearchAgent(llm=FakeLLM(bad_citation_rate=0), search=search)
            result = await agent.research(
                "How does Raft handle leader failures?", out=tmp_path / name
            )
            assert result.valid
            saved = SourceArchive.load(tmp_path / name / "sources.json")
            assert not check_report(result.report, saved)
            assert all(
                saved.doc(e.doc_id, e.snapshot).text == TEXT
                for e in result.report.evidence
            )
            assert all(r["provider"] == name for r in result.search_receipts)
            assert "test-only" not in (tmp_path / name / "summary.json").read_text()
            fetches = [
                r for r in requests if r.method == "GET" or r.url.path == "/extract"
            ]
            assert len(fetches) == (0 if name == "perplexity" else 1)
            if name == "openai":
                assert all(
                    r["usage"]["input_tokens"] == 8 and r["web_search_calls"] == 1
                    for r in result.search_receipts
                )

    asyncio.run(run())


@pytest.mark.parametrize("name", ["tavily", "perplexity", "openai"])
@pytest.mark.parametrize(
    "status,error",
    [(401, PermanentToolError), (429, TransientToolError), (503, TransientToolError)],
)
def test_http_failures_preserve_retry_classification(name, status, error):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(status))
        ) as client:
            extra = {"model": "test-model"} if name == "openai" else {}
            backend = make_search(name, api_key="test-only", client=client, **extra)
            with pytest.raises(error) as failure:
                await backend.search("Raft", op_id="R00/search")
            assert "test-only" not in str(failure.value)

    asyncio.run(run())


def test_openai_generated_answer_cannot_replace_web_sources():
    async def run():
        response = {"output": [{"type": "message", "content": [{"text": TEXT}]}]}
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response))
        ) as client:
            backend = make_search(
                "openai", api_key="test-only", model="test-model", client=client
            )
            with pytest.raises(PermanentToolError, match="no web sources"):
                await backend.search("Raft", op_id="R00/search")

    asyncio.run(run())


def test_direct_fetch_checks_redirect_and_rejects_pdf(monkeypatch):
    async def run():
        checked = []

        async def public(url):
            checked.append(url)
            if "internal" in url:
                raise PermanentToolError("source must resolve to a public address")

        monkeypatch.setattr(web_pages, "_check_public_url", public)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(302, headers={"location": "http://internal/"})
            )
        ) as client:
            with pytest.raises(PermanentToolError, match="public address"):
                await web_pages.fetch_page_text(URL, client=client)
        assert checked == [URL, "http://internal/"]
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, content=b"%PDF", headers={"content-type": "application/pdf"}
                )
            )
        ) as client:
            with pytest.raises(PermanentToolError, match="HTML and plain-text"):
                await web_pages.fetch_page_text(URL, client=client)

    asyncio.run(run())
