"""Retrieve local or web sources for the research workflow.

Backends return search results and document text, and retain the exact source
versions used by the citation checker. The graph handles planning, evidence
extraction, and report generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Protocol

from .corpus import (
    Corpus,
    Doc,
    SearchHit,
    Tools,
    PermanentToolError,
    TransientToolError,
)


class DocumentStore(Protocol):
    def has(self, doc_id: str) -> bool: ...
    def snapshot_of(self, doc_id: str, sha256: str) -> str | None: ...
    def doc(self, doc_id: str, snapshot: str | None = None) -> Doc: ...


class SearchBackend(Protocol):
    """The retrieval interface used by TracingClient.

    Return a duration for simulated local tools, or None to measure elapsed
    time for a real API call. Async search/fetch methods must yield while
    waiting for network I/O so that research branches can run concurrently.
    """

    name: str
    corpus: DocumentStore

    def duration(self, kind: str, op_id: str, attempt: int) -> float | None: ...
    def search(
        self, query: str, *, op_id: str, attempt: int = 1, k: int = 5
    ) -> list[SearchHit] | Awaitable[list[SearchHit]]: ...
    def fetch(
        self, doc_id: str, *, op_id: str, attempt: int = 1
    ) -> Doc | Awaitable[Doc]: ...


class LocalSearch(Tools):
    """Search the supplied corpus, with the assignment's simulated tool delays."""

    name = "local-corpus"

    def __init__(self, corpus: Corpus | None = None, *, latency=None, faults=None):
        super().__init__(corpus or Corpus("v2"), latency, faults)


class SourceArchive:
    """Keep fetched text by content hash so citations survive changes to a web page."""

    def __init__(self, documents=()):
        self._docs: dict[str, dict[str, Doc]] = {}
        for doc in documents:
            self.add(doc)

    def add(self, doc: Doc) -> None:
        if hashlib.sha256(doc.text.encode()).hexdigest() != doc.sha256:
            raise ValueError("source text does not match its SHA-256 hash")
        self._docs.setdefault(doc.doc_id, {})[doc.sha256] = doc

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def snapshot_of(self, doc_id: str, sha256: str) -> str | None:
        return sha256 if sha256 in self._docs.get(doc_id, {}) else None

    def doc(self, doc_id: str, snapshot: str | None = None) -> Doc:
        versions = self._docs[doc_id]
        return (
            versions[snapshot]
            if snapshot is not None
            else next(reversed(versions.values()))
        )

    def save(self, path: str | Path) -> None:
        docs = [
            asdict(doc) for versions in self._docs.values() for doc in versions.values()
        ]
        Path(path).write_text(
            json.dumps(
                {"schema": "research-sources/1", "documents": docs},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: str | Path) -> "SourceArchive":
        data = json.loads(Path(path).read_text())
        if data.get("schema") != "research-sources/1":
            raise ValueError("unsupported source archive format")
        return cls(Doc(**doc) for doc in data["documents"])


class _WebSearch:
    """Share HTTP handling and source snapshots across the web adapters."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_chars: int = 24000,
        timeout: float = 30.0,
        client=None,
    ):
        self._api_key = api_key or os.environ.get(self.key_env, "")
        if not self._api_key:
            raise ValueError(f"set {self.key_env} before using {type(self).__name__}")
        if max_chars <= 0 or timeout <= 0:
            raise ValueError("max_chars and timeout must be positive")
        self.max_chars, self.timeout, self._client = max_chars, timeout, client
        self.corpus = SourceArchive()
        self._hits: dict[str, SearchHit] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.receipts: list[dict] = []

    def duration(self, kind: str, op_id: str, attempt: int) -> None:
        return None  # TracingClient measures the actual network operation.

    def _headers(self):
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _post(
        self, endpoint: str, payload: dict, op_id: str, attempt: int
    ) -> dict:
        import httpx

        async def send(client):
            return await client.post(
                f"{self.base_url}/{endpoint}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )

        try:
            if self._client is None:
                async with httpx.AsyncClient() as client:
                    response = await send(client)
            else:
                response = await send(self._client)
        except httpx.RequestError as error:
            raise TransientToolError(
                f"{self.name} {endpoint}: {type(error).__name__}"
            ) from error
        code = response.status_code
        if code == 429 or code >= 500:
            raise TransientToolError(f"{self.name} {endpoint}: HTTP {code}")
        if not 200 <= code < 300:
            raise PermanentToolError(
                f"{self.name} {endpoint}: HTTP {code}; check the API key and account"
            )
        try:
            data = response.json()
        except ValueError as error:
            raise PermanentToolError(
                f"{self.name} {endpoint}: response was not JSON"
            ) from error
        if not isinstance(data, dict):
            raise PermanentToolError(
                f"{self.name} {endpoint}: response was not an object"
            )
        self.receipts.append(
            {
                "op_id": op_id,
                "attempt": attempt,
                "provider": self.name,
                "endpoint": endpoint,
                "request_id": data.get("requestId")
                or data.get("request_id")
                or data.get("id"),
                "cost_dollars": data.get("costDollars"),
                "usage": data.get("usage"),
            }
        )
        return data

    def _results(self, data):
        if not isinstance(data.get("results"), list):
            raise PermanentToolError(f"{self.name}: response has no results list")
        return data["results"]

    def _remember_hits(self, results, k):
        hits, seen = [], set()
        for result in results:
            if not isinstance(result, dict):
                continue
            url = result.get("url", "")
            if (
                not isinstance(url, str)
                or not url.startswith(("https://", "http://"))
                or url in seen
            ):
                continue
            seen.add(url)
            doc_id = "web-" + hashlib.sha256(url.encode()).hexdigest()[:20]
            hit = SearchHit(
                doc_id,
                str(result.get("title") or url),
                url,
                "",
                float(result.get("score") or 0),
                "",
            )
            self._hits[doc_id] = hit
            hits.append(hit)
        return hits[:k]

    def _save_text(self, hit, text, title=None):
        if not isinstance(text, str) or not text.strip():
            raise PermanentToolError(
                f"{self.name} fetch: no source text returned for {hit.uri}"
            )
        text = text[: self.max_chars]
        sha = hashlib.sha256(text.encode()).hexdigest()
        doc = Doc(
            hit.doc_id,
            "web",
            str(title or hit.title),
            hit.uri,
            datetime.now(timezone.utc).isoformat(),
            sha,
            0,
            hit.uri,
            text,
        )
        self.corpus.add(doc)
        return doc

    async def fetch(self, doc_id: str, *, op_id: str, attempt: int = 1) -> Doc:
        if doc_id not in self._hits:
            raise PermanentToolError(
                f"{self.name} fetch: document was not returned by search"
            )
        async with self._locks.setdefault(doc_id, asyncio.Lock()):
            if self.corpus.has(doc_id):
                return self.corpus.doc(doc_id)
            return await self._fetch_source(self._hits[doc_id], op_id, attempt)


class ExaSearch(_WebSearch):
    """Retrieve pages with Exa Search and Contents; set EXA_API_KEY.

    Each URL is fetched once per adapter instance. Create a new instance
    when a new run should retrieve fresh versions of the sources.
    """

    name, key_env, base_url = "exa", "EXA_API_KEY", "https://api.exa.ai"

    def _headers(self):
        return {"x-api-key": self._api_key}

    async def search(
        self, query: str, *, op_id: str, attempt: int = 1, k: int = 5
    ) -> list[SearchHit]:
        if not 1 <= k <= 100:
            raise ValueError("Exa search k must be in 1..100")
        data = await self._post(
            "search", {"query": query, "numResults": k, "type": "auto"}, op_id, attempt
        )
        return self._remember_hits(self._results(data), k)

    async def _fetch_source(self, hit, op_id, attempt):
        data = await self._post(
            "contents", {"urls": [hit.uri], "text": True}, op_id, attempt
        )
        results = self._results(data)
        # A single-URL request may return the page's canonical URL after a redirect.
        result = (
            results[0] if len(results) == 1 and isinstance(results[0], dict) else {}
        )
        return self._save_text(hit, result.get("text"), result.get("title"))


class TavilySearch(_WebSearch):
    """Retrieve pages with Tavily Search and Extract; set TAVILY_API_KEY."""

    name, key_env, base_url = "tavily", "TAVILY_API_KEY", "https://api.tavily.com"

    async def search(
        self, query: str, *, op_id: str, attempt: int = 1, k: int = 5
    ) -> list[SearchHit]:
        if not 1 <= k <= 20:
            raise ValueError("Tavily search k must be in 1..20")
        data = await self._post(
            "search",
            {
                "query": query,
                "max_results": k,
                "search_depth": "basic",
                "include_answer": False,
                "include_usage": True,
            },
            op_id,
            attempt,
        )
        return self._remember_hits(self._results(data), k)

    async def _fetch_source(self, hit, op_id, attempt):
        data = await self._post(
            "extract",
            {
                "urls": [hit.uri],
                "format": "text",
                "extract_depth": "basic",
                "include_usage": True,
            },
            op_id,
            attempt,
        )
        results = self._results(data)
        result = (
            results[0] if len(results) == 1 and isinstance(results[0], dict) else {}
        )
        return self._save_text(hit, result.get("raw_content"))


class PerplexitySearch(_WebSearch):
    """Use Perplexity's Search API and its extracted page text; set PERPLEXITY_API_KEY.

    Search includes retrieval. Fetch returns that saved excerpt, which can
    be shorter than the full page. It never uses a generated Sonar answer.
    """

    name, key_env, base_url = (
        "perplexity",
        "PERPLEXITY_API_KEY",
        "https://api.perplexity.ai",
    )

    async def search(
        self, query: str, *, op_id: str, attempt: int = 1, k: int = 5
    ) -> list[SearchHit]:
        if not 1 <= k <= 20:
            raise ValueError("Perplexity search k must be in 1..20")
        data = await self._post(
            "search",
            {
                "query": query,
                "max_results": k,
                "max_tokens_per_page": 4096,
                "max_tokens": k * 4096,
            },
            op_id,
            attempt,
        )
        results = self._results(data)
        hits = self._remember_hits(results, k)
        by_url = {r.get("url"): r for r in reversed(results) if isinstance(r, dict)}
        for hit in hits:
            text = by_url[hit.uri].get("snippet")
            # Keep the first source version across parallel searches in this run.
            if (
                not self.corpus.has(hit.doc_id)
                and isinstance(text, str)
                and text.strip()
            ):
                self._save_text(hit, text)
        return hits

    async def _fetch_source(self, hit, op_id, attempt):
        raise PermanentToolError(
            f"Perplexity search returned no source excerpt for {hit.uri}"
        )


class OpenAISearch(_WebSearch):
    """Discover URLs with OpenAI web search, then fetch HTML or text pages directly.

    Set OPENAI_API_KEY and AGENTKV_SEARCH_MODEL to a Responses model that
    supports web_search. Search consumes model tokens as well as tool calls;
    their usage is recorded separately from the graph's Bedrock LLM calls.
    Generated answers are discarded. Citation text comes from fetched pages.
    """

    name, key_env, base_url = "openai", "OPENAI_API_KEY", "https://api.openai.com/v1"

    def __init__(self, api_key=None, *, model=None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.model = model or os.environ.get("AGENTKV_SEARCH_MODEL", "")
        if not self.model:
            raise ValueError(
                "set AGENTKV_SEARCH_MODEL to an OpenAI model that supports web_search"
            )

    async def search(
        self, query: str, *, op_id: str, attempt: int = 1, k: int = 5
    ) -> list[SearchHit]:
        if not 1 <= k <= 20:
            raise ValueError("OpenAI search k must be in 1..20")
        data = await self._post(
            "responses",
            {
                "model": self.model,
                "store": False,
                "instructions": "Search for sources for the user's query. Return a short list of source URLs.",
                "input": query,
                "tools": [{"type": "web_search"}],
                "tool_choice": "required",
                "max_tool_calls": 1,
                "max_output_tokens": 1000,
                "include": ["web_search_call.action.sources"],
            },
            op_id,
            attempt,
        )
        receipt = self.receipts[-1]
        receipt["model"] = self.model
        output = data.get("output")
        if not isinstance(output, list):
            raise PermanentToolError("OpenAI search: response has no output list")
        calls = [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") == "web_search_call"
        ]
        receipt["web_search_calls"] = len(calls)
        results = []
        for call in calls:
            results.extend(call.get("action", {}).get("sources", []))
        # Citation annotations sometimes have titles absent from action.sources.
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content", []):
                    results.extend(
                        a
                        for a in part.get("annotations", [])
                        if a.get("type") == "url_citation"
                    )
        hits = self._remember_hits(results, k)
        if not calls or not hits:
            raise PermanentToolError(
                "OpenAI search: no web sources returned; check the model and token limit"
            )
        return hits

    async def _fetch_source(self, hit, op_id, attempt):
        from .web_pages import fetch_page_text

        text = await fetch_page_text(hit.uri, timeout=self.timeout, client=self._client)
        return self._save_text(hit, text)


def make_search(provider: str, **kwargs) -> SearchBackend:
    """Select local, exa, tavily, perplexity, or openai retrieval for ResearchAgent."""
    providers = {
        "local": LocalSearch,
        "exa": ExaSearch,
        "tavily": TavilySearch,
        "perplexity": PerplexitySearch,
        "openai": OpenAISearch,
    }
    try:
        cls = providers[provider.lower()]
    except KeyError:
        raise ValueError(
            f"unknown search provider {provider!r}; choose from {', '.join(providers)}"
        ) from None
    return cls(**kwargs)
