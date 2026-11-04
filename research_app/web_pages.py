"""Fetch public HTML/text sources for the OpenAI search adapter."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from .corpus import PermanentToolError, TransientToolError


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self.hidden.append(tag)
        elif not self.hidden and tag in {
            "p",
            "div",
            "br",
            "li",
            "h1",
            "h2",
            "h3",
            "tr",
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()
        if not self.hidden and tag in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


async def _check_public_url(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 80, 443}
    ):
        raise PermanentToolError("page fetch: expected a public HTTP(S) URL")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as error:
        raise TransientToolError("page fetch: hostname lookup failed") from error
    if not addresses or any(
        not ipaddress.ip_address(row[4][0]).is_global for row in addresses
    ):
        raise PermanentToolError("page fetch: source must resolve to a public address")


async def fetch_page_text(url: str, *, timeout: float = 30, client=None) -> str:
    """Extract text from a public page, with a 2 MiB limit and at most five redirects.

    This fetcher does not run JavaScript or read PDFs. Such pages fail as tool
    operations, so the agent can use other sources instead of fabricated text.
    """
    import httpx

    async def fetch(http):
        target = url
        for _ in range(6):
            await _check_public_url(target)
            async with http.stream(
                "GET", target, timeout=timeout, follow_redirects=False
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise PermanentToolError(
                            "page fetch: redirect has no destination"
                        )
                    target = urljoin(target, location)
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    raise TransientToolError(f"page fetch: HTTP {response.status_code}")
                if not 200 <= response.status_code < 300:
                    raise PermanentToolError(f"page fetch: HTTP {response.status_code}")
                kind = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if kind not in {"text/html", "application/xhtml+xml", "text/plain"}:
                    raise PermanentToolError(
                        "page fetch: only HTML and plain-text sources are supported"
                    )
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 2 * 1024 * 1024:
                        raise PermanentToolError(
                            "page fetch: source exceeds the 2 MiB download limit"
                        )
                    chunks.append(chunk)
                text = b"".join(chunks).decode(
                    response.encoding or "utf-8", errors="replace"
                )
                if kind != "text/plain":
                    parser = _PageText()
                    parser.feed(text)
                    text = "\n".join(
                        line
                        for raw in "".join(parser.parts).splitlines()
                        if (line := " ".join(raw.split()))
                    )
                if not text.strip():
                    raise PermanentToolError(
                        "page fetch: source contained no readable text"
                    )
                return text
        raise PermanentToolError("page fetch: too many redirects")

    try:
        if client is not None:
            return await fetch(client)
        async with httpx.AsyncClient() as http:
            return await fetch(http)
    except httpx.RequestError as error:
        raise TransientToolError(f"page fetch: {type(error).__name__}") from error
