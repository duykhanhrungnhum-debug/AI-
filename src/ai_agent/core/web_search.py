"""Concrete public web search provider for deployments that need keyless discovery."""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urljoin, urlparse
from urllib.request import Request, urlopen

from .search import SearchProvider, SearchResult


class _DuckParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._href = ""
        self._title = ""
        self._capture = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "result__a" in attrs.get("class", ""):
            self._href = attrs.get("href", "")
            self._title = ""
            self._capture = True

    def handle_data(self, data):
        if self._capture:
            self._title += data

    def handle_endtag(self, tag):
        if tag == "a" and self._capture:
            href = self._href
            parsed = urlparse(href)
            if parsed.path.startswith("/l/"):
                target = parse_qs(parsed.query).get("uddg", [""])[0]
                href = target
            if href.startswith("http"):
                self.results.append(SearchResult(href, self._title.strip()))
            self._href = ""
            self._title = ""
            self._capture = False


class DuckDuckGoSearchProvider(SearchProvider):
    """Query DuckDuckGo's public HTML endpoint and return candidate URLs.

    Search results are discovery only; callers must fetch and verify sources.
    """

    def __init__(self, *, timeout: float = 15.0, user_agent: str = "AI-Agent-Search/0.1") -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.user_agent = user_agent

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        uri = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        request = Request(uri, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
        parser = _DuckParser()
        parser.feed(html)
        unique: list[SearchResult] = []
        seen: set[str] = set()
        for result in parser.results:
            if result.url not in seen:
                seen.add(result.url)
                unique.append(result)
            if len(unique) >= limit:
                break
        return unique
