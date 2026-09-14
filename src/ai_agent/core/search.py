"""Search abstractions and concrete Internet discovery providers."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen
import json


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""


class SearchProvider:
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class StaticSearchProvider(SearchProvider):
    def __init__(self, results: list[SearchResult]):
        self.results = list(results)

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        return self.results[:limit]


class JsonSearchProvider(SearchProvider):
    """Call a deployment-configured HTTP endpoint returning JSON results."""

    def __init__(self, endpoint: str, *, timeout: float = 15.0, user_agent: str = "AI-Agent-Search/0.1"):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("search endpoint must use HTTP(S)")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoint = endpoint
        self.timeout = timeout
        self.user_agent = user_agent

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        separator = "&" if "?" in self.endpoint else "?"
        uri = f"{self.endpoint}{separator}q={quote(query)}&limit={limit}"
        request = Request(uri, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("search response must contain a result list")
        return [SearchResult(str(row["url"]), str(row.get("title", "")), str(row.get("snippet", "")))
                for row in rows[:limit] if isinstance(row, dict) and row.get("url")]


class _DuckParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.results: list[SearchResult] = []
        self._current_url = ""
        self._title: list[str] = []
        self._capture = False

    def handle_starttag(self, tag: str, attrs):
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._current_url = urljoin(self.base_url, attributes.get("href", ""))
            self._title = []
            self._capture = True

    def handle_data(self, data: str):
        if self._capture:
            self._title.append(data)

    def handle_endtag(self, tag: str):
        if tag == "a" and self._capture:
            title = " ".join("".join(self._title).split())
            if self._current_url:
                self.results.append(SearchResult(self._current_url, title))
            self._current_url = ""
            self._title = []
            self._capture = False


class DuckDuckGoSearchProvider(SearchProvider):
    """Concrete public-web search provider using DuckDuckGo's HTML results."""

    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, *, timeout: float = 15.0, user_agent: str = "AI-Agent-Search/0.1"):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.user_agent = user_agent

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        request = Request(
            f"{self.endpoint}?q={quote(query)}",
            headers={"User-Agent": self.user_agent, "Accept": "text/html"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            html = response.read(2_000_000).decode("utf-8", errors="replace")
        parser = _DuckParser(self.endpoint)
        parser.feed(html)
        return parser.results[:limit]
