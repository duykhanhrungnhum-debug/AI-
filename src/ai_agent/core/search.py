"""Search abstractions for autonomous Internet research.

Search discovery is deliberately separated from document retrieval and
verification. A provider returns candidate URLs; the learning pipeline must
still fetch, inspect, and verify those sources before accepting knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen
import json


@dataclass(frozen=True)
class SearchResult:
    """A candidate source discovered by a search provider."""

    url: str
    title: str = ""
    snippet: str = ""


class SearchProvider:
    """Interface implemented by concrete search backends."""

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class StaticSearchProvider(SearchProvider):
    """Deterministic provider useful for tests and controlled deployments."""

    def __init__(self, results: list[SearchResult]):
        self.results = list(results)

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        return self.results[:limit]


class JsonSearchProvider(SearchProvider):
    """Call a configured HTTP endpoint returning a simple JSON result list.

    The endpoint is supplied by the deployment rather than hardcoding a search
    engine. It must accept ``?q=<query>&limit=<limit>`` and return either a
    list of objects or ``{"results": [...]}``, where each object contains
    ``url`` and optional ``title``/``snippet`` fields.
    """

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
        results: list[SearchResult] = []
        for row in rows[:limit]:
            if not isinstance(row, dict) or not row.get("url"):
                continue
            results.append(
                SearchResult(
                    url=str(row["url"]),
                    title=str(row.get("title", "")),
                    snippet=str(row.get("snippet", "")),
                )
            )
        return results
