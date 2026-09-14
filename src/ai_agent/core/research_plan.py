"""Turn a learning objective into a bounded list of candidate sources."""
from __future__ import annotations

from dataclasses import dataclass

from .invariants import assert_core_invariants
from .search import SearchProvider, SearchResult


@dataclass(frozen=True)
class ResearchPlan:
    query: str
    sources: list[SearchResult]


class ResearchPlanner:
    """Discover candidate sources without promoting any source to truth."""

    def __init__(self, provider: SearchProvider, *, max_sources: int = 5):
        if max_sources <= 0:
            raise ValueError("max_sources must be positive")
        self.provider = provider
        self.max_sources = max_sources

    def plan(self, objective: str) -> ResearchPlan:
        assert_core_invariants()
        objective = objective.strip()
        if not objective:
            raise ValueError("research objective must not be empty")
        results = self.provider.search(objective, limit=self.max_sources)
        unique: list[SearchResult] = []
        seen: set[str] = set()
        for result in results:
            if result.url not in seen:
                seen.add(result.url)
                unique.append(result)
        return ResearchPlan(query=objective, sources=unique[: self.max_sources])
