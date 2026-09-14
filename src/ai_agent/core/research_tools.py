"""Explicit Internet research tools for the agent runtime.

These tools expose search, fetch, comparison, conflict analysis, and knowledge
capture through the same registered-tool boundary used by other agent actions.
Web content is provenance-bearing input, never an automatic truth claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

from .conflict_resolution import ConflictResolver
from .invariants import assert_core_invariants
from .learning import LearningEngine
from .runtime_types import ActionExecution
from .research_plan import ResearchPlanner
from .researcher import InternetResearcher
from .search import SearchProvider
from .source_comparison import SourceComparator

if TYPE_CHECKING:
    from .knowledge_store import KnowledgeStore
    from .tool_executor import ToolExecutor


@dataclass(frozen=True)
class ResearchRequest:
    """Strict, JSON-serializable request selected by the model/host."""

    tool: str
    query: str = ""
    url: str = ""
    max_sources: int = 3


class InternetResearchTools:
    """Build registered tools backed by the public-web research stack."""

    TOOL_NAMES = ("web_search", "web_fetch", "web_research")

    def __init__(self, search_provider: SearchProvider, researcher: InternetResearcher | None = None, learner: LearningEngine | None = None, knowledge_store: KnowledgeStore | None = None, *, max_sources: int = 3) -> None:
        if max_sources <= 0:
            raise ValueError("max_sources must be positive")
        self.search_provider = search_provider
        self.researcher = researcher or InternetResearcher()
        self.learner = learner or LearningEngine()
        self.knowledge_store = knowledge_store
        self.max_sources = max_sources
        self.planner = ResearchPlanner(search_provider, max_sources=max_sources)
        self.comparator = SourceComparator(min_sources=2)
        self.conflict_resolver = ConflictResolver()

    @staticmethod
    def parse_request(text: str) -> ResearchRequest:
        """Parse only the small structured tool contract; reject free-form calls."""
        assert_core_invariants()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("tool request must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("tool request must be a JSON object")
        tool = payload.get("tool")
        if not isinstance(tool, str) or tool not in InternetResearchTools.TOOL_NAMES:
            raise ValueError("unsupported research tool")
        query = payload.get("query", "")
        url = payload.get("url", "")
        max_sources = payload.get("max_sources", 3)
        if not isinstance(query, str) or not isinstance(url, str):
            raise ValueError("query and url must be strings")
        if not isinstance(max_sources, int) or isinstance(max_sources, bool) or max_sources <= 0:
            raise ValueError("max_sources must be a positive integer")
        return ResearchRequest(tool, query.strip(), url.strip(), min(max_sources, 10))

    def _persist_knowledge(self) -> None:
        if self.knowledge_store is not None:
            self.knowledge_store.save(self.learner.knowledge)

    def execute(self, request: ResearchRequest) -> ActionExecution:
        """Execute one explicit research operation and return provenance evidence."""
        assert_core_invariants()
        if request.tool == "web_search":
            if not request.query:
                raise ValueError("web_search requires query")
            results = self.search_provider.search(request.query, limit=request.max_sources)
            if not results:
                return ActionExecution(False, "Web search returned no results.")
            evidence = tuple(result.url for result in results)
            outcome = json.dumps([{"url": r.url, "title": r.title, "snippet": r.snippet} for r in results], ensure_ascii=False)
            return ActionExecution(True, outcome, evidence)

        if request.tool == "web_fetch":
            if not request.url:
                raise ValueError("web_fetch requires url")
            document = self.researcher.fetch(request.url)
            return ActionExecution(True, document.content, (f"source:{document.uri}", f"sha256:{document.content_hash}"))

        if not request.query:
            raise ValueError("web_research requires query")
        plan = self.planner.plan(request.query)
        documents = []
        failures = []
        for source in plan.sources:
            try:
                documents.append(self.researcher.fetch(source.url))
            except Exception as exc:
                failures.append(f"{source.url}: {exc}")
        if not documents:
            return ActionExecution(False, "No research sources could be fetched.", tuple(failures))

        comparison = self.comparator.compare(request.query, documents)
        conflict = self.conflict_resolver.analyze(request.query, documents)
        knowledge_ids = []
        knowledge_statuses = []
        for document in documents:
            item = self.learner.observe(request.query, document, kind="lesson", tags=["internet-research", "web-research"])
            if conflict.conflicted:
                self.learner.conflict(item, conflict.reason)
            elif comparison.corroborated:
                self.learner.verify(item, comparison.evidence)
            knowledge_ids.append(item.id)
            knowledge_statuses.append(item.status.value)
        self._persist_knowledge()

        evidence = tuple(doc.uri for doc in documents) + tuple(f"sha256:{doc.content_hash}" for doc in documents)
        outcome = json.dumps({
            "sources": [doc.uri for doc in documents],
            "corroborated": comparison.corroborated,
            "conflicted": conflict.conflicted,
            "reason": conflict.reason,
            "fetch_failures": failures,
            "knowledge_ids": knowledge_ids,
            "knowledge_statuses": knowledge_statuses,
        }, ensure_ascii=False)
        return ActionExecution(True, outcome, evidence)

    def execute_json(self, text: str, *, expected_tool: str | None = None) -> ActionExecution:
        """Parse and execute a strictly bounded research request."""
        try:
            request = self.parse_request(text)
            if expected_tool is not None and request.tool != expected_tool:
                return ActionExecution(False, f"Selected tool mismatch: expected {expected_tool}.")
            return self.execute(request)
        except Exception as exc:
            return ActionExecution(False, f"Research request failed: {exc}")

    def register(self, executor: ToolExecutor) -> None:
        """Register all research operations with the shared ToolExecutor."""
        from .tool_executor import Tool
        for name in self.TOOL_NAMES:
            executor.register(Tool(name, lambda decision, selected=name: self.execute_json(decision.response.text, expected_tool=selected)))
