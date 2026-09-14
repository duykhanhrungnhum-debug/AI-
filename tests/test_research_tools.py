from ai_agent.core.research_tools import InternetResearchTools
from ai_agent.core.researcher import ResearchDocument
from ai_agent.core.search import SearchResult, StaticSearchProvider
from ai_agent.core.tool_executor import ToolExecutor
from ai_agent.core.model import ModelResponse
from ai_agent.core.model_agent import ModelDecision


class FakeResearcher:
    def fetch(self, uri: str) -> ResearchDocument:
        content = f"Source confirms the target fact at {uri}."
        import hashlib
        return ResearchDocument(
            uri=uri,
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            retrieved_at="2026-09-14T00:00:00+00:00",
            content_type="text/plain",
        )


def decision(text: str) -> ModelDecision:
    return ModelDecision(
        "research task",
        "research-001",
        "research",
        ModelResponse(text, "fake", "fake-model"),
    )


def test_research_tools_register_and_execute_selected():
    provider = StaticSearchProvider([
        SearchResult("https://example.com/a", "A", "target fact"),
        SearchResult("https://example.com/b", "B", "target fact"),
    ])
    research = InternetResearchTools(provider, FakeResearcher(), max_sources=2)
    executor = ToolExecutor()
    research.register(executor)

    result = executor.execute_selected(decision(
        '{"tool":"web_research","query":"target fact","max_sources":2}'
    ))

    assert result.success is True
    assert "corroborated" in result.outcome
    assert "https://example.com/a" in result.evidence


def test_research_tool_rejects_malformed_model_request():
    provider = StaticSearchProvider([])
    research = InternetResearchTools(provider, FakeResearcher())
    result = research.execute_json("not-json")
    assert result.success is False


def test_tool_router_rejects_unknown_tool():
    executor = ToolExecutor()
    research = InternetResearchTools(StaticSearchProvider([]), FakeResearcher())
    research.register(executor)
    result = executor.execute_selected(decision('{"tool":"shell"}'))
    assert result.success is False
    assert "Unknown tool" in result.outcome
