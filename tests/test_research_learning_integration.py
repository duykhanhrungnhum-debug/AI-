import json

from ai_agent.core.knowledge import KnowledgeStatus
from ai_agent.core.learning import LearningEngine
from ai_agent.core.knowledge_store import KnowledgeStore
from ai_agent.core.model import ModelResponse
from ai_agent.core.model_agent import ModelAgent
from ai_agent.core.researcher import ResearchDocument
from ai_agent.core.research_tools import InternetResearchTools
from ai_agent.core.search import SearchResult, StaticSearchProvider
from ai_agent.core.tool_executor import ToolExecutor


class FakeResearcher:
    def __init__(self, documents):
        self.documents = documents

    def fetch(self, uri):
        return self.documents[uri]


def test_web_research_captures_and_verifies_corroborated_knowledge(tmp_path):
    docs = {
        "https://one.example": ResearchDocument("https://one.example", "Python was created by Guido van Rossum.", "hash-one", "2026-01-01T00:00:00+00:00"),
        "https://two.example": ResearchDocument("https://two.example", "Python was created by Guido van Rossum.", "hash-two", "2026-01-01T00:00:00+00:00"),
    }
    provider = StaticSearchProvider([
        SearchResult("https://one.example", "one"),
        SearchResult("https://two.example", "two"),
    ])
    learner = LearningEngine()
    store = KnowledgeStore(tmp_path / "knowledge.json")
    tools = InternetResearchTools(provider, FakeResearcher(docs), learner, store, max_sources=2)

    result = tools.execute_json(json.dumps({"tool": "web_research", "query": "Python created Guido van Rossum", "max_sources": 2}))

    assert result.success
    assert len(learner.verified()) == 2
    assert all(item.status is KnowledgeStatus.VERIFIED for item in learner.knowledge)
    assert store.load()[0].status is KnowledgeStatus.VERIFIED
    assert "knowledge_ids" in result.outcome


def test_registered_research_tool_is_selected_through_shared_boundary():
    provider = StaticSearchProvider([SearchResult("https://example.com", "example")])
    tools = InternetResearchTools(provider, FakeResearcher({}))
    executor = ToolExecutor()
    tools.register(executor)

    decision = ModelAgent(type("Model", (), {"generate": lambda self, prompt: ModelResponse(
        json.dumps({"tool": "web_search", "query": "example", "max_sources": 1}), "fake", "test"
    )})()).decide("Research", "step-1", "Find sources", available_tools=executor.names)

    result = executor.execute_selected(decision)

    assert result.success
    assert "example.com" in result.outcome


def test_research_tool_rejects_unregistered_tool_name():
    provider = StaticSearchProvider([])
    tools = InternetResearchTools(provider, FakeResearcher({}))
    result = tools.execute_json(json.dumps({"tool": "shell", "query": "rm -rf /"}))

    assert not result.success
    assert "unsupported research tool" in result.outcome
