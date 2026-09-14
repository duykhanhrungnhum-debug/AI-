from hashlib import sha256

from ai_agent.core.autonomous_learning import AutonomousLearningCoordinator
from ai_agent.core.experience import ExperienceStore
from ai_agent.core.learning import LearningEngine
from ai_agent.core.research_plan import ResearchPlanner
from ai_agent.core.search import SearchResult, StaticSearchProvider
from ai_agent.core.researcher import ResearchDocument


class FakeResearcher:
    def fetch(self, uri: str) -> ResearchDocument:
        content = "learning systems require evidence"
        return ResearchDocument(
            uri=uri,
            content=content,
            content_hash=sha256(content.encode()).hexdigest(),
            retrieved_at="2026-09-14T00:00:00+00:00",
        )


def test_successful_autonomous_learning_records_experience():
    store = ExperienceStore()
    provider = StaticSearchProvider([SearchResult("https://example.test/a", "A", "evidence")])
    coordinator = AutonomousLearningCoordinator(
        research_planner=ResearchPlanner(provider, max_sources=1),
        researcher=FakeResearcher(),
        learner=LearningEngine(),
        experience_store=store,
    )

    session = coordinator.run_once("learning systems require evidence", evidence=["source supports goal"])

    assert session.completed
    experiences = store.all()
    assert len(experiences) == 1
    assert experiences[0].success is True


def test_failed_research_is_remembered():
    class BrokenResearcher:
        def fetch(self, uri: str):
            raise RuntimeError("network unavailable")

    store = ExperienceStore()
    provider = StaticSearchProvider([SearchResult("https://example.test/a", "A", "evidence")])
    coordinator = AutonomousLearningCoordinator(
        research_planner=ResearchPlanner(provider, max_sources=1),
        researcher=BrokenResearcher(),
        experience_store=store,
    )

    session = coordinator.run_once("learning systems require evidence")

    assert not session.completed
    assert len(store.all()) == 1
    assert store.all()[0].success is False
    assert "network unavailable" in store.all()[0].lesson
