from ai_agent.core.autonomous_learning import AutonomousLearningCoordinator
from ai_agent.core.research_plan import ResearchPlanner
from ai_agent.core.researcher import ResearchDocument
from ai_agent.core.search import SearchResult, StaticSearchProvider


class FakeResearcher:
    def fetch(self, uri):
        return ResearchDocument(
            uri=uri,
            content="evidence-backed learning material",
            content_hash="hash-1",
            retrieved_at="2026-01-01T00:00:00+00:00",
            content_type="text/plain",
        )


def test_coordinator_connects_plan_research_and_verify():
    coordinator = AutonomousLearningCoordinator(researcher=FakeResearcher())
    session = coordinator.run_once(
        "learn autonomous planning",
        "https://example.test/planning",
        ["The retrieved material supports the learning objective."],
    )

    assert len(session.attempts) == 1
    assert session.attempts[0].status == "verified"
    assert session.completed is True
    assert coordinator.learner.verified()[0].status.value == "verified"


def test_coordinator_does_not_loop_when_no_gap_exists():
    coordinator = AutonomousLearningCoordinator(researcher=FakeResearcher())
    existing = coordinator.learner.observe(
        "autonomous planning is a verified capability",
        coordinator.researcher.fetch("https://example.test/existing"),
    )
    coordinator.learner.verify(existing, ["existing evidence"])

    session = coordinator.run_once(
        "autonomous planning",
        "https://example.test/planning",
        ["new evidence"],
    )

    assert session.attempts[0].status == "no_gap"
    assert len(coordinator.learner.knowledge) == 1


def test_coordinator_discovers_and_researches_multiple_sources_without_auto_verifying():
    provider = StaticSearchProvider(
        [
            SearchResult("https://example.test/a", title="A"),
            SearchResult("https://example.test/a", title="A duplicate"),
            SearchResult("https://example.test/b", title="B"),
        ]
    )
    coordinator = AutonomousLearningCoordinator(
        research_planner=ResearchPlanner(provider, max_sources=2),
        researcher=FakeResearcher(),
    )

    session = coordinator.run_once("learn autonomous planning")

    assert [attempt.status for attempt in session.attempts] == ["proposed", "proposed"]
    assert [attempt.source_uri for attempt in session.attempts] == [
        "https://example.test/a",
        "https://example.test/b",
    ]
    assert coordinator.learner.verified() == []
    assert len(coordinator.learner.knowledge) == 2
