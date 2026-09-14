from ai_agent.core.autonomous_learning import AutonomousLearningCoordinator
from ai_agent.core.knowledge import KnowledgeStatus
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


class ConflictResearcher:
    def fetch(self, uri):
        content = (
            "autonomous planning is a useful capability"
            if uri.endswith("/support")
            else "autonomous planning is not a useful capability"
        )
        return ResearchDocument(
            uri=uri,
            content=content,
            content_hash=uri.rsplit("/", 1)[-1],
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
    session = coordinator.run_once("autonomous planning", "https://example.test/planning", ["new evidence"])
    assert session.attempts[0].status == "no_gap"
    assert len(coordinator.learner.knowledge) == 1


def test_coordinator_discovers_and_researches_multiple_sources_without_auto_verifying():
    provider = StaticSearchProvider([
        SearchResult("https://example.test/a", title="A"),
        SearchResult("https://example.test/a", title="A duplicate"),
        SearchResult("https://example.test/b", title="B"),
    ])
    coordinator = AutonomousLearningCoordinator(
        research_planner=ResearchPlanner(provider, max_sources=2), researcher=FakeResearcher())
    session = coordinator.run_once("learn autonomous planning")
    assert [attempt.status for attempt in session.attempts] == ["proposed", "proposed"]
    assert [attempt.source_uri for attempt in session.attempts] == [
        "https://example.test/a", "https://example.test/b"]
    assert coordinator.learner.verified() == []
    assert len(coordinator.learner.knowledge) == 2


def test_conflicting_sources_are_not_auto_verified():
    provider = StaticSearchProvider([
        SearchResult("https://example.test/support", title="support"),
        SearchResult("https://example.test/opposition", title="opposition"),
    ])
    coordinator = AutonomousLearningCoordinator(
        research_planner=ResearchPlanner(provider, max_sources=2), researcher=ConflictResearcher())
    session = coordinator.run_once("autonomous planning is a useful capability")
    assert session.conflict_analysis is not None
    assert session.conflict_analysis.conflicted is True
    assert [attempt.status for attempt in session.attempts] == ["conflicted", "conflicted"]
    assert session.completed is False
    assert coordinator.learner.verified() == []
    assert all(item.status is KnowledgeStatus.CONFLICTED for item in coordinator.learner.knowledge)


def test_long_running_loop_has_explicit_cycle_bound_and_survives_cycle_failure():
    coordinator = AutonomousLearningCoordinator(researcher=FakeResearcher())
    seen = []
    goals = iter(["first learning goal", "second learning goal", "third learning goal"])
    calls = {"count": 0}

    def run_once(goal, uri=None, evidence=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("transient failure")
        return coordinator.run_once(goal, "https://example.test/item", ["verified evidence"])

    coordinator.run_once = run_once
    cycles = coordinator.run_forever(lambda: next(goals), interval_seconds=0, max_cycles=3, on_session=seen.append)
    assert cycles == 3
    assert len(seen) == 3
    assert seen[1].attempts[0].status == "failed"
    assert any(not item.success for item in coordinator.experience_store.all())
