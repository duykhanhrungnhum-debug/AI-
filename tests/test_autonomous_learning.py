from ai_agent.core.autonomous_learning import AutonomousLearningCoordinator
from ai_agent.core.researcher import ResearchDocument


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
