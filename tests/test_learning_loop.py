from ai_agent.core.knowledge import KnowledgeStatus
from ai_agent.core.learning_loop import AutonomousLearningLoop
from ai_agent.core.researcher import ResearchDocument


class FakeResearcher:
    def fetch(self, uri):
        return ResearchDocument(uri, "test", "hash-1", "2026-01-01T00:00:00+00:00")


def test_learning_loop_keeps_retrieval_unverified():
    loop = AutonomousLearningLoop(researcher=FakeResearcher())
    result = loop.research_and_propose("Python is a programming language", "https://example.test/python")
    assert result.verified is False
    assert result.item.status is KnowledgeStatus.PROPOSED


def test_learning_loop_requires_evidence_to_promote():
    loop = AutonomousLearningLoop(researcher=FakeResearcher())
    result = loop.research_and_propose("Python is a programming language", "https://example.test/python")
    promoted = loop.verify_proposal(result.item, ["independent verification evidence"])
    assert promoted.verified is True
    assert promoted.item.status is KnowledgeStatus.VERIFIED
