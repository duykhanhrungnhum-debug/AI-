from ai_agent.core.knowledge import KnowledgeStatus
from ai_agent.core.learning import LearningEngine
from ai_agent.core.researcher import ResearchDocument
from ai_agent.core.curriculum import get_curriculum


def document():
    return ResearchDocument(
        uri="https://example.com/source",
        content="source content",
        content_hash="abc123",
        retrieved_at="2026-01-01T00:00:00+00:00",
        content_type="text/plain",
    )


def test_learning_starts_as_proposed():
    engine = LearningEngine()
    item = engine.observe("Python is a programming language.", document())
    assert item.status is KnowledgeStatus.PROPOSED
    assert item.sources[0].uri == "https://example.com/source"


def test_learning_requires_evidence_and_provenance():
    engine = LearningEngine()
    item = engine.observe("claim", document())
    try:
        engine.verify(item, [])
    except ValueError as exc:
        assert "evidence" in str(exc)
    else:
        raise AssertionError("verification without evidence must fail")


def test_verified_knowledge_is_explicit():
    engine = LearningEngine()
    item = engine.observe("claim", document())
    engine.verify(item, ["independent check passed"])
    assert item.status is KnowledgeStatus.VERIFIED
    assert engine.verified() == [item]


def test_curriculum_has_broad_capability_coverage():
    domains = {exercise.domain for exercise in get_curriculum()}
    assert {"reasoning", "programming", "internet_research", "verification", "recovery"} <= domains
