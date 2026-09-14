from ai_agent.core.knowledge import KnowledgeStatus
from ai_agent.core.knowledge_store import KnowledgeStore
from ai_agent.core.learning import LearningEngine
from ai_agent.core.researcher import ResearchDocument


def test_knowledge_store_round_trip(tmp_path):
    engine = LearningEngine()
    document = ResearchDocument(
        uri="https://example.com/source",
        content="source",
        content_hash="hash",
        retrieved_at="2026-01-01T00:00:00+00:00",
    )
    item = engine.observe("a claim", document)
    engine.verify(item, ["evidence"])

    store = KnowledgeStore(tmp_path / "knowledge.json")
    store.save(engine.knowledge)
    loaded = store.load()

    assert loaded[0].statement == "a claim"
    assert loaded[0].status is KnowledgeStatus.VERIFIED
    assert loaded[0].sources[0].content_hash == "hash"
