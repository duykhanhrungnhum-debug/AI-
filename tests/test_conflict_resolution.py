from ai_agent.core.conflict_resolution import ConflictResolver
from ai_agent.core.researcher import ResearchDocument


def doc(uri, content, hash_value):
    return ResearchDocument(
        uri=uri,
        content=content,
        content_hash=hash_value,
        retrieved_at="2026-01-01T00:00:00+00:00",
        content_type="text/plain",
    )


def test_detects_opposing_lexical_signals():
    analysis = ConflictResolver().analyze(
        "python is safe",
        [
            doc("https://example.test/a", "python is safe and useful", "a"),
            doc("https://example.test/b", "python is not safe in this context", "b"),
        ],
    )
    assert analysis.conflicted is True
    assert analysis.supporting_sources == ["https://example.test/a"]
    assert analysis.opposing_sources == ["https://example.test/b"]


def test_does_not_claim_conflict_without_opposing_signal():
    analysis = ConflictResolver().analyze(
        "python is safe",
        [
            doc("https://example.test/a", "python is safe and useful", "a"),
            doc("https://example.test/b", "python is safe for this use", "b"),
        ],
    )
    assert analysis.conflicted is False
