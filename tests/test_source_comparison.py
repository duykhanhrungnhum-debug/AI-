from ai_agent.core.researcher import ResearchDocument
from ai_agent.core.source_comparison import SourceComparator


def doc(uri, text, digest):
    return ResearchDocument(uri, text, digest, "2026-01-01T00:00:00+00:00", "text/plain")


def test_source_comparator_requires_two_distinct_sources():
    comparator = SourceComparator()
    result = comparator.compare(
        "autonomous planning",
        [doc("https://a.test", "autonomous planning is useful", "a"),
         doc("https://b.test", "autonomous planning is useful", "b")],
    )
    assert result.corroborated is True
    assert result.distinct_content_count == 2
    assert len(result.evidence) == 2


def test_source_comparator_keeps_weak_evidence_unverified():
    result = SourceComparator().compare(
        "autonomous planning",
        [doc("https://a.test", "planning only", "a")],
    )
    assert result.corroborated is False
