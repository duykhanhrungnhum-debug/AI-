from ai_agent.core.research_plan import ResearchPlanner
from ai_agent.core.search import SearchResult, StaticSearchProvider


def test_research_planner_deduplicates_and_bounds_sources():
    provider = StaticSearchProvider(
        [
            SearchResult("https://example.test/a"),
            SearchResult("https://example.test/a"),
            SearchResult("https://example.test/b"),
        ]
    )
    plan = ResearchPlanner(provider, max_sources=2).plan("learn verification")
    assert plan.query == "learn verification"
    assert [item.url for item in plan.sources] == [
        "https://example.test/a",
        "https://example.test/b",
    ]
