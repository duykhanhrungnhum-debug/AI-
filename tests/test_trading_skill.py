from ai_agent.core.trading_skill import (
    SourceTier,
    TradingEvidence,
    TradingEvidenceVerifier,
    source_tier,
)


def _e(category: str, url: str, n: int) -> TradingEvidence:
    return TradingEvidence.from_retrieval(
        category=category,
        title=f"source {n}",
        url=url,
        content_hash=f"hash-{n}",
        retrieved_at="2026-09-20T00:00:00+00:00",
    )


def test_source_hierarchy_prefers_official_and_high_quality_news():
    assert source_tier("https://www.eia.gov/petroleum/") == SourceTier.PRIMARY
    assert source_tier("https://www.reuters.com/world/") == SourceTier.HIGH_QUALITY_NEWS
    assert source_tier("https://example.com/oil") == SourceTier.SECONDARY


def test_verifier_rejects_chart_only_or_single_source_learning():
    result = TradingEvidenceVerifier().verify([
        _e("market", "https://www.cmegroup.com/markets/energy/crude-oil.html", 1),
    ])
    assert not result.passed
    assert result.independent_domains == 1


def test_verifier_accepts_multifactor_verified_research_episode():
    result = TradingEvidenceVerifier().verify([
        _e("supply_demand", "https://www.eia.gov/petroleum/", 1),
        _e("market", "https://www.cftc.gov/MarketReports/index.htm", 2),
        _e("geopolitics", "https://www.reuters.com/world/", 3),
    ])
    assert result.passed
    assert result.independent_domains == 3
    assert result.primary_sources >= 1
