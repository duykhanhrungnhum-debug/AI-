"""Verification-first trading research skill.

This module deliberately separates market research from trade execution. It can
collect and score evidence for training/paper-trading, but it never submits a
broker order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from urllib.parse import urlparse
from typing import Iterable

from .invariants import assert_core_invariants


class SourceTier(IntEnum):
    PRIMARY = 1
    HIGH_QUALITY_NEWS = 2
    SECONDARY = 3
    UNKNOWN = 4


PRIMARY_DOMAINS = {
    "eia.gov", "opec.org", "iea.org", "cftc.gov", "cmegroup.com",
    "ice.com", "federalreserve.gov", "fred.stlouisfed.org", "noaa.gov",
    "nhc.noaa.gov", "treasury.gov",
}
NEWS_DOMAINS = {"reuters.com", "apnews.com", "bloomberg.com", "ft.com", "wsj.com"}

# Stable public hubs are fetched every cycle so training does not depend on one
# search engine's HTML layout. Search remains an enrichment layer.
OIL_SOURCE_HUBS: tuple[tuple[str, str, str], ...] = (
    ("supply_demand", "EIA Weekly Petroleum Status", "https://www.eia.gov/petroleum/weekly/"),
    ("supply_demand", "OPEC", "https://www.opec.org/"),
    ("supply_demand", "IEA Oil Market Report", "https://www.iea.org/reports/oil-market-report"),
    ("market", "CFTC Commitments of Traders", "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"),
    ("market", "CME WTI Crude Oil", "https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.html"),
    ("macro", "Federal Reserve News", "https://www.federalreserve.gov/newsevents.htm"),
    ("weather", "National Hurricane Center", "https://www.nhc.noaa.gov/"),
)

OIL_RESEARCH_QUERIES: dict[str, tuple[str, ...]] = {
    "supply_demand": (
        "WTI crude oil EIA weekly petroleum status report inventories production",
        "OPEC latest oil production policy monthly oil market report",
        "IEA latest oil market report demand supply",
    ),
    "market": (
        "WTI Brent futures curve spread CME ICE crude oil latest",
        "CFTC crude oil commitments of traders managed money latest",
    ),
    "macro": (
        "Federal Reserve latest rates dollar financial conditions oil demand",
        "US Treasury yields dollar latest macro crude oil",
    ),
    "geopolitics": (
        "latest oil sanctions shipping Strait of Hormuz Red Sea Reuters",
        "latest crude oil pipeline refinery disruption Reuters",
    ),
    "weather": (
        "NOAA NHC Gulf of Mexico hurricane oil production latest",
    ),
}


def _host(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.lower().removeprefix("www.")


def _domain_matches(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def source_tier(url: str) -> SourceTier:
    host = _host(url)
    if _domain_matches(host, PRIMARY_DOMAINS):
        return SourceTier.PRIMARY
    if _domain_matches(host, NEWS_DOMAINS):
        return SourceTier.HIGH_QUALITY_NEWS
    if host:
        return SourceTier.SECONDARY
    return SourceTier.UNKNOWN


@dataclass(frozen=True)
class TradingEvidence:
    category: str
    title: str
    url: str
    content_hash: str
    retrieved_at: str
    tier: int
    source_host: str

    @classmethod
    def from_retrieval(
        cls, *, category: str, title: str, url: str,
        content_hash: str, retrieved_at: str
    ) -> "TradingEvidence":
        if not category.strip() or not url.startswith(("http://", "https://")):
            raise ValueError("category and absolute HTTP(S) url are required")
        if not content_hash.strip() or not retrieved_at.strip():
            raise ValueError("retrieval hash and timestamp are required")
        return cls(
            category=category,
            title=title.strip(),
            url=url,
            content_hash=content_hash,
            retrieved_at=retrieved_at,
            tier=int(source_tier(url)),
            source_host=_host(url),
        )


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reasons: tuple[str, ...]
    independent_domains: int
    primary_sources: int
    categories: tuple[str, ...]


@dataclass
class TradingResearchEpisode:
    asset: str = "WTI/Brent crude oil"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evidence: list[TradingEvidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    verification: VerificationResult | None = None
    mode: str = "research-training-only"

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.verification is not None:
            payload["verification"] = asdict(self.verification)
        return payload


class TradingEvidenceVerifier:
    """Require multi-source, multi-factor evidence before accepting a learning episode."""

    def verify(self, evidence: Iterable[TradingEvidence]) -> VerificationResult:
        assert_core_invariants()
        items = list(evidence)
        hosts = {item.source_host for item in items if item.source_host}
        categories = {item.category for item in items}
        primary = sum(item.tier == int(SourceTier.PRIMARY) for item in items)

        reasons: list[str] = []
        if len(hosts) < 3:
            reasons.append("Need at least 3 independent source domains.")
        if primary < 1:
            reasons.append("Need at least 1 primary/official source.")
        if "supply_demand" not in categories:
            reasons.append("Supply/demand evidence is required.")
        if "market" not in categories:
            reasons.append("Market/positioning evidence is required.")
        if not ({"macro", "geopolitics", "weather"} & categories):
            reasons.append("At least one external driver category is required.")
        if any(not item.content_hash for item in items):
            reasons.append("Every accepted source must have retrieval evidence.")

        return VerificationResult(
            passed=not reasons,
            reasons=tuple(reasons) if reasons else ("Evidence gate passed.",),
            independent_domains=len(hosts),
            primary_sources=primary,
            categories=tuple(sorted(categories)),
        )


def source_hubs() -> list[tuple[str, str, str]]:
    assert_core_invariants()
    return list(OIL_SOURCE_HUBS)


def research_plan() -> list[tuple[str, str]]:
    """Return deterministic oil research goals for one background cycle."""
    assert_core_invariants()
    return [(category, query) for category, queries in OIL_RESEARCH_QUERIES.items() for query in queries]
