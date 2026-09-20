"""Keyless current-news discovery for market research.

RSS items are discovery/context signals. They do not satisfy the authoritative
source verification gate by themselves.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class NewsSignal:
    category: str
    headline: str
    publisher: str
    published_at: str
    link: str
    feed_hash: str
    retrieved_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class GoogleNewsRSSProvider:
    endpoint = "https://news.google.com/rss/search"

    def __init__(self, *, timeout: float = 15.0, user_agent: str = "AI-Agent-News/0.1") -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.user_agent = user_agent

    def _parse(self, xml_bytes: bytes, *, category: str, limit: int) -> list[NewsSignal]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        root = ET.fromstring(xml_bytes)
        digest = sha256(xml_bytes).hexdigest()
        retrieved_at = datetime.now(timezone.utc).isoformat()
        signals: list[NewsSignal] = []
        for item in root.findall("./channel/item"):
            headline = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            published_at = (item.findtext("pubDate") or "").strip()
            source = item.find("source")
            publisher = ((source.text or "") if source is not None else "").strip()
            if not headline or not link:
                continue
            signals.append(NewsSignal(
                category=category,
                headline=headline,
                publisher=publisher,
                published_at=published_at,
                link=link,
                feed_hash=digest,
                retrieved_at=retrieved_at,
            ))
            if len(signals) >= limit:
                break
        return signals

    def search(self, query: str, *, category: str, limit: int = 5) -> list[NewsSignal]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not category.strip():
            raise ValueError("category must not be empty")
        uri = (
            f"{self.endpoint}?q={quote_plus(query)}"
            "&hl=en-US&gl=US&ceid=US:en"
        )
        request = Request(uri, headers={"User-Agent": self.user_agent, "Accept": "application/rss+xml, application/xml"})
        with urlopen(request, timeout=self.timeout) as response:
            data = response.read(1_500_000)
        return self._parse(data, category=category, limit=limit)
