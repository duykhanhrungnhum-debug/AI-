"""Deterministic normalization of trading research signals into event records.

This module does not predict price direction and does not execute trades. It only
turns already-collected research signals into a stable, auditable schema for the
next curriculum stages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256


@dataclass(frozen=True)
class NormalizedTradingEvent:
    event_id: str
    category: str
    headline: str
    publisher: str
    source_link: str
    published_at: str | None
    retrieved_at: str
    source_feed_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def _utc_iso(value: str) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def normalize_news_signal(signal: dict) -> NormalizedTradingEvent | None:
    """Normalize one collected news signal without inventing missing facts."""
    category = str(signal.get("category", "")).strip()
    headline = str(signal.get("headline", "")).strip()
    publisher = str(signal.get("publisher", "")).strip()
    link = str(signal.get("link", "")).strip()
    retrieved_at = _utc_iso(str(signal.get("retrieved_at", "")))
    feed_hash = str(signal.get("feed_hash", "")).strip()
    if not category or not headline or not link or not retrieved_at:
        return None

    published_at = _utc_iso(str(signal.get("published_at", "")))
    identity = "\n".join((category, headline, publisher, link, published_at or "", feed_hash))
    event_id = sha256(identity.encode("utf-8")).hexdigest()[:24]
    return NormalizedTradingEvent(
        event_id=event_id,
        category=category,
        headline=headline,
        publisher=publisher,
        source_link=link,
        published_at=published_at,
        retrieved_at=retrieved_at,
        source_feed_hash=feed_hash,
    )


def normalize_news_signals(signals: list[dict]) -> list[dict]:
    """Normalize and de-duplicate signals while preserving first-seen order."""
    output: list[dict] = []
    seen: set[str] = set()
    for signal in signals:
        event = normalize_news_signal(signal)
        if event is None or event.event_id in seen:
            continue
        seen.add(event.event_id)
        output.append(event.to_dict())
    return output
