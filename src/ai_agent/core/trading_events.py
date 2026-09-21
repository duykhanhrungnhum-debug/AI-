"""Deterministic normalization and no-lookahead gating for trading research events.

This module does not predict price direction and does not execute trades. It
turns already-collected research signals into a stable, auditable schema and
provides a conservative as-of gate for later historical studies/backtests.
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


def _iso_dt(value: str | None) -> datetime | None:
    normalized = _utc_iso(value or "")
    if normalized is None:
        return None
    return datetime.fromisoformat(normalized)


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


def enforce_as_of_cutoff(events: list[dict], as_of: str) -> dict:
    """Conservatively prevent look-ahead use of information.

    An event is eligible only after the Agent has actually retrieved it and,
    when a publication timestamp exists, after that publication timestamp too.
    The available_at time is the later of published_at and retrieved_at.
    Missing publication time is never invented; retrieved_at alone is used.
    """
    cutoff = _iso_dt(as_of)
    if cutoff is None:
        raise ValueError("as_of must be a valid timestamp")

    eligible: list[dict] = []
    rejected: list[dict] = []
    for raw in events:
        event = dict(raw)
        retrieved = _iso_dt(str(event.get("retrieved_at", "")))
        published = _iso_dt(event.get("published_at"))
        if retrieved is None:
            event["available_at"] = None
            event["eligible_as_of"] = False
            event["rejection_reason"] = "missing_retrieved_at"
            rejected.append(event)
            continue

        available = max(retrieved, published) if published is not None else retrieved
        event["available_at"] = available.astimezone(timezone.utc).isoformat()
        event["eligible_as_of"] = available <= cutoff

        if event["eligible_as_of"]:
            event["rejection_reason"] = None
            eligible.append(event)
            continue

        if retrieved > cutoff:
            reason = "retrieved_after_cutoff"
        elif published is not None and published > cutoff:
            reason = "published_after_cutoff"
        else:
            reason = "available_after_cutoff"
        event["rejection_reason"] = reason
        rejected.append(event)

    return {
        "as_of": cutoff.astimezone(timezone.utc).isoformat(),
        "eligible_count": len(eligible),
        "rejected_count": len(rejected),
        "eligible_events": eligible,
        "rejected_events": rejected,
    }
