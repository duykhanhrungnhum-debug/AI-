"""No-lookahead event/price reaction study for daily WTI/Brent histories."""
from __future__ import annotations

from datetime import datetime, timedelta


def _event_date(event: dict) -> str | None:
    raw = event.get("published_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def study_event_price_reactions(events: list[dict], history: dict, *, horizons: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Measure close-to-close reactions without using same-day prices as the anchor.

    Daily price histories do not carry an intraday timestamp, so the anchor is
    always the latest observation strictly before the event date. Horizon N is
    the first available market close on or after calendar day N, counting the
    event date as day 1. This preserves the intended 1/3/5-day event windows
    across weekends and market closures without using any pre-event future data.

    Recent events may not yet have every requested horizon. Those rows remain
    useful runtime evidence but are explicitly marked incomplete and do not
    prevent verification when at least one fully observable event/asset row
    proves the complete no-lookahead measurement path.
    """
    rows: list[dict] = []
    required = {str(item) for item in horizons if item > 0}
    for event in events:
        day = _event_date(event)
        if day is None:
            continue
        for asset in ("WTI", "BRENT"):
            observations = history.get(asset, {}).get("observations", [])
            before = [item for item in observations if item.get("date", "") < day]
            after = [item for item in observations if item.get("date", "") >= day]
            if not before:
                continue
            anchor = before[-1]
            reactions = {}
            event_day = datetime.fromisoformat(day).date()
            for horizon in horizons:
                if horizon <= 0:
                    continue
                target_day = (event_day + timedelta(days=horizon - 1)).isoformat()
                candidates = [item for item in after if item.get("date", "") >= target_day]
                if not candidates:
                    continue
                target = candidates[0]
                base = float(anchor["usd_per_barrel"])
                value = float(target["usd_per_barrel"])
                reactions[str(horizon)] = {
                    "target_date": target["date"],
                    "usd_per_barrel": value,
                    "return_pct": round((value / base - 1.0) * 100.0, 6),
                }
            if reactions:
                complete = required.issubset(reactions)
                rows.append({
                    "event_id": event.get("event_id"),
                    "category": event.get("category"),
                    "published_at": event.get("published_at"),
                    "asset": asset,
                    "anchor_date": anchor["date"],
                    "anchor_usd_per_barrel": float(anchor["usd_per_barrel"]),
                    "horizons_trading_days": reactions,
                    "complete_horizons": complete,
                })
    complete_rows = [
        row for row in rows
        if row["complete_horizons"] and row["anchor_date"] < str(row["published_at"])[:10]
    ]
    verified = bool(complete_rows)
    return {
        "verified": verified,
        "method": "previous-trading-day-close to first market close on/after calendar-day horizon",
        "no_lookahead": True,
        "horizons_trading_days": list(horizons),
        "reaction_count": len(rows),
        "complete_reaction_count": len(complete_rows),
        "incomplete_reaction_count": len(rows) - len(complete_rows),
        "reactions": rows,
    }
