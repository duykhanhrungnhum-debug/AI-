"""No-lookahead event/price reaction study for daily WTI/Brent histories."""
from __future__ import annotations

from datetime import datetime


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

    EIA histories are daily and do not carry an intraday timestamp. To avoid
    accidentally using a close that occurred after an event, the anchor is the
    latest observation strictly before the event date. Horizon N is the Nth
    available trading observation on/after the event date.
    """
    rows: list[dict] = []
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
            for horizon in horizons:
                if horizon <= 0 or len(after) < horizon:
                    continue
                target = after[horizon - 1]
                base = float(anchor["usd_per_barrel"])
                value = float(target["usd_per_barrel"])
                reactions[str(horizon)] = {
                    "target_date": target["date"],
                    "usd_per_barrel": value,
                    "return_pct": round((value / base - 1.0) * 100.0, 6),
                }
            if reactions:
                rows.append({
                    "event_id": event.get("event_id"),
                    "category": event.get("category"),
                    "published_at": event.get("published_at"),
                    "asset": asset,
                    "anchor_date": anchor["date"],
                    "anchor_usd_per_barrel": float(anchor["usd_per_barrel"]),
                    "horizons_trading_days": reactions,
                })
    required = {str(item) for item in horizons}
    verified = bool(rows) and all(
        row["anchor_date"] < str(row["published_at"])[:10]
        and required.issubset(row["horizons_trading_days"])
        for row in rows
    )
    return {
        "verified": verified,
        "method": "previous-trading-day-close to Nth available close on/after event date",
        "no_lookahead": True,
        "horizons_trading_days": list(horizons),
        "reaction_count": len(rows),
        "reactions": rows,
    }
