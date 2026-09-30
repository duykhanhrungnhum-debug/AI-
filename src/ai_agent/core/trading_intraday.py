"""Point-in-time intraday event-study primitives for WTI/Brent research.

Research/paper evaluation only. No broker integration or live-order capability.
The caller must provide timestamped bars/trades observed from a legitimate feed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

HORIZONS_MINUTES = (5, 15, 30, 60)


def _ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return dt.astimezone(timezone.utc)


def order_flow_features(trades: list[dict], *, as_of: str) -> dict:
    """Compute only features observable at or before as_of.

    aggressor_side must be BUY/SELL when the feed supplies it. Missing side is
    excluded rather than inferred, so unavailable microstructure is never faked.
    """
    cutoff = _ts(as_of)
    seen = [t for t in trades if _ts(str(t["timestamp"])) <= cutoff]
    buy = sum(float(t["size"]) for t in seen if t.get("aggressor_side") == "BUY")
    sell = sum(float(t["size"]) for t in seen if t.get("aggressor_side") == "SELL")
    classified = buy + sell
    return {
        "as_of": cutoff.isoformat(),
        "trade_count": len(seen),
        "aggressor_buy_volume": buy,
        "aggressor_sell_volume": sell,
        "volume_delta": buy - sell,
        "buy_sell_imbalance": None if classified == 0 else (buy - sell) / classified,
        "classified_volume": classified,
    }


def event_study(event: dict, bars: list[dict], *, horizons=HORIZONS_MINUTES) -> dict:
    """Measure 5/15/30/60m post-event returns without using future data as inputs."""
    published = _ts(str(event["published_at"]))
    ordered = sorted(bars, key=lambda b: _ts(str(b["timestamp"])))
    pre = [b for b in ordered if _ts(str(b["timestamp"])) <= published]
    if not pre:
        return {"event_id": event.get("event_id"), "verified": False, "reason": "no_pre_event_bar"}
    anchor = pre[-1]
    anchor_ts = _ts(str(anchor["timestamp"]))
    anchor_px = float(anchor["close"])
    results = {}
    for minutes in horizons:
        target = published + timedelta(minutes=int(minutes))
        eligible = [b for b in ordered if published < _ts(str(b["timestamp"])) <= target]
        if not eligible:
            results[str(minutes)] = None
            continue
        end = eligible[-1]
        results[str(minutes)] = {
            "end_timestamp": _ts(str(end["timestamp"])).isoformat(),
            "return_pct": round((float(end["close"]) / anchor_px - 1.0) * 100.0, 8),
            "volume": sum(float(b.get("volume") or 0.0) for b in eligible),
        }
    no_lookahead = anchor_ts <= published
    return {
        "event_id": event.get("event_id"),
        "published_at": published.isoformat(),
        "anchor_timestamp": anchor_ts.isoformat(),
        "horizons_minutes": list(horizons),
        "results": results,
        "no_lookahead": no_lookahead,
        "verified": no_lookahead and results.get("60") is not None,
    }


def lock_prediction(*, prediction_at: str, asset: str, direction: str, confidence: float,
                    inputs: dict, model_version: str, horizon_minutes: int = 60) -> dict:
    """Create an immutable-shaped paper-prediction record before outcome is known."""
    if direction not in {"UP", "DOWN", "NEUTRAL"}:
        raise ValueError("direction must be UP, DOWN, or NEUTRAL")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if horizon_minutes != 60:
        raise ValueError("primary paper-prediction horizon must be 60 minutes")
    return {
        "prediction_at": _ts(prediction_at).isoformat(),
        "asset": asset,
        "direction": direction,
        "confidence": confidence,
        "horizon_minutes": horizon_minutes,
        "inputs": inputs,
        "model_version": model_version,
        "outcome": None,
        "research_only": True,
    }
