"""Generate auditable 1H runtime evidence from real delayed/reference market bars."""
from __future__ import annotations

import json
from pathlib import Path

from ai_agent.core.trading_intraday import event_study, lock_prediction
from ai_agent.core.trading_intraday_feed import fetch_reference_bars


def build_evidence() -> dict:
    out = {"schema_version": 1, "research_only": True, "assets": {}}
    for asset in ("WTI", "BRENT"):
        feed = fetch_reference_bars(asset)
        bars = feed["bars"]
        if len(bars) < 14:
            raise RuntimeError(f"{asset}: need >=14 five-minute bars for 60m evidence")
        # Pick a completed observational checkpoint with 60m of real post-bars.
        anchor_index = len(bars) - 13
        observed_at = bars[anchor_index]["timestamp"]
        smoke_event = {
            "event_id": f"runtime-observation-{asset.lower()}",
            "published_at": observed_at,
            "event_type": "runtime_observational_checkpoint",
        }
        study = event_study(smoke_event, bars)
        if not study["verified"] or not study["no_lookahead"]:
            raise RuntimeError(f"{asset}: 60m event-study runtime evidence failed")
        # Lock a paper prediction at the latest observed bar. Outcome intentionally absent.
        prediction = lock_prediction(
            prediction_at=bars[-1]["timestamp"],
            asset=asset,
            direction="NEUTRAL",
            confidence=0.5,
            inputs={
                "source": feed["source"],
                "latest_close": bars[-1]["close"],
                "microstructure_available": False,
                "purpose": "runtime schema lock; not a trained trading signal",
            },
            model_version="intraday-runtime-smoke-v1",
            horizon_minutes=60,
        )
        out["assets"][asset] = {
            "feed": {k: v for k, v in feed.items() if k != "bars"},
            "bar_count": len(bars),
            "first_bar": bars[0]["timestamp"],
            "last_bar": bars[-1]["timestamp"],
            "event_study": study,
            "paper_prediction": prediction,
        }
    out["verified_runtime_scope"] = all(
        x["event_study"]["verified"]
        and x["event_study"]["no_lookahead"]
        and x["paper_prediction"]["horizon_minutes"] == 60
        and x["paper_prediction"]["outcome"] is None
        for x in out["assets"].values()
    )
    return out


if __name__ == "__main__":
    evidence = build_evidence()
    path = Path("artifacts/trading_intraday_1h_runtime.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
    print(json.dumps(evidence, indent=2, sort_keys=True))
