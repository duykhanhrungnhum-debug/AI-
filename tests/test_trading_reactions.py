from ai_agent.core.trading_reactions import study_event_price_reactions


def test_reaction_study_uses_only_pre_event_anchor_and_clear_horizons():
    events = [{"event_id": "e1", "category": "supply_demand", "published_at": "2026-09-02T12:00:00+00:00"}]
    observations = [
        {"date": "2026-09-01", "usd_per_barrel": 100.0},
        {"date": "2026-09-02", "usd_per_barrel": 101.0},
        {"date": "2026-09-03", "usd_per_barrel": 103.0},
        {"date": "2026-09-04", "usd_per_barrel": 102.0},
        {"date": "2026-09-07", "usd_per_barrel": 105.0},
    ]
    history = {asset: {"observations": observations} for asset in ("WTI", "BRENT")}
    study = study_event_price_reactions(events, history, horizons=(1, 3, 5))
    assert study["verified"] is True
    assert study["no_lookahead"] is True
    assert study["complete_reaction_count"] == 2
    row = study["reactions"][0]
    assert row["anchor_date"] == "2026-09-01"
    assert row["horizons_trading_days"]["1"]["target_date"] == "2026-09-02"
    assert row["horizons_trading_days"]["3"]["return_pct"] == 2.0
    assert row["horizons_trading_days"]["5"]["return_pct"] == 5.0


def test_reaction_study_not_verified_when_full_horizons_unavailable():
    events = [{"event_id": "e1", "published_at": "2026-09-02T12:00:00+00:00"}]
    history = {asset: {"observations": [
        {"date": "2026-09-01", "usd_per_barrel": 100.0},
        {"date": "2026-09-02", "usd_per_barrel": 101.0},
    ]} for asset in ("WTI", "BRENT")}
    study = study_event_price_reactions(events, history, horizons=(1, 3, 5))
    assert study["verified"] is False
    assert study["complete_reaction_count"] == 0


def test_recent_incomplete_event_does_not_invalidate_mature_reaction_evidence():
    events = [
        {"event_id": "mature", "published_at": "2026-09-02T12:00:00+00:00"},
        {"event_id": "recent", "published_at": "2026-09-07T12:00:00+00:00"},
    ]
    observations = [
        {"date": "2026-09-01", "usd_per_barrel": 100.0},
        {"date": "2026-09-02", "usd_per_barrel": 101.0},
        {"date": "2026-09-03", "usd_per_barrel": 102.0},
        {"date": "2026-09-04", "usd_per_barrel": 103.0},
        {"date": "2026-09-07", "usd_per_barrel": 104.0},
    ]
    history = {asset: {"observations": observations} for asset in ("WTI", "BRENT")}
    study = study_event_price_reactions(events, history, horizons=(1, 3, 5))
    assert study["verified"] is True
    assert study["complete_reaction_count"] == 2
    assert study["incomplete_reaction_count"] == 2
    assert all(row["anchor_date"] < row["published_at"][:10] for row in study["reactions"])
