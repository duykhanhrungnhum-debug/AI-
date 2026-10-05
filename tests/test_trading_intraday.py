from ai_agent.core.trading_intraday import event_study, lock_prediction, order_flow_features

def test_order_flow_respects_asof_and_missing_side():
    trades=[
        {"timestamp":"2026-09-29T10:00:00Z","size":2,"aggressor_side":"BUY"},
        {"timestamp":"2026-09-29T10:01:00Z","size":1,"aggressor_side":"SELL"},
        {"timestamp":"2026-09-29T10:02:00Z","size":9},
        {"timestamp":"2026-09-29T10:03:00Z","size":99,"aggressor_side":"BUY"},
    ]
    x=order_flow_features(trades,as_of="2026-09-29T10:02:00Z")
    assert x["trade_count"]==3
    assert x["classified_volume"]==3
    assert x["volume_delta"]==1
    assert abs(x["buy_sell_imbalance"]-1/3)<1e-12

def test_event_study_has_intraday_horizons_and_no_lookahead():
    event={"event_id":"eia","published_at":"2026-09-29T10:30:00Z"}
    bars=[
        {"timestamp":"2026-09-29T10:30:00Z","close":100,"volume":1},
        {"timestamp":"2026-09-29T10:35:00Z","close":101,"volume":2},
        {"timestamp":"2026-09-29T10:45:00Z","close":99,"volume":3},
        {"timestamp":"2026-09-29T11:00:00Z","close":102,"volume":4},
        {"timestamp":"2026-09-29T11:30:00Z","close":103,"volume":5},
    ]
    r=event_study(event,bars)
    assert r["verified"] is True
    assert r["no_lookahead"] is True
    assert r["horizons_minutes"]==[5,15,30,60]
    assert r["results"]["5"]["return_pct"]==1.0
    assert r["results"]["60"]["return_pct"]==3.0

def test_prediction_is_60m_research_only_and_unscored():
    p=lock_prediction(prediction_at="2026-09-29T10:00:00Z",asset="WTI",direction="UP",confidence=.6,inputs={"delta":2},model_version="intraday-v1")
    assert p["horizon_minutes"]==60
    assert p["outcome"] is None
    assert p["research_only"] is True
