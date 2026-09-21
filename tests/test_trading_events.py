from ai_agent.core.trading_events import normalize_news_signal, normalize_news_signals


def sample(**overrides):
    item = {
        "category": "supply_demand",
        "headline": "Oil inventories change after weekly report",
        "publisher": "Example Wire",
        "published_at": "Sun, 20 Sep 2026 18:00:00 GMT",
        "link": "https://example.com/oil-report",
        "feed_hash": "abc123",
        "retrieved_at": "2026-09-20T18:05:00+00:00",
    }
    item.update(overrides)
    return item


def test_normalizes_timestamp_and_stable_identity():
    first = normalize_news_signal(sample())
    second = normalize_news_signal(sample())
    assert first is not None
    assert first.event_id == second.event_id
    assert first.published_at == "2026-09-20T18:00:00+00:00"
    assert first.retrieved_at == "2026-09-20T18:05:00+00:00"
    assert first.category == "supply_demand"


def test_missing_publish_time_is_preserved_as_unknown_not_invented():
    event = normalize_news_signal(sample(published_at=""))
    assert event is not None
    assert event.published_at is None


def test_missing_required_provenance_is_rejected():
    assert normalize_news_signal(sample(link="")) is None
    assert normalize_news_signal(sample(retrieved_at="not-a-time")) is None


def test_batch_deduplicates_same_event():
    events = normalize_news_signals([sample(), sample()])
    assert len(events) == 1
