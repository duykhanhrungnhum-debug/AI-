from ai_agent.core.trading_events import (
    enforce_as_of_cutoff,
    normalize_news_signal,
    normalize_news_signals,
)


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


def test_as_of_gate_accepts_event_only_after_retrieval():
    events = normalize_news_signals([sample()])
    before = enforce_as_of_cutoff(events, "2026-09-20T18:03:00+00:00")
    after = enforce_as_of_cutoff(events, "2026-09-20T18:06:00+00:00")
    assert before["eligible_count"] == 0
    assert before["rejected_events"][0]["rejection_reason"] == "retrieved_after_cutoff"
    assert after["eligible_count"] == 1
    assert after["eligible_events"][0]["available_at"] == "2026-09-20T18:05:00+00:00"


def test_as_of_gate_rejects_future_publication_metadata():
    events = normalize_news_signals([
        sample(
            published_at="2026-09-20T18:10:00+00:00",
            retrieved_at="2026-09-20T18:05:00+00:00",
        )
    ])
    gate = enforce_as_of_cutoff(events, "2026-09-20T18:06:00+00:00")
    assert gate["eligible_count"] == 0
    assert gate["rejected_events"][0]["rejection_reason"] == "published_after_cutoff"
    assert gate["rejected_events"][0]["available_at"] == "2026-09-20T18:10:00+00:00"


def test_as_of_gate_uses_retrieval_when_publish_time_unknown():
    events = normalize_news_signals([sample(published_at="")])
    gate = enforce_as_of_cutoff(events, "2026-09-20T18:06:00+00:00")
    assert gate["eligible_count"] == 1
    assert gate["eligible_events"][0]["available_at"] == "2026-09-20T18:05:00+00:00"
