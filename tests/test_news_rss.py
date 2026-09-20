from ai_agent.core.news_rss import GoogleNewsRSSProvider


def test_google_news_rss_parser_produces_hashed_current_event_signals():
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel><item>
      <title>Oil supply disruption test headline</title>
      <link>https://news.google.com/rss/articles/test</link>
      <pubDate>Sun, 20 Sep 2026 08:00:00 GMT</pubDate>
      <source url="https://example.com">Example Publisher</source>
    </item></channel></rss>"""
    provider = GoogleNewsRSSProvider()
    rows = provider._parse(xml, category="geopolitics", limit=3)
    assert len(rows) == 1
    assert rows[0].category == "geopolitics"
    assert rows[0].publisher == "Example Publisher"
    assert rows[0].headline.startswith("Oil supply")
    assert len(rows[0].feed_hash) == 64
    assert rows[0].retrieved_at
