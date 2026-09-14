import json

from ai_agent.core.search import DuckDuckGoSearchProvider, JsonSearchProvider, SearchResult, StaticSearchProvider


def test_static_search_provider_is_bounded():
    provider = StaticSearchProvider(
        [SearchResult("https://example.test/1"), SearchResult("https://example.test/2")]
    )
    results = provider.search("autonomous learning", limit=1)
    assert len(results) == 1
    assert results[0].url.endswith("/1")


def test_json_search_provider_parses_results(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"results": [{"url": "https://example.test/a", "title": "A", "snippet": "B"}]}
            ).encode()

    monkeypatch.setattr("ai_agent.core.search.urlopen", lambda request, timeout: FakeResponse())
    provider = JsonSearchProvider("https://search.example.test/api")
    results = provider.search("agent learning", limit=2)
    assert results == [SearchResult("https://example.test/a", "A", "B")]


def test_duckduckgo_provider_parses_public_html(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return b'<a class="result__a" href="https://example.test/a">Example A</a>'

    monkeypatch.setattr("ai_agent.core.search.urlopen", lambda request, timeout: FakeResponse())
    results = DuckDuckGoSearchProvider().search("agent", limit=1)
    assert results == [SearchResult("https://example.test/a", "Example A")]
