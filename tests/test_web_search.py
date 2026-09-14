from ai_agent.core.web_search import _DuckParser


def test_duckduckgo_parser_extracts_result_links():
    parser = _DuckParser()
    parser.feed('<a class="result__a" href="https://example.com">Example</a>')
    assert parser.results[0].url == "https://example.com"
    assert parser.results[0].title == "Example"
