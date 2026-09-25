from ai_agent.core.production_learning import HttpProductionLearningStore


def test_token_provider_refreshes_for_each_request_boundary():
    tokens = iter(("token-1", "token-2"))
    store = HttpProductionLearningStore(
        base_url="https://example.test/functions/v1/story-processor-api",
        token_provider=lambda: next(tokens),
    )

    assert store._token() == "token-1"
    assert store._token() == "token-2"


def test_static_token_remains_supported():
    store = HttpProductionLearningStore(
        base_url="https://example.test/functions/v1/story-processor-api",
        bearer_token="static-token",
    )

    assert store._token() == "static-token"
