import pytest

from ai_agent.core.local_model import OllamaModel
from ai_agent.core.model import NullModelProvider, OpenAICompatibleModel
from ai_agent.core.model_factory import ModelSettings, build_model_provider
from ai_agent.core.runtime_factory import build_model_runtime
from ai_agent.core.engine import ExecutionEngine
from ai_agent.core.project_state import StateStore
from ai_agent.core.recovery import RecoveryGuard


class NoopExecutor:
    def execute(self, decision):
        raise AssertionError("not used in composition test")


def test_model_settings_build_local_ollama_provider():
    settings = ModelSettings.from_env({
        "AI_MODEL_PROVIDER": "ollama",
        "AI_MODEL_NAME": "qwen-test",
    })
    provider = build_model_provider(settings)

    assert isinstance(provider, OllamaModel)
    assert provider.model == "qwen-test"
    assert provider.base_url == "http://127.0.0.1:11434"


def test_model_settings_support_custom_local_endpoint():
    settings = ModelSettings.from_env({
        "AI_MODEL_PROVIDER": "ollama",
        "AI_MODEL_NAME": "local-model",
        "AI_OLLAMA_BASE_URL": "http://10.0.0.5:11434",
        "AI_MODEL_TIMEOUT": "30",
    })
    provider = build_model_provider(settings)

    assert isinstance(provider, OllamaModel)
    assert provider.base_url == "http://10.0.0.5:11434"
    assert provider.timeout == 30


def test_model_factory_keeps_null_provider_explicit():
    assert isinstance(build_model_provider(ModelSettings()), NullModelProvider)


def test_model_factory_builds_openai_compatible_provider():
    provider = build_model_provider(ModelSettings(
        provider="openai-compatible",
        endpoint="https://example.com/v1/chat/completions",
        api_key="secret",
        model="test-model",
    ))
    assert isinstance(provider, OpenAICompatibleModel)


def test_model_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        build_model_provider(ModelSettings(provider="mystery"))


def test_runtime_factory_wires_local_provider(tmp_path):
    runtime = build_model_runtime(
        executor=NoopExecutor(),
        engine=ExecutionEngine(StateStore(tmp_path / "state.json"), RecoveryGuard()),
        settings=ModelSettings(provider="ollama", model="qwen-test", endpoint="http://127.0.0.1:11434"),
    )

    assert isinstance(runtime.model_agent.provider, OllamaModel)


def test_model_factory_builds_kaggle_open_model_provider():
    from ai_agent.core.kaggle_model import KaggleModelProvider

    settings = ModelSettings.from_env({
        "AI_MODEL_PROVIDER": "kaggle",
        "KAGGLE_API_TOKEN": "KGAT_test_secret",
        "KAGGLE_USERNAME": "duykhanhta",
        "KAGGLE_LLM_KERNEL_SLUG": "ai-agent-test-llm",
    })
    provider = build_model_provider(settings)

    assert isinstance(provider, KaggleModelProvider)
    assert provider.model == "Qwen/Qwen2.5-3B-Instruct"
    assert provider.kernel_slug == "ai-agent-test-llm"
    assert provider.worker.username == "duykhanhta"


def test_model_factory_requires_kaggle_username():
    with pytest.raises(ValueError, match="KAGGLE_USERNAME"):
        build_model_provider(ModelSettings(
            provider="kaggle",
            model="Qwen/Qwen2.5-3B-Instruct",
            api_key="KGAT_test_secret",
        ))
