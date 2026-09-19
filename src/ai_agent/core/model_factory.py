"""Configuration-driven model provider construction."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import os

from .kaggle_model import KaggleModelProvider
from .kaggle_worker import KaggleGpuWorker
from .local_model import OllamaModel
from .model import ModelProvider, NullModelProvider, OpenAICompatibleModel


@dataclass(frozen=True)
class ModelSettings:
    provider: str = "null"
    model: str = ""
    endpoint: str = ""
    api_key: str = ""
    timeout: float = 120.0
    kaggle_username: str = ""
    kaggle_kernel_slug: str = "ai-agent-llm-worker"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ModelSettings":
        source = os.environ if env is None else env
        provider = source.get("AI_MODEL_PROVIDER", "null").strip().lower()
        model = source.get("AI_MODEL_NAME", "").strip()
        endpoint = source.get("AI_MODEL_ENDPOINT", "").strip()
        if provider == "ollama":
            endpoint = source.get("AI_OLLAMA_BASE_URL", endpoint or "http://127.0.0.1:11434").strip()
        api_key = source.get("AI_MODEL_API_KEY", "").strip()
        if provider == "kaggle":
            api_key = source.get("KAGGLE_API_TOKEN", api_key).strip()
            if not model:
                model = "Qwen/Qwen2.5-3B-Instruct"
        kaggle_username = source.get("KAGGLE_USERNAME", "").strip()
        kaggle_kernel_slug = source.get("KAGGLE_LLM_KERNEL_SLUG", "ai-agent-llm-worker").strip()
        try:
            timeout = float(source.get("AI_MODEL_TIMEOUT", "120"))
        except ValueError as exc:
            raise ValueError("AI_MODEL_TIMEOUT must be numeric") from exc
        return cls(
            provider=provider,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            kaggle_username=kaggle_username,
            kaggle_kernel_slug=kaggle_kernel_slug,
        )


def build_model_provider(settings: ModelSettings | None = None) -> ModelProvider:
    settings = settings or ModelSettings.from_env()
    if settings.timeout <= 0:
        raise ValueError("model timeout must be positive")
    if settings.provider in {"", "null", "none"}:
        return NullModelProvider()
    if settings.provider == "ollama":
        if not settings.model:
            raise ValueError("AI_MODEL_NAME is required for Ollama")
        return OllamaModel(
            model=settings.model,
            base_url=settings.endpoint or "http://127.0.0.1:11434",
            timeout=settings.timeout,
        )
    if settings.provider == "kaggle":
        if not settings.api_key:
            raise ValueError("KAGGLE_API_TOKEN is required for Kaggle model provider")
        if not settings.kaggle_username:
            raise ValueError("KAGGLE_USERNAME is required for Kaggle model provider")
        worker = KaggleGpuWorker(
            api_token=settings.api_key,
            username=settings.kaggle_username,
            timeout=settings.timeout,
        )
        return KaggleModelProvider(
            worker=worker,
            model=settings.model or "Qwen/Qwen2.5-3B-Instruct",
            kernel_slug=settings.kaggle_kernel_slug,
        )
    if settings.provider in {"openai-compatible", "openai_compatible"}:
        if not settings.endpoint:
            raise ValueError("AI_MODEL_ENDPOINT is required for OpenAI-compatible provider")
        return OpenAICompatibleModel(
            endpoint=settings.endpoint,
            api_key=settings.api_key,
            model=settings.model,
            timeout=settings.timeout,
        )
    raise ValueError(f"unsupported AI_MODEL_PROVIDER: {settings.provider}")
