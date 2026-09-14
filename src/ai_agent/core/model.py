"""Model-provider boundary with an optional OpenAI-compatible HTTP adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import json
from urllib.request import Request, urlopen

from .invariants import assert_core_invariants


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: str
    model: str


class ModelProvider(Protocol):
    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response with provider/model provenance."""


class NullModelProvider:
    """Explicit no-model adapter used until a real model is configured."""

    def generate(self, prompt: str) -> ModelResponse:
        raise RuntimeError("no AI model is configured; install/configure a ModelProvider")


@dataclass
class OpenAICompatibleModel:
    """Adapter for a configured OpenAI-compatible chat-completions endpoint."""

    endpoint: str
    api_key: str
    model: str
    timeout: float = 60.0
    provider: str = "openai-compatible"

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must use HTTP(S)")
        if not self.api_key or not self.model.strip():
            raise ValueError("api_key and model are required")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def generate(self, prompt: str) -> ModelResponse:
        assert_core_invariants()
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        request = Request(self.endpoint, data=payload, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AI-Agent-Model/0.1",
        })
        with urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("model response does not contain choices[0].message.content") from exc
        if not isinstance(content, str):
            raise ValueError("model content must be text")
        return ModelResponse(content, self.provider, self.model)
