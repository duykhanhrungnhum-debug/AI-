"""Local model providers for zero-API-cost runtimes."""
from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.request import Request, urlopen

from .invariants import assert_core_invariants
from .model import ModelResponse


@dataclass
class OllamaModel:
    """ModelProvider backed by a local Ollama server."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 120.0
    provider: str = "ollama-local"

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use HTTP(S)")
        if not self.model.strip():
            raise ValueError("model is required")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/api/chat"

    def generate(self, prompt: str) -> ModelResponse:
        assert_core_invariants()
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "AI-Agent-Local/0.1"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Ollama response does not contain message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama model content must be non-empty text")
        return ModelResponse(content, self.provider, self.model)
