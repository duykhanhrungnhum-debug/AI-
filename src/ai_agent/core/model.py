"""Model-provider boundary; the agent core does not assume a specific vendor or model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: str
    model: str


class ModelProvider(Protocol):
    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response; adapters must expose provider/model provenance."""


class NullModelProvider:
    """Explicit no-model adapter used until a real model is configured."""

    def generate(self, prompt: str) -> ModelResponse:
        raise RuntimeError("no AI model is configured; install/configure a ModelProvider")
