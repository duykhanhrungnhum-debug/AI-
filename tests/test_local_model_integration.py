"""Opt-in live integration test for a real local Ollama model."""
from __future__ import annotations

import os
import pytest

from ai_agent.core.local_model import OllamaModel


@pytest.mark.integration
def test_live_local_ollama_model():
    if os.getenv("AI_LOCAL_MODEL_INTEGRATION") != "1":
        pytest.skip("local model integration is not enabled")

    model_name = os.getenv("AI_MODEL_NAME")
    if not model_name:
        pytest.skip("AI_MODEL_NAME is not configured")

    model = OllamaModel(
        model=model_name,
        base_url=os.getenv("AI_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        timeout=float(os.getenv("AI_MODEL_TIMEOUT", "120")),
    )
    response = model.generate("Reply with exactly: LOCAL_MODEL_OK. Do not add any other text.")

    assert response.provider == "ollama-local"
    assert response.model == model_name
    assert response.text.strip() == "LOCAL_MODEL_OK"
