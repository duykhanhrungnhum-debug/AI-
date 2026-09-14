"""Opt-in live integration test for a real OpenAI-compatible model endpoint."""
from __future__ import annotations

import os

import pytest

from ai_agent.core.model import OpenAICompatibleModel


@pytest.mark.integration
def test_live_openai_compatible_model():
    endpoint = os.getenv("AI_MODEL_ENDPOINT")
    api_key = os.getenv("AI_MODEL_API_KEY")
    model_name = os.getenv("AI_MODEL_NAME")

    if not all((endpoint, api_key, model_name)):
        pytest.skip("live model credentials are not configured")

    model = OpenAICompatibleModel(
        endpoint=endpoint,
        api_key=api_key,
        model=model_name,
        timeout=float(os.getenv("AI_MODEL_TIMEOUT", "60")),
    )
    response = model.generate(
        "Reply with exactly: LIVE_MODEL_OK. Do not add any other text."
    )

    assert response.provider == "openai-compatible"
    assert response.model == model_name
    assert response.text.strip() == "LIVE_MODEL_OK"
