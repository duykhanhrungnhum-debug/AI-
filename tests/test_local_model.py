import json

import pytest

from ai_agent.core.local_model import OllamaModel


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_model_calls_local_chat_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "LOCAL_OK"}})

    monkeypatch.setattr("ai_agent.core.local_model.urlopen", fake_urlopen)
    provider = OllamaModel(model="test-model", base_url="http://127.0.0.1:11434", timeout=5)

    result = provider.generate("hello")

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 5
    assert captured["body"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    assert result.text == "LOCAL_OK"
    assert result.provider == "ollama-local"
    assert result.model == "test-model"


def test_ollama_model_rejects_empty_prompt():
    with pytest.raises(ValueError):
        OllamaModel(model="test-model").generate("   ")
