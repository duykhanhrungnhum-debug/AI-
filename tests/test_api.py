import json
import os
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ai_agent.api import AIRequestHandler
from ai_agent.core.model import ModelResponse


class FakeProvider:
    def generate(self, prompt):
        return ModelResponse("Xin chào", "fake", "fake-model")


@pytest.fixture
def api(monkeypatch):
    from http.server import ThreadingHTTPServer
    monkeypatch.setenv("AI_AGENT_API_TOKEN", "family-token")
    monkeypatch.setattr("ai_agent.api.build_model_provider", lambda: FakeProvider())
    server = ThreadingHTTPServer(("127.0.0.1", 0), AIRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()


def test_health(api):
    with urlopen(api + "/health") as response:
        assert json.load(response)["service"] == "AI-"


def test_chat_ui_is_served(api):
    with urlopen(api + "/chat") as response:
        body = response.read().decode("utf-8")
        assert response.headers["content-type"].startswith("text/html")
        assert "AI- Chat" in body
        assert "/v1/chat" in body


def test_translate_requires_family_token(api):
    req = Request(api + "/v1/translate", data=b'{"text":"hello"}',
                  headers={"content-type": "application/json"}, method="POST")
    with pytest.raises(HTTPError) as exc:
        urlopen(req)
    assert exc.value.code == 401


def test_translate_returns_ai_result(api):
    body = json.dumps({"text": "hello", "target_language": "Vietnamese"}).encode()
    req = Request(api + "/v1/translate", data=body, method="POST", headers={
        "content-type": "application/json",
        "authorization": "Bearer family-token",
    })
    with urlopen(req) as response:
        payload = json.load(response)
    assert payload == {"text": "Xin chào", "provider": "fake", "model": "fake-model"}


def test_generate_returns_ai_result(api):
    body = json.dumps({"prompt": "Xin chao AI"}).encode()
    req = Request(api + "/v1/generate", data=body, method="POST", headers={
        "content-type": "application/json",
        "authorization": "Bearer family-token",
    })
    with urlopen(req) as response:
        payload = json.load(response)
    assert payload == {"text": "Xin chào", "provider": "fake", "model": "fake-model"}
