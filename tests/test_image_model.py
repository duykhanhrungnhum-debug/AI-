import json

import pytest

from ai_agent.core.image_model import ComfyUIImageProvider, ImageGenerationRequest


class FakeResponse:
    def __init__(self, payload, *, raw=False):
        self.payload = payload
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if self.raw:
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


def workflow(request):
    return {"positive": {"prompt": request.prompt, "width": request.width, "height": request.height}}


def test_comfyui_provider_queues_polls_and_downloads(monkeypatch):
    calls = []
    responses = iter([
        FakeResponse({"prompt_id": "p1"}),
        FakeResponse({}),
        FakeResponse({"p1": {"outputs": {"9": {"images": [
            {"filename": "out.png", "subfolder": "", "type": "output"}
        ]}}}}),
        FakeResponse(b"PNGDATA", raw=True),
    ])

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return next(responses)

    monkeypatch.setattr("ai_agent.core.image_model.urlopen", fake_urlopen)
    monkeypatch.setattr("ai_agent.core.image_model.time.sleep", lambda _: None)

    provider = ComfyUIImageProvider(
        workflow_factory=workflow,
        model="local-checkpoint",
        poll_interval=0.01,
        max_poll_attempts=3,
    )
    artifact = provider.generate(ImageGenerationRequest("dark hallway", width=1280, height=720))

    assert artifact.data == b"PNGDATA"
    assert artifact.provider == "comfyui-local"
    assert artifact.model == "local-checkpoint"
    assert artifact.mime_type == "image/png"
    assert any(item.startswith("image_sha256:") for item in artifact.evidence)
    assert calls[0].endswith("/prompt")
    assert calls[1].endswith("/history/p1")
    assert "/view?" in calls[-1]


def test_comfyui_provider_has_bounded_polling(monkeypatch):
    responses = iter([
        FakeResponse({"prompt_id": "p1"}),
        FakeResponse({}),
        FakeResponse({}),
    ])

    monkeypatch.setattr("ai_agent.core.image_model.urlopen", lambda request, timeout: next(responses))
    monkeypatch.setattr("ai_agent.core.image_model.time.sleep", lambda _: None)

    provider = ComfyUIImageProvider(
        workflow_factory=workflow,
        model="local-checkpoint",
        poll_interval=0.01,
        max_poll_attempts=2,
    )

    with pytest.raises(TimeoutError):
        provider.generate(ImageGenerationRequest("scene"))


def test_image_request_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        ImageGenerationRequest("scene", width=0, height=720)
