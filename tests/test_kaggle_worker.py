import json

import pytest

from ai_agent.core.kaggle_worker import KaggleGpuWorker


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_kaggle_worker_submits_private_gpu_script_with_bearer_token(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({
            "versionNumber": 3,
            "kernelId": 123,
            "url": "https://www.kaggle.com/code/testuser/gpu-smoke",
        })

    monkeypatch.setattr("ai_agent.core.kaggle_worker.urlopen", fake_urlopen)
    worker = KaggleGpuWorker(api_token="KGAT_secret", username="testuser", timeout=10)

    submission = worker.submit_script(
        slug="gpu-smoke",
        title="GPU Smoke",
        source="print('ok')",
    )

    assert captured["url"] == "https://www.kaggle.com/api/v1/kernels/push"
    assert captured["headers"]["Authorization"] == "Bearer KGAT_secret"
    assert captured["payload"]["slug"] == "testuser/gpu-smoke"
    assert captured["payload"]["enableGpu"] is True
    assert captured["payload"]["machineShape"] == "NvidiaTeslaT4"
    assert captured["payload"]["isPrivate"] is True
    assert submission.ref == "testuser/gpu-smoke"
    assert submission.version_number == 3


def test_kaggle_worker_reads_kernel_status(monkeypatch):
    def fake_urlopen(request, timeout):
        assert "userName=testuser" in request.full_url
        assert "kernelSlug=gpu-smoke" in request.full_url
        return FakeResponse({"status": "COMPLETE", "failureMessage": ""})

    monkeypatch.setattr("ai_agent.core.kaggle_worker.urlopen", fake_urlopen)
    status = KaggleGpuWorker(api_token="KGAT_secret", username="testuser").status("gpu-smoke")

    assert status.terminal is True
    assert status.successful is True


def test_kaggle_worker_rejects_owner_prefixed_slug():
    worker = KaggleGpuWorker(api_token="KGAT_secret", username="testuser")

    with pytest.raises(ValueError):
        worker.submit_script(
            slug="someone/gpu-smoke",
            title="GPU Smoke",
            source="print('ok')",
        )


def test_kaggle_status_requires_status_field(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.core.kaggle_worker.urlopen",
        lambda request, timeout: FakeResponse({}),
    )
    worker = KaggleGpuWorker(api_token="KGAT_secret", username="testuser")

    with pytest.raises(ValueError, match="status"):
        worker.status("gpu-smoke")
