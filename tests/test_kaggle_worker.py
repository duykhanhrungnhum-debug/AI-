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
    assert "machineShape" not in captured["payload"]
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


def test_kaggle_worker_can_request_specific_machine_shape(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"versionNumber": 1, "kernelId": 1})

    monkeypatch.setattr("ai_agent.core.kaggle_worker.urlopen", fake_urlopen)
    worker = KaggleGpuWorker(api_token="KGAT_secret", username="testuser")
    worker.submit_script(
        slug="gpu-smoke",
        title="GPU Smoke",
        source="print('ok')",
        machine_shape="NvidiaTeslaT4",
    )

    assert captured["payload"]["machineShape"] == "NvidiaTeslaT4"


def test_kaggle_worker_downloads_named_output_file(monkeypatch):
    calls = []

    class RawResponse:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.data

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if "/kernels/output?" in request.full_url:
            return FakeResponse({
                "files": [
                    {"fileName": "gpu_report.json", "url": "https://signed.example/report"}
                ]
            })
        return RawResponse(b'{"gpu_available": true}')

    monkeypatch.setattr("ai_agent.core.kaggle_worker.urlopen", fake_urlopen)
    worker = KaggleGpuWorker(api_token="KGAT_secret", username="testuser")

    data = worker.download_output_file("gpu-smoke", "gpu_report.json")

    assert data == b'{"gpu_available": true}'
    assert any("/kernels/output?" in url for url in calls)
    assert calls[-1] == "https://signed.example/report"
