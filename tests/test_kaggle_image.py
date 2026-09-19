import hashlib
import json

import pytest

from ai_agent.core.image_model import ImageGenerationRequest
from ai_agent.core.kaggle_image import KaggleImageProvider
from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission


class FakeWorker:
    def __init__(self):
        self.submitted = None
        self.statuses = [
            KaggleKernelStatus("RUNNING"),
            KaggleKernelStatus("COMPLETE"),
        ]
        self.image = b"PNG-BYTES"

    def submit_script(self, **kwargs):
        self.submitted = kwargs
        return KaggleKernelSubmission(
            owner="testuser",
            slug=kwargs["slug"],
            version_number=1,
            kernel_id=1,
            url=None,
        )

    def status(self, slug):
        return self.statuses.pop(0)

    def download_output_file(self, slug, filename):
        if filename == "generated.png":
            return self.image
        if filename == "image_report.json":
            return json.dumps({
                "image_sha256": hashlib.sha256(self.image).hexdigest(),
                "width": 512,
                "height": 512,
                "seed": 42,
                "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "gpu_name": "Tesla T4",
            }).encode()
        raise FileNotFoundError(filename)


def test_kaggle_image_provider_runs_open_model_and_verifies_evidence(monkeypatch):
    worker = FakeWorker()
    provider = KaggleImageProvider(
        worker=worker,
        poll_interval=0,
        max_poll_attempts=3,
    )

    artifact = provider.generate(ImageGenerationRequest(
        "cinematic haunted house at night",
        negative_prompt="text, watermark",
        width=512,
        height=512,
        seed=42,
    ))

    assert worker.submitted["enable_internet"] is True
    assert worker.submitted["is_private"] is True
    assert "StableDiffusionPipeline.from_pretrained" in worker.submitted["source"]
    assert artifact.data == b"PNG-BYTES"
    assert artifact.provider == "kaggle-gpu-local-model"
    assert artifact.mime_type == "image/png"
    assert "gpu:Tesla T4" in artifact.evidence
    assert any(item.startswith("image_sha256:") for item in artifact.evidence)


def test_kaggle_image_provider_rejects_non_divisible_dimensions():
    provider = KaggleImageProvider(worker=FakeWorker(), poll_interval=0)

    with pytest.raises(ValueError, match="divisible by 8"):
        provider.generate(ImageGenerationRequest("scene", width=513, height=512))


def test_kaggle_image_provider_rejects_hash_mismatch():
    worker = FakeWorker()

    def bad_download(slug, filename):
        if filename == "generated.png":
            return b"BAD-IMAGE"
        return json.dumps({
            "image_sha256": "wrong",
            "width": 512,
            "height": 512,
            "seed": 42,
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "gpu_name": "Tesla T4",
        }).encode()

    worker.download_output_file = bad_download
    provider = KaggleImageProvider(worker=worker, poll_interval=0)

    with pytest.raises(ValueError, match="hash"):
        provider.generate(ImageGenerationRequest("scene", width=512, height=512, seed=42))
