from io import BytesIO
import json
import zipfile

import pytest

from ai_agent.core.kaggle_video import KaggleBatchVideoProvider, SceneVideoRequest
from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission


def _mp4_bytes(tag: bytes = b"A") -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + tag * 128


class FakeWorker:
    def __init__(self):
        self.source = ""
        self.submissions = 0

    def submit_script(self, *, slug, title, source, **kwargs):
        self.source = source
        self.submissions += 1
        return KaggleKernelSubmission("owner", slug, 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def logs(self, slug):
        return ""

    def download_output_file(self, slug, filename):
        video = _mp4_bytes()
        digest = __import__("hashlib").sha256(video).hexdigest()
        if filename == "videos.zip":
            out = BytesIO()
            with zipfile.ZipFile(out, "w") as archive:
                archive.writestr("scene_0000.mp4", video)
            return out.getvalue()
        if filename == "video_batch_report.json":
            return json.dumps({
                "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "gpu_name": "Tesla T4",
                "scenes": [{
                    "scene_id": "scene-1",
                    "filename": "scene_0000.mp4",
                    "video_sha256": digest,
                    "width": 832,
                    "height": 480,
                    "fps": 16.0,
                    "duration_seconds": 1.0625,
                    "seed": 7,
                }],
            }).encode()
        raise FileNotFoundError(filename)


def test_kaggle_video_provider_builds_worker_and_verifies_output():
    worker = FakeWorker()
    provider = KaggleBatchVideoProvider(
        worker=worker,
        poll_interval=0,
        max_poll_attempts=1,
        inference_steps=8,
        min_video_bytes=32,
    )
    result = provider.generate_batch([
        SceneVideoRequest(
            scene_id="scene-1",
            prompt="A woman walks through a Vietnamese night market, cinematic motion.",
            width=832,
            height=480,
            num_frames=17,
            fps=16,
            seed=7,
        )
    ])

    assert result.verified is True
    assert result.scenes[0].artifact.width == 832
    assert result.scenes[0].artifact.height == 480
    assert result.scenes[0].artifact.duration_seconds == pytest.approx(1.0625)
    assert any(item == "gpu:Tesla T4" for item in result.scenes[0].artifact.evidence)
    assert "WanPipeline" in worker.source
    assert "enable_model_cpu_offload" in worker.source
    assert "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" in worker.source


def test_scene_video_request_enforces_wan_frame_rule():
    with pytest.raises(ValueError, match="4\\*k\\+1"):
        SceneVideoRequest(scene_id="bad", prompt="motion", num_frames=18)


def test_provider_rejects_duplicate_scene_ids():
    provider = KaggleBatchVideoProvider(worker=FakeWorker(), poll_interval=0, max_poll_attempts=1)
    request = SceneVideoRequest(scene_id="same", prompt="motion")
    with pytest.raises(ValueError, match="unique"):
        provider.generate_batch([request, request])


def test_worker_source_compiles_before_kaggle_submission():
    provider = KaggleBatchVideoProvider(
        worker=FakeWorker(),
        poll_interval=0,
        max_poll_attempts=1,
        inference_steps=8,
        min_video_bytes=32,
    )
    request = SceneVideoRequest(
        scene_id="compile-1",
        prompt="A realistic food stall with subtle motion.",
        width=832,
        height=480,
        num_frames=17,
        fps=16,
        seed=1,
    )

    source = provider._build_worker_source((request,))
    compile(source, "kaggle-video-worker.py", "exec")
    assert source.startswith("from __future__ import annotations")
