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


def test_worker_preencodes_prompts_on_cpu_before_gpu_offload():
    provider = KaggleBatchVideoProvider(
        worker=FakeWorker(),
        poll_interval=0,
        max_poll_attempts=1,
        inference_steps=4,
        min_video_bytes=32,
    )
    request = SceneVideoRequest(
        scene_id="memory-1",
        prompt="A lantern sways in a quiet food stall.",
        width=480,
        height=272,
        num_frames=9,
        fps=8,
        seed=3,
    )

    source = provider._build_worker_source((request,))
    encode_at = source.index("pipe.encode_prompt(")
    release_at = source.index("pipe.text_encoder = None")
    offload_at = source.index("pipe.enable_model_cpu_offload()")
    generate_at = source.index("frames = pipe(")

    assert "device=cpu_device" in source
    assert 'torch.Generator(device="cpu")' in source
    assert encode_at < release_at < offload_at < generate_at
    assert "CUBLAS_STATUS_ALLOC_FAILED" in source
    assert '"diffusers==0.35.2"' in source


class FailingWorker(FakeWorker):
    def status(self, slug):
        return KaggleKernelStatus("ERROR", "")

    def logs(self, slug):
        return "old-prefix-" + ("x" * 9000) + "-ACTUAL-TERMINAL-ERROR"


def test_failure_reporting_keeps_terminal_log_tail():
    provider = KaggleBatchVideoProvider(
        worker=FailingWorker(),
        poll_interval=0,
        max_poll_attempts=1,
    )
    with pytest.raises(RuntimeError) as exc:
        provider._wait()

    message = str(exc.value)
    assert "ACTUAL-TERMINAL-ERROR" in message
    assert "old-prefix-" not in message
    assert "...[tail]" in message
