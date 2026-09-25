from io import BytesIO
import hashlib
import json
import zipfile

import pytest

from ai_agent.core.kaggle_i2v import (
    KaggleBatchImageToVideoProvider,
    SceneImageToVideoRequest,
)
from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission


def _mp4_bytes(tag: bytes = b"A") -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + tag * 256


class FakeWorker:
    def __init__(self):
        self.source = ""

    def submit_script(self, *, slug, title, source, **kwargs):
        self.source = source
        return KaggleKernelSubmission("owner", slug, 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def logs(self, slug):
        return ""

    def download_output_file(self, slug, filename):
        video = _mp4_bytes()
        digest = hashlib.sha256(video).hexdigest()
        image = b"PNG-REFERENCE"
        image_digest = hashlib.sha256(image).hexdigest()
        if filename == "i2v_videos.zip":
            out = BytesIO()
            with zipfile.ZipFile(out, "w") as archive:
                archive.writestr("scene_0000.mp4", video)
            return out.getvalue()
        if filename == "i2v_batch_report.json":
            return json.dumps({
                "model": "stabilityai/stable-video-diffusion-img2vid-xt",
                "gpu_name": "Tesla T4",
                "scenes": [{
                    "scene_id": "scene-1",
                    "filename": "scene_0000.mp4",
                    "video_sha256": digest,
                    "input_image_sha256": image_digest,
                    "width": 1024,
                    "height": 576,
                    "fps": 7.0,
                    "duration_seconds": 2.0,
                    "seed": 7,
                    "motion_bucket_id": 96,
                    "noise_aug_strength": 0.02,
                    "first_frame_similarity": 0.91,
                    "last_frame_similarity": 0.78,
                    "motion_delta": 8.5,
                }],
            }).encode()
        raise FileNotFoundError(filename)


def test_i2v_provider_verifies_motion_and_identity():
    provider = KaggleBatchImageToVideoProvider(
        worker=FakeWorker(),
        poll_interval=0,
        max_poll_attempts=1,
        min_video_bytes=32,
    )
    result = provider.generate_batch([
        SceneImageToVideoRequest(
            scene_id="scene-1",
            image=b"PNG-REFERENCE",
            width=512,
            height=288,
            num_frames=14,
            fps=7,
            seed=7,
        )
    ])

    assert result.verified is True
    assert result.scenes[0].artifact.width == 1024
    assert result.scenes[0].artifact.height == 576
    evidence = result.scenes[0].artifact.evidence
    assert "last_frame_similarity:0.780000" in evidence
    assert "motion_delta:8.500000" in evidence
    assert "conditioning_dimensions:512x288" in evidence
    assert "dimensions:1024x576" in evidence
    assert "gpu:Tesla T4" in evidence


def test_i2v_worker_source_compiles_and_reuses_models():
    worker = FakeWorker()
    provider = KaggleBatchImageToVideoProvider(
        worker=worker,
        poll_interval=0,
        max_poll_attempts=1,
        min_video_bytes=32,
    )
    request = SceneImageToVideoRequest(
        scene_id="compile-1",
        image=b"image",
        width=512,
        height=288,
        num_frames=14,
        fps=7,
        seed=1,
    )
    source = provider._build_worker_source((request,))
    compile(source, "kaggle-i2v-worker.py", "exec")

    assert "StableVideoDiffusionPipeline" in source
    assert "enable_model_cpu_offload" in source
    assert "enable_forward_chunking" in source
    assert source.count("StableVideoDiffusionPipeline.from_pretrained") == 1
    assert source.count("CLIPVisionModel.from_pretrained") == 1


def test_i2v_rejects_identity_drift_and_frozen_motion():
    provider = KaggleBatchImageToVideoProvider(worker=FakeWorker())
    request = SceneImageToVideoRequest(
        scene_id="scene-1",
        image=b"PNG-REFERENCE",
        width=512,
        height=288,
        num_frames=14,
        fps=7,
        seed=7,
    )
    video = _mp4_bytes()
    digest = hashlib.sha256(video).hexdigest()
    issues = provider._verify_scene(
        request,
        {
            "video_sha256": digest,
            "input_image_sha256": hashlib.sha256(request.image).hexdigest(),
            "width": 800,
            "height": 600,
            "duration_seconds": 2.0,
            "first_frame_similarity": 0.80,
            "last_frame_similarity": 0.30,
            "motion_delta": 0.1,
        },
        video,
        digest,
    )
    assert any("identity similarity below threshold" in issue for issue in issues)
    assert any("insufficient motion" in issue for issue in issues)
    assert any("video aspect ratio mismatch" in issue for issue in issues)


def test_i2v_request_rejects_empty_image():
    with pytest.raises(ValueError, match="image is required"):
        SceneImageToVideoRequest(scene_id="bad", image=b"")
