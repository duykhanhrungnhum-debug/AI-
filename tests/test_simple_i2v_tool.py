from __future__ import annotations

from hashlib import sha256
import json

import pytest

from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission
from ai_agent.core.simple_i2v_tool import SimpleKaggleI2VTool


IMAGE = b"reference-image-bytes"
VIDEO = b"\x00\x00\x00\x18ftypisom" + b"V" * 512


class FakeWorker:
    def __init__(self):
        self.source = ""
        self.slug = ""

    def submit_script(self, *, slug, title, source, **kwargs):
        self.source = source
        self.slug = slug
        assert title == slug
        assert kwargs["machine_shape"] == "NvidiaTeslaT4"
        return KaggleKernelSubmission("owner", slug, 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def download_output_file(self, slug, filename):
        if filename == "video_tool.mp4":
            return VIDEO
        if filename == "video_tool_report.json":
            return json.dumps({
                "run_token": self._run_token(),
                "status": "success",
                "stage": "done",
                "model": "Lightricks/LTX-Video",
                "gpu_name": "Tesla T4",
                "input_image_sha256": sha256(IMAGE).hexdigest(),
                "video_sha256": sha256(VIDEO).hexdigest(),
                "width": 512,
                "height": 288,
                "fps": 8.0,
                "num_frames": 33,
                "duration_seconds": 4.125,
                "inference_steps": 30,
                "generation_seconds": 42.0,
                "first_frame_similarity": 0.91,
                "last_frame_similarity": 0.80,
                "motion_delta": 7.5,
                "quality_verified": False,
            }).encode()
        raise FileNotFoundError(filename)

    def _run_token(self):
        marker = '"run_token": "'
        start = self.source.index(marker) + len(marker)
        return self.source[start:self.source.index('"', start)]


def test_ltx_i2v_backend_verifies_exact_artifact_and_source_compiles():
    worker = FakeWorker()
    tool = SimpleKaggleI2VTool(
        worker=worker,
        kernel_slug="aika-i2v-test",
        poll_interval=0,
        max_wait_seconds=1,
        min_video_bytes=32,
    )
    artifact = tool.generate(
        "a paper lantern sways gently",
        IMAGE,
        width=512,
        height=288,
        num_frames=33,
        fps=8,
        seed=7,
    )

    compile(worker.source, "aika-ltx-i2v-worker.py", "exec")
    assert artifact.data == VIDEO
    assert artifact.model == "Lightricks/LTX-Video"
    assert artifact.duration_seconds == pytest.approx(4.125)
    assert "LTXImageToVideoPipeline" in worker.source
    assert "enable_model_cpu_offload" in worker.source
    assert '"quality_verified": False' in worker.source
    assert "StableVideoDiffusionPipeline" not in worker.source
    assert "WanPipeline" not in worker.source
    assert any(item.startswith("input_image_sha256:") for item in artifact.evidence)


def test_ltx_i2v_rejects_invalid_temporal_grid():
    tool = SimpleKaggleI2VTool(worker=FakeWorker(), kernel_slug="bad")
    with pytest.raises(ValueError, match="8\\*k\\+1"):
        tool.generate(
            "motion",
            IMAGE,
            width=512,
            height=288,
            num_frames=32,
            fps=8,
            seed=1,
        )
