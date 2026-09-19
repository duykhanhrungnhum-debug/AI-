import hashlib
from io import BytesIO
import json
import struct
import zlib
import zipfile

import pytest

from ai_agent.core.kaggle_image_batch import (
    KaggleBatchImageProvider,
    SceneImageRequest,
)
from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission


def png_bytes(width=768, height=432, marker=b"A"):
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data
    ihdr += struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    payload = marker * 32
    text = struct.pack(">I", len(payload)) + b"tEXt" + payload
    text += struct.pack(">I", zlib.crc32(b"tEXt" + payload) & 0xFFFFFFFF)
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return signature + ihdr + text + iend


class FakeWorker:
    def __init__(self, duplicate=False, low_variation=False):
        self.submissions = []
        self.calls = 0
        self.duplicate = duplicate
        self.low_variation = low_variation

    def submit_script(self, **kwargs):
        self.submissions.append(kwargs)
        return KaggleKernelSubmission("testuser", kwargs["slug"], 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def download_output_file(self, slug, filename):
        source = self.submissions[-1]["source"]
        if filename == "images.zip":
            first = png_bytes(marker=b"A")
            second = first if self.duplicate else png_bytes(marker=b"B")
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as zipped:
                zipped.writestr("scene_0000.png", first)
                if "scene-2" in source:
                    zipped.writestr("scene_0001.png", second)
            return buffer.getvalue()

        first = png_bytes(marker=b"A")
        second = first if self.duplicate else png_bytes(marker=b"B")
        scenes = [{
            "scene_id": "scene-1",
            "filename": "scene_0000.png",
            "image_sha256": hashlib.sha256(first).hexdigest(),
            "width": 768,
            "height": 432,
            "seed": 10,
            "pixel_std": 1.0 if self.low_variation else 40.0,
        }]
        if "scene-2" in source:
            scenes.append({
                "scene_id": "scene-2",
                "filename": "scene_0001.png",
                "image_sha256": hashlib.sha256(second).hexdigest(),
                "width": 768,
                "height": 432,
                "seed": 20,
                "pixel_std": 35.0,
            })
        return json.dumps({
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "gpu_name": "Tesla T4",
            "scenes": scenes,
        }).encode()


def requests():
    return [
        SceneImageRequest("scene-1", "haunted house", seed=10),
        SceneImageRequest("scene-2", "old letter on table", seed=20),
    ]


def test_batch_loads_model_once_and_verifies_each_scene():
    worker = FakeWorker()
    provider = KaggleBatchImageProvider(worker=worker, poll_interval=0)

    result = provider.generate_batch(requests())

    assert result.verified is True
    assert len(result.scenes) == 2
    assert len(worker.submissions) == 1
    source = worker.submissions[0]["source"]
    assert source.count("StableDiffusionPipeline.from_pretrained") == 1
    assert "for index, scene in enumerate" in source
    assert all("gpu:Tesla T4" in item.artifact.evidence for item in result.scenes)


def test_batch_rejects_duplicate_scene_ids():
    provider = KaggleBatchImageProvider(worker=FakeWorker(), poll_interval=0)
    duplicate = [
        SceneImageRequest("same", "one"),
        SceneImageRequest("same", "two"),
    ]

    with pytest.raises(ValueError, match="unique"):
        provider.generate_batch(duplicate)


def test_batch_marks_duplicate_images_for_retry():
    provider = KaggleBatchImageProvider(worker=FakeWorker(duplicate=True), poll_interval=0)

    result = provider.generate_batch(requests())

    assert result.verified is False
    assert result.failed_scene_ids == ("scene-1", "scene-2")
    assert all("duplicate image content" in scene.issues for scene in result.scenes)


def test_batch_marks_flat_image_as_failed():
    provider = KaggleBatchImageProvider(worker=FakeWorker(low_variation=True), poll_interval=0)

    result = provider.generate_batch([requests()[0]])

    assert result.verified is False
    assert "image lacks visual variation" in result.scenes[0].issues[0]
