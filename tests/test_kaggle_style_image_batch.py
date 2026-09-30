import hashlib
import json
import struct
import zlib

from ai_agent.core.kaggle_style_image_batch import (
    KaggleStyleRoutedImageBatchProvider,
    StyleRoutedImageItem,
)
from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission


def png_bytes(marker=b"A"):
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1024, 1024, 8, 2, 0, 0, 0)
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data
    ihdr += struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    payload = marker * 24
    text = struct.pack(">I", len(payload)) + b"tEXt" + payload
    text += struct.pack(">I", zlib.crc32(b"tEXt" + payload) & 0xFFFFFFFF)
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return signature + ihdr + text + iend


class FakeWorker:
    def __init__(self):
        self.submissions = []
        self.images = {
            "real__candidate_0.png": png_bytes(b"R"),
            "cute3d__candidate_0.png": png_bytes(b"A"),
            "cute3d__candidate_1.png": png_bytes(b"B"),
            "cute3d__candidate_2.png": png_bytes(b"C"),
        }

    def submit_script(self, **kwargs):
        self.submissions.append(kwargs)
        return KaggleKernelSubmission("testuser", kwargs["slug"], 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def logs(self, slug):
        return ""

    def download_output_file(self, slug, filename):
        if filename in self.images:
            return self.images[filename]
        if filename != "style_batch_report.json":
            raise FileNotFoundError(filename)
        def candidate(name, index, engine, model):
            data = self.images[name]
            return {
                "candidate_index": index,
                "filename": name,
                "seed": 10 + index,
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "engine_id": engine,
                "model": model,
                "scheduler": "edm_dpm" if engine == "mascot3d" else "dpm_karras",
            }
        return json.dumps({
            "gpu_name": "Tesla T4",
            "items": {
                "real": {
                    "prompt": "one water buffalo in a field",
                    "subject_class": "animal",
                    "style_class": "photo",
                    "subject_name": "water buffalo",
                    "semantic_profile": "animal_photo_premium",
                    "candidates": [candidate("real__candidate_0.png", 0, "photoreal", "SG161222/RealVisXL_V5.0")],
                },
                "cute3d": {
                    "prompt": "one cute 3D water buffalo mascot",
                    "subject_class": "animal",
                    "style_class": "3d",
                    "subject_name": "water buffalo",
                    "semantic_profile": "mascot_premium",
                    "candidates": [
                        candidate(f"cute3d__candidate_{index}.png", index, "mascot3d", "playgroundai/playground-v2.5-1024px-aesthetic")
                        for index in range(3)
                    ],
                },
            },
        }).encode()


def provider(worker=None):
    engines = {
        "animal_photo_premium": {
            "engine_id": "photoreal", "model": "SG161222/RealVisXL_V5.0", "variant": "fp16",
            "steps": 28, "guidance": 4.0, "scheduler": "dpm_karras", "candidate_count": 1,
        },
        "mascot_premium": {
            "engine_id": "mascot3d", "model": "playgroundai/playground-v2.5-1024px-aesthetic", "variant": "fp16",
            "steps": 40, "guidance": 3.0, "scheduler": "edm_dpm", "candidate_count": 3,
        },
        "human_photo_premium": {
            "engine_id": "photoreal", "model": "SG161222/RealVisXL_V5.0", "variant": "fp16",
            "steps": 28, "guidance": 4.0, "scheduler": "dpm_karras", "candidate_count": 1,
        },
        "general_premium": {
            "engine_id": "photoreal", "model": "SG161222/RealVisXL_V5.0", "variant": "fp16",
            "steps": 28, "guidance": 4.0, "scheduler": "dpm_karras", "candidate_count": 1,
        },
    }
    keys = set(engines)
    return KaggleStyleRoutedImageBatchProvider(
        worker=worker or FakeWorker(),
        profile_engines=engines,
        profile_positive_constraints={key: "premium quality" for key in keys},
        profile_negative_constraints={key: "bad anatomy, text, logo" for key in keys},
        profile_dimensions={key: (1024, 1024) for key in keys},
        poll_interval=0,
    )


def test_worker_uses_semantic_router_without_species_whitelist_and_two_style_engines():
    source = provider()._build_worker_source((
        StyleRoutedImageItem("real", "Tạo ảnh con trâu thật", 10),
        StyleRoutedImageItem("cute3d", "Tạo ảnh con trâu 3D cute", 20),
    ))
    compile(source, "<style-routed-worker>", "exec")

    assert "Do not depend on a fixed list of animal species" in source
    assert "Do not invent anatomy counts or appendages" in source
    assert "SG161222/RealVisXL_V5.0" in source
    assert "playgroundai/playground-v2.5-1024px-aesthetic" in source
    assert "EDMDPMSolverMultistepScheduler" in source
    assert "DPMSolverMultistepScheduler" in source
    assert "engine_groups" in source
    assert "species-anatomy guard" in source.casefold()
    assert "extra horns" in source


def test_style_engines_load_sequentially_inside_one_worker_source():
    source = provider()._build_worker_source((
        StyleRoutedImageItem("real", "Tạo ảnh voi thật", 10),
        StyleRoutedImageItem("cute3d", "Tạo ảnh voi 3D mascot", 20),
    ))
    assert source.count("AutoPipelineForText2Image.from_pretrained") == 1
    assert "for engine_id, engine_items in engine_groups.items():" in source
    assert "del pipe" in source
    assert "torch.cuda.empty_cache()" in source


def test_provider_materializes_one_photo_and_three_mascot_candidates():
    worker = FakeWorker()
    result = provider(worker).generate_batch((
        StyleRoutedImageItem("real", "Tạo ảnh trâu thật", 10),
        StyleRoutedImageItem("cute3d", "Tạo ảnh trâu 3D cute", 20),
    ))

    assert len(worker.submissions) == 1
    assert len(result.candidates["real"]) == 1
    assert len(result.candidates["cute3d"]) == 3
    assert result.candidates["real"][0].artifact.model == "SG161222/RealVisXL_V5.0"
    assert {item.artifact.model for item in result.candidates["cute3d"]} == {
        "playgroundai/playground-v2.5-1024px-aesthetic"
    }
