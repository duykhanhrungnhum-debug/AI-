import hashlib
import json

from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission
from ai_agent.core.media_v2 import (
    IMAGE_GUIDANCE,
    IMAGE_MODEL,
    IMAGE_STEPS,
    RECAPTION_MODEL,
    ImageRequestV2,
    KaggleImageV2Provider,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"A" * 128


class FakeWorker:
    def __init__(self):
        self.source = ""

    def submit_script(self, **kwargs):
        self.source = kwargs["source"]
        return KaggleKernelSubmission("user", kwargs["slug"], 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def logs(self, slug):
        return ""

    def download_output_file(self, slug, filename):
        if filename == "image_v2_report.json":
            return json.dumps({
                "gpu_name": "Tesla T4",
                "items": {
                    "x": {
                        "filename": "x.png",
                        "prompt": "A faithful English description of one Vietnamese water buffalo.",
                        "sha256": hashlib.sha256(PNG).hexdigest(),
                        "elapsed_seconds": 1.25,
                    }
                },
            }).encode()
        if filename == "x.png":
            return PNG
        raise FileNotFoundError(filename)


def test_v2_is_one_recaption_plus_one_unified_image_model():
    worker = FakeWorker()
    provider = KaggleImageV2Provider(worker=worker, poll_interval=0)
    result = provider.generate_many((
        ImageRequestV2("x", "Tạo đúng một con trâu nước Việt Nam thật ngoài đồng", 7),
    ))[0]

    assert result.model == IMAGE_MODEL
    assert IMAGE_MODEL == "black-forest-labs/FLUX.2-klein-4B"
    assert RECAPTION_MODEL == "Qwen/Qwen3-1.7B"
    assert IMAGE_STEPS == 4
    assert IMAGE_GUIDANCE == 1.0
    assert result.data == PNG

    source = worker.source
    assert "Flux2KleinPipeline" in source
    assert "Preserve exactly the requested subject or species" in source
    assert "Output only the final English description" in source
    assert "num_inference_steps=int(CONFIG[\"steps\"])" in source
    assert "guidance_scale=float(CONFIG[\"guidance\"])" in source

    # Production V2 must not recreate the architecture that caused prior loops.
    banned = (
        "GroundingDINO",
        "Qwen3-VL",
        "vision_quality",
        "hard_gate",
        "candidate_count",
        "RealVisXL",
        "Playground",
        "SUBJECT_CLASS",
        "STYLE_CLASS",
        "anatomy contract",
    )
    for token in banned:
        assert token not in source


def test_natural_request_contract_has_no_species_table_or_json_planner():
    provider = KaggleImageV2Provider(worker=FakeWorker(), poll_interval=0)
    source = provider._build_worker_source((
        ImageRequestV2("a", "Vẽ một con sao la 3D cute đứng trong rừng", 1),
    ))
    assert "no labels, JSON, explanation, scoring, or commentary" in source
    assert "subject_class" not in source.casefold()
    assert "style_class" not in source.casefold()
    assert "species table" not in source.casefold()


def test_request_validation_is_small_and_deterministic():
    import pytest

    with pytest.raises(ValueError):
        ImageRequestV2("", "hello", 1)
    with pytest.raises(ValueError):
        ImageRequestV2("x", "", 1)
    with pytest.raises(ValueError):
        ImageRequestV2("x", "hello", -1)
    with pytest.raises(ValueError):
        ImageRequestV2("x", "hello", 1, width=1001)
