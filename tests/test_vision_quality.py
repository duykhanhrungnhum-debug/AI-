import hashlib
from io import BytesIO
import json

from PIL import Image

from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission
from ai_agent.core.vision_quality import KaggleVisionQualityVerifier, VisionQualityRequest


class FakeWorker:
    def __init__(self, reviews):
        self.reviews = reviews
        self.submitted = None

    def submit_script(self, **kwargs):
        self.submitted = kwargs
        return KaggleKernelSubmission("testuser", kwargs["slug"], 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def download_output_file(self, slug, filename):
        assert filename == "vision_quality.json"
        items = []
        for item_id, image, review in self.reviews:
            review_image = KaggleVisionQualityVerifier._compact_review_image(image)
            items.append({
                "item_id": item_id,
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "review_image_sha256": hashlib.sha256(review_image).hexdigest(),
                "review_text": json.dumps(review),
            })
        return json.dumps({
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "gpu_name": "Tesla T4",
            "items": items,
        }).encode()


def png_bytes(size=(64, 64), color=(120, 80, 40)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def request(item_id="human-1", image=None, expected=1):
    image = image or png_bytes()
    return VisionQualityRequest(
        item_id=item_id,
        image=image,
        prompt="one premium human portrait",
        profile="human_photo_premium",
        rubric=("correct anatomy", "no artifacts"),
        expected_subject_count=expected,
    )


def test_vlm_quality_gate_passes_only_clean_high_scoring_image():
    review = {
        "pass": True,
        "quality_score": 9.0,
        "prompt_match_score": 9.5,
        "structure_score": 9.0,
        "detail_score": 9.0,
        "aesthetic_score": 9.0,
        "composition_score": 9.0,
        "subject_count": 1,
        "major_issues": [],
        "minor_issues": ["tiny background text"],
        "summary": "production ready",
    }
    verifier = KaggleVisionQualityVerifier(
        worker=FakeWorker([("human-1", png_bytes(), review)]),
        poll_interval=0,
    )
    result = verifier.verify(request())
    assert result.passed is True
    assert result.quality_score == 9.0
    assert result.subject_count == 1
    assert result.major_issues == ()
    assert "gpu:Tesla T4" in result.evidence


def test_vlm_quality_gate_rejects_major_visual_issue_even_with_high_scores():
    review = {
        "pass": False,
        "quality_score": 9.0,
        "prompt_match_score": 9.0,
        "structure_score": 9.0,
        "detail_score": 9.0,
        "aesthetic_score": 9.0,
        "composition_score": 9.0,
        "subject_count": 1,
        "major_issues": ["right hand has fused fingers"],
        "minor_issues": [],
        "summary": "hand defect",
    }
    verifier = KaggleVisionQualityVerifier(
        worker=FakeWorker([("human-1", png_bytes(), review)]),
        poll_interval=0,
    )
    result = verifier.verify(request())
    assert result.passed is False
    assert "right hand has fused fingers" in result.major_issues


def test_vlm_quality_gate_rejects_wrong_subject_count():
    review = {
        "pass": True,
        "quality_score": 9.0,
        "prompt_match_score": 9.0,
        "structure_score": 9.0,
        "detail_score": 9.0,
        "aesthetic_score": 9.0,
        "composition_score": 9.0,
        "subject_count": 3,
        "major_issues": [],
        "minor_issues": [],
        "summary": "three subjects",
    }
    verifier = KaggleVisionQualityVerifier(
        worker=FakeWorker([("mascot-1", png_bytes(color=(220, 80, 40)), review)]),
        poll_interval=0,
    )
    result = verifier.verify(VisionQualityRequest(
        item_id="mascot-1",
        image=png_bytes(color=(220, 80, 40)),
        prompt="exactly one crab mascot",
        profile="mascot_premium",
        rubric=("exactly one subject", "clean geometry"),
        expected_subject_count=1,
    ))
    assert result.passed is False
    assert any("subject count mismatch" in issue for issue in result.major_issues)


def test_vlm_review_parser_accepts_fenced_json():
    raw = '```json\n{"pass": true, "quality_score": 8.5, "prompt_match_score": 9, "subject_count": 1, "major_issues": [], "minor_issues": []}\n```'
    parsed = KaggleVisionQualityVerifier._parse_review(raw)
    assert parsed["pass"] is True
    assert parsed["quality_score"] == 8.5


def test_worker_source_loads_open_multimodal_model_and_strict_rubric():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert "Qwen3VLForConditionalGeneration" in source
    assert "Qwen/Qwen3-VL-4B-Instruct" in source
    assert "views = [" in source
    assert "five views of the SAME generated image" in source
    assert "wrong subject count" in source
    assert "malformed or fused hands/fingers/limbs" in source
    assert "quality_score >= 8" in source


def test_vlm_review_image_compaction_stays_far_below_kaggle_source_limit():
    large = png_bytes(size=(1024, 1024), color=(123, 87, 45))
    compact = KaggleVisionQualityVerifier._compact_review_image(large)

    assert compact.startswith(b"\xff\xd8")
    assert len(compact) < 300_000

    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((
        request(item_id="one", image=large),
        request(item_id="two", image=large),
    ))
    assert len(source.encode("utf-8")) < 900_000


def test_worker_source_is_valid_python_after_prompt_embedding():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    compile(source, "<vision-quality-worker>", "exec")


def test_vlm_quality_gate_rejects_low_structure_even_when_overall_score_is_high():
    review = {
        "pass": True,
        "quality_score": 9.2,
        "prompt_match_score": 9.1,
        "structure_score": 6.5,
        "detail_score": 9.0,
        "aesthetic_score": 9.0,
        "composition_score": 9.0,
        "subject_count": 1,
        "major_issues": [],
        "minor_issues": [],
        "summary": "good overall but hand structure is weak",
    }
    verifier = KaggleVisionQualityVerifier(
        worker=FakeWorker([("human-1", png_bytes(), review)]),
        poll_interval=0,
    )
    result = verifier.verify(request())
    assert result.passed is False
    assert any("structure score below threshold" in issue for issue in result.major_issues)


def test_qwen3_worker_uses_multiview_and_hard_component_gates():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert 'image.crop((0, 0, width, max(1, height // 2)))' in source
    assert 'image.crop((0, height // 2, width, height))' in source
    assert 'structure_score' in source
    assert 'detail_score' in source
    assert 'aesthetic_score' in source
    assert 'composition_score' in source
    compile(source, "<qwen3-vision-quality-worker>", "exec")
