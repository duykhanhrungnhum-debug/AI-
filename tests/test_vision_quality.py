import hashlib
from io import BytesIO
import json

from PIL import Image

from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission
from ai_agent.core.vision_quality import HybridVisionQualityVerifier, KaggleVisionQualityVerifier, VisionQualityRequest, result_from_inline_review


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
                "defect_review_text": json.dumps({
                    "reject": False,
                    "critical_defects": [],
                    "benchmark_failures": [],
                    "uncertain_regions": [],
                    "summary": "no blocking defects",
                }),
                "review_text": json.dumps(review),
            })
        return json.dumps({
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "gpu_name": "Tesla T4",
            "attention_backend": "sdpa",
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
        "benchmark_match_score": 9.0,
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
        "benchmark_match_score": 9.0,
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
        "benchmark_match_score": 9.0,
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
    assert "Qwen/Qwen3-VL-8B-Instruct" in source
    assert "views = [" in source
    assert "five views of the SAME generated image" in source
    assert "wrong subject count" in source
    assert "malformed or fused hands/fingers/limbs" in source
    assert "HARD CLEAN-ASSET RULE" in source
    assert "all four corner crops" in source
    assert "every score including benchmark_match_score is >= 8" in source


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
        "benchmark_match_score": 9.0,
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
    assert 'image.crop((0, 0, half_w, half_h))' in source
    assert 'image.crop((half_w, 0, width, half_h))' in source
    assert 'image.crop((0, half_h, half_w, height))' in source
    assert 'image.crop((half_w, half_h, width, height))' in source
    assert 'structure_score' in source
    assert 'detail_score' in source
    assert 'aesthetic_score' in source
    assert 'composition_score' in source
    assert 'benchmark_match_score' in source
    assert 'user-approved benchmark target' in source
    compile(source, "<qwen3-vision-quality-worker>", "exec")


def test_vlm_quality_gate_rejects_low_benchmark_match_even_if_generic_quality_is_high():
    review = {
        "pass": True,
        "quality_score": 9.5,
        "prompt_match_score": 9.5,
        "structure_score": 9.5,
        "detail_score": 9.5,
        "aesthetic_score": 9.5,
        "composition_score": 9.5,
        "benchmark_match_score": 5.0,
        "subject_count": 1,
        "major_issues": [],
        "minor_issues": [],
        "summary": "technically polished but wrong benchmark style",
    }
    verifier = KaggleVisionQualityVerifier(
        worker=FakeWorker([("human-1", png_bytes(), review)]),
        poll_interval=0,
    )
    result = verifier.verify(request())
    assert result.passed is False
    assert result.benchmark_match_score == 5.0
    assert any("benchmark_match score below threshold" in issue for issue in result.major_issues)


class DefectWorker(FakeWorker):
    def download_output_file(self, slug, filename):
        assert filename == "vision_quality.json"
        items = []
        for item_id, image, review in self.reviews:
            review_image = KaggleVisionQualityVerifier._compact_review_image(image)
            items.append({
                "item_id": item_id,
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "review_image_sha256": hashlib.sha256(review_image).hexdigest(),
                "defect_review_text": json.dumps({
                    "reject": True,
                    "critical_defects": ["right hand has fused fingers"],
                    "benchmark_failures": [],
                    "uncertain_regions": [],
                    "summary": "blocking anatomy defect",
                }),
                "review_text": json.dumps(review),
            })
        return json.dumps({
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "gpu_name": "Tesla T4",
            "attention_backend": "sdpa",
            "items": items,
        }).encode()


def test_adversarial_defect_hunter_can_veto_high_scoring_scorer():
    review = {
        "pass": True,
        "quality_score": 9.8,
        "prompt_match_score": 9.8,
        "structure_score": 9.8,
        "detail_score": 9.8,
        "aesthetic_score": 9.8,
        "composition_score": 9.8,
        "benchmark_match_score": 9.8,
        "subject_count": 1,
        "major_issues": [],
        "minor_issues": [],
        "summary": "looks excellent",
    }
    verifier = KaggleVisionQualityVerifier(
        worker=DefectWorker([("human-1", png_bytes(), review)]),
        poll_interval=0,
    )
    result = verifier.verify(request())
    assert result.passed is False
    assert any("defect hunter:" in issue for issue in result.major_issues)
    assert any("defect_hunter_reject:True" == item for item in result.evidence)


def test_qwen3_8b_worker_source_compiles_with_single_pass_critic():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert "Qwen/Qwen3-VL-8B-Instruct" in source
    assert "load_in_4bit=True" in source
    assert "Act as scorer and defect hunter in this single pass" in source
    assert "uncertain_regions" in source
    assert "max_new_tokens=200" in source
    assert "defect_review_text = review_text" in source
    assert "run_review(defect_instruction)" not in source
    compile(source, "<qwen3-8b-single-pass-worker>", "exec")


def test_user_rejected_image_hash_vetoes_vlm_pass(monkeypatch):
    image = png_bytes()
    digest = hashlib.sha256(image).hexdigest()
    monkeypatch.setattr(
        "ai_agent.core.vision_quality.benchmark_manifest",
        lambda: {
            "profiles": {
                "human_photo_premium": {
                    "known_rejected_sha256": [digest],
                }
            }
        },
    )
    review = {
        "pass": True,
        "quality_score": 9.9,
        "prompt_match_score": 9.9,
        "structure_score": 9.9,
        "detail_score": 9.9,
        "aesthetic_score": 9.9,
        "composition_score": 9.9,
        "benchmark_match_score": 9.9,
        "subject_count": 1,
        "major_issues": [],
        "minor_issues": [],
        "summary": "model thinks it is excellent",
    }
    verifier = KaggleVisionQualityVerifier(
        worker=FakeWorker([("human-1", image, review)]),
        poll_interval=0,
    )
    result = verifier.verify(request(image=image))
    assert result.passed is False
    assert "image exactly matches a user-rejected quality example" in result.major_issues
    assert "user_rejected_feedback:True" in result.evidence


def test_qwen3_8b_worker_requires_current_bitsandbytes_for_4bit():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert "bitsandbytes>=0.46.1" in source


def test_qwen3_8b_worker_checks_bitsandbytes_import_before_model_load():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert "import bitsandbytes" in source
    assert source.index("import bitsandbytes") < source.index("Qwen3VLForConditionalGeneration.from_pretrained")


def test_qwen3_animal_photo_critic_rejects_overprocessed_photoshop_look():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((VisionQualityRequest(
        item_id="dog-photo",
        image=png_bytes(),
        prompt="one realistic golden retriever outdoors",
        profile="animal_photo_premium",
        rubric=("natural unretouched camera realism", "no HDR or advertising retouch"),
        expected_subject_count=1,
    ),))
    assert "HDR-like tone mapping" in source
    assert "excessive sharpening" in source
    assert "advertising-style retouch" in source
    assert "Photoshopped rather than naturally camera-captured" in source
    assert "Natural camera softness and small imperfections are desirable" in source


def test_single_pass_qa_keeps_strict_uncertainty_and_benchmark_fields():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert "critical_defects" in source
    assert "benchmark_failures" in source
    assert "uncertain_regions" in source
    assert "pass MUST be false" in source
    assert "benchmark_match_score" in source
    assert source.count("run_review(instruction)") == 1


def test_compact_vlm_review_schema_expands_without_losing_quality_fields():
    parsed = KaggleVisionQualityVerifier._parse_review(
        '{"p":true,"q":9.2,"m":9.1,"s":9.0,"d":8.8,"a":9.3,"c":9.0,"b":8.9,"n":1,"x":[],"f":[],"u":[],"i":["tiny issue"]}'
    )
    assert parsed["pass"] is True
    assert parsed["quality_score"] == 9.2
    assert parsed["prompt_match_score"] == 9.1
    assert parsed["structure_score"] == 9.0
    assert parsed["detail_score"] == 8.8
    assert parsed["aesthetic_score"] == 9.3
    assert parsed["composition_score"] == 9.0
    assert parsed["benchmark_match_score"] == 8.9
    assert parsed["subject_count"] == 1
    assert parsed["critical_defects"] == []
    assert parsed["benchmark_failures"] == []
    assert parsed["uncertain_regions"] == []
    assert parsed["minor_issues"] == ["tiny issue"]


def test_compact_worker_prompt_requires_json_only_and_short_issue_phrases():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert "Return ONLY one compact JSON object" in source
    assert "Use short issue phrases" in source
    assert "p=pass" in source
    assert "torch.inference_mode()" in source
    assert "use_cache=True" in source


def test_qwen3_worker_records_latency_breakdown_without_changing_scoring():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert "worker_started = time.perf_counter()" in source
    assert "model_load_started = time.perf_counter()" in source
    assert '"model_load_seconds"' in source
    assert '"review_seconds"' in source
    assert '"worker_total_seconds"' in source
    compile(source, "<qwen3-qa-timing-worker>", "exec")


def test_qwen3_worker_uses_sdpa_attention_without_changing_model_or_thresholds():
    verifier = KaggleVisionQualityVerifier(worker=FakeWorker([]), poll_interval=0)
    source = verifier._build_worker_source((request(),))
    assert verifier.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert verifier.min_quality_score == 8.0
    assert verifier.min_prompt_match_score == 8.0
    assert verifier.attention_backend == "sdpa"
    assert 'attn_implementation=CONFIG["attention_backend"]' in source
    assert '"attention_backend": "sdpa"' in source.lower()
    compile(source, "<qwen3-sdpa-worker>", "exec")


def test_qwen3_worker_rejects_unknown_attention_backend():
    import pytest
    with pytest.raises(ValueError, match="attention_backend"):
        KaggleVisionQualityVerifier(worker=FakeWorker([]), attention_backend="unknown")


class HybridFakeWorker:
    def __init__(self, image, fast_review, final_review=None):
        self.image = image
        self.fast_review = fast_review
        self.final_review = final_review or fast_review
        self.submissions = []
        self.current_slug = ""

    def submit_script(self, **kwargs):
        self.current_slug = kwargs["slug"]
        self.submissions.append(self.current_slug)
        return KaggleKernelSubmission("testuser", kwargs["slug"], 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def download_output_file(self, slug, filename):
        assert filename == "vision_quality.json"
        fast = slug.endswith("-fast")
        review = self.fast_review if fast else self.final_review
        model = "Qwen/Qwen3-VL-2B-Instruct" if fast else "Qwen/Qwen3-VL-8B-Instruct"
        review_image = KaggleVisionQualityVerifier._compact_review_image(self.image)
        return json.dumps({
            "model": model,
            "gpu_name": "Tesla T4",
            "attention_backend": "sdpa",
            "items": [{
                "item_id": "human-1",
                "image_sha256": hashlib.sha256(self.image).hexdigest(),
                "review_image_sha256": hashlib.sha256(review_image).hexdigest(),
                "defect_review_text": json.dumps({
                    "reject": False,
                    "critical_defects": [],
                    "benchmark_failures": [],
                    "uncertain_regions": [],
                }),
                "review_text": json.dumps(review),
            }],
        }).encode()


def _hybrid_review(score):
    return {
        "pass": True,
        "quality_score": score,
        "prompt_match_score": score,
        "structure_score": score,
        "detail_score": score,
        "aesthetic_score": score,
        "composition_score": score,
        "benchmark_match_score": score,
        "subject_count": 1,
        "major_issues": [],
        "minor_issues": [],
    }


def test_hybrid_quality_gate_accepts_clear_9_image_without_loading_8b():
    image = png_bytes()
    worker = HybridFakeWorker(image, _hybrid_review(9.0))
    verifier = HybridVisionQualityVerifier(worker=worker, poll_interval=0)

    result = verifier.verify(request(image=image))

    assert result.passed is True
    assert worker.submissions == ["ai-agent-hybrid-vision-quality-fast"]
    assert "hybrid_stage:fast_clear_pass" in result.evidence
    assert "hybrid_fast_model:Qwen/Qwen3-VL-2B-Instruct" in result.evidence


def test_hybrid_quality_gate_uses_8b_only_for_borderline_fast_review():
    image = png_bytes()
    fast_review = _hybrid_review(8.6)
    fast_review["pass"] = False
    worker = HybridFakeWorker(
        image,
        fast_review,
        _hybrid_review(9.6),
    )
    verifier = HybridVisionQualityVerifier(worker=worker, poll_interval=0)

    result = verifier.verify(request(image=image))

    assert result.passed is True
    assert worker.submissions == [
        "ai-agent-hybrid-vision-quality-fast",
        "ai-agent-hybrid-vision-quality-final",
    ]
    assert "hybrid_stage:final_fallback" in result.evidence
    assert "hybrid_final_model:Qwen/Qwen3-VL-8B-Instruct" in result.evidence


def test_hybrid_quality_gate_fast_rejects_clearly_low_image_without_8b():
    image = png_bytes()
    review = _hybrid_review(8.0)
    review["pass"] = False
    worker = HybridFakeWorker(image, review, _hybrid_review(9.8))
    verifier = HybridVisionQualityVerifier(worker=worker, poll_interval=0)

    result = verifier.verify(request(image=image))

    assert result.passed is False
    assert worker.submissions == ["ai-agent-hybrid-vision-quality-fast"]
    assert "hybrid_stage:fast_reject" in result.evidence
    assert any("below threshold" in issue for issue in result.major_issues)


def test_hybrid_quality_gate_validates_confidence_band():
    import pytest
    with pytest.raises(ValueError, match="fast_reject_score"):
        HybridVisionQualityVerifier(
            worker=HybridFakeWorker(png_bytes(), _hybrid_review(9.0)),
            fast_reject_score=9.0,
        )



def test_inline_review_parser_enforces_nine_out_of_ten():
    image = png_bytes()
    req = request(image=image)
    review = json.dumps({
        "p": True,
        "q": 9,
        "m": 9,
        "s": 9,
        "d": 9,
        "a": 9,
        "c": 9,
        "b": 9,
        "n": 1,
        "x": [],
        "f": [],
        "u": [],
        "i": [],
    })
    result = result_from_inline_review(
        req,
        review,
        model="Qwen/Qwen3-VL-2B-Instruct",
        evidence=("inline_vlm:True",),
        min_quality_score=9.0,
        min_prompt_match_score=9.0,
    )
    assert result.passed is True
    assert result.quality_score == 9.0
    assert "vlm_model:Qwen/Qwen3-VL-2B-Instruct" in result.evidence


def test_inline_review_parser_rejects_visible_defect_even_with_high_scores():
    image = png_bytes()
    req = request(image=image)
    review = json.dumps({
        "p": False,
        "q": 9,
        "m": 9,
        "s": 9,
        "d": 9,
        "a": 9,
        "c": 9,
        "b": 9,
        "n": 1,
        "x": ["fused paw"],
        "f": [],
        "u": [],
        "i": [],
    })
    result = result_from_inline_review(
        req,
        review,
        model="Qwen/Qwen3-VL-2B-Instruct",
    )
    assert result.passed is False
    assert any("fused paw" in issue for issue in result.major_issues)
