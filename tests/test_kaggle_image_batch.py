import hashlib
from io import BytesIO
import json
import struct
import zlib
import zipfile

import pytest

from ai_agent.core.kaggle_image_batch import (
    KaggleBatchImageProvider,
    KaggleSemanticImageBatchProvider,
    SceneImageRequest,
    SemanticImageBatchItem,
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
            "prompt_alignment_score": 0.35,
            "visual_quality_margin": 0.05,
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
                "prompt_alignment_score": 0.34,
                "visual_quality_margin": 0.05,
            })
        return json.dumps({
            "model": "SG161222/RealVisXL_V4.0",
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
    assert source.count("AutoPipelineForText2Image.from_pretrained") == 1
    assert "SG161222/RealVisXL_V4.0" in source
    assert "DPMSolverMultistepScheduler" in source
    assert "use_karras_sigmas=True" in source
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


def test_reference_worker_uses_sdxl_face_adapter_and_cpu_offload():
    provider = KaggleBatchImageProvider(worker=FakeWorker(), poll_interval=0)
    source = provider._build_worker_source(
        (SceneImageRequest("scene-1", "same character in a library", seed=1),),
        reference_image=b"reference-image-bytes",
        reference_scale=0.75,
    )

    assert "ip-adapter-plus-face_sdxl_vit-h.safetensors" in source
    assert '"sdxl_models" if is_sdxl else "models"' in source
    assert "pipe.enable_model_cpu_offload()" in source
    assert "pipe.enable_vae_slicing()" in source


def test_safety_checker_block_is_reported():
    provider = KaggleBatchImageProvider(worker=FakeWorker(), poll_interval=0)
    request = SceneImageRequest("scene-1", "safe portrait", seed=10)
    image = png_bytes()
    digest = hashlib.sha256(image).hexdigest()
    issues = provider._verify_scene(
        request,
        {
            "image_sha256": digest,
            "pixel_std": 40.0,
            "safety_blocked": True,
            "prompt_alignment_score": 0.35,
            "visual_quality_margin": 0.05,
        },
        image,
        digest,
    )

    assert "safety checker blocked generated image" in issues


def test_retry_refines_prompt_from_identity_and_duplicate_failures():
    request = SceneImageRequest(
        "scene-1",
        "hero in an old library",
        negative_prompt="text, watermark",
        seed=7,
    )

    refined = KaggleBatchImageProvider._refined_retry_request(
        request,
        (
            "identity similarity below threshold: 0.30 < 0.40",
            "duplicate image content",
        ),
        round_index=1,
    )

    assert refined.seed == 8
    assert "STRICT IDENTITY LOCK" in refined.prompt
    assert "different pose, staging or camera framing" in refined.prompt
    assert "face drift" in refined.negative_prompt
    assert "same composition as previous attempt" in refined.negative_prompt


def test_retry_turns_safety_failure_into_non_graphic_refinement():
    request = SceneImageRequest("scene-1", "story scene", seed=3)

    refined = KaggleBatchImageProvider._refined_retry_request(
        request,
        ("safety checker blocked generated image",),
        round_index=2,
    )

    assert refined.seed == 5
    assert "non-graphic and safe" in refined.prompt


def test_prompt_alignment_below_threshold_is_rejected():
    provider = KaggleBatchImageProvider(worker=FakeWorker(), poll_interval=0)
    request = SceneImageRequest("scene-1", "elderly scholar in an old library", seed=10)
    image = png_bytes()
    digest = hashlib.sha256(image).hexdigest()

    issues = provider._verify_scene(
        request,
        {
            "image_sha256": digest,
            "pixel_std": 40.0,
            "safety_blocked": False,
            "prompt_alignment_score": 0.10,
            "visual_quality_margin": 0.05,
        },
        image,
        digest,
    )

    assert any("prompt alignment below threshold" in issue for issue in issues)


def test_worker_computes_prompt_alignment_with_clip_text_and_image():
    provider = KaggleBatchImageProvider(worker=FakeWorker(), poll_interval=0)
    source = provider._build_worker_source(
        (SceneImageRequest("scene-1", "elderly scholar in an old library", seed=1),),
        reference_image=b"reference-image-bytes",
        reference_scale=0.75,
    )

    assert "CLIPVisionModelWithProjection" in source
    assert "CLIPTextModelWithProjection" in source
    assert ".image_embeds[0].float()" in source
    assert ".text_embeds[0].float()" in source
    assert "prompt_alignment_score" in source


def test_worker_uses_shared_clip_projection_space_for_dot_product():
    provider = KaggleBatchImageProvider(worker=FakeWorker(), poll_interval=0)
    source = provider._build_worker_source(
        (SceneImageRequest("scene-1", "elderly scholar in an old library", seed=1),),
        reference_image=b"reference-image-bytes",
        reference_scale=0.75,
    )

    assert "clip_vision(pixel_values=values).image_embeds[0].float()" in source
    assert "clip_text(**values).text_embeds[0].float()" in source


def test_visual_quality_margin_below_threshold_is_rejected():
    provider = KaggleBatchImageProvider(worker=FakeWorker(), poll_interval=0)
    request = SceneImageRequest("scene-1", "elderly scholar at a desk", seed=10)
    image = png_bytes()
    digest = hashlib.sha256(image).hexdigest()

    issues = provider._verify_scene(
        request,
        {
            "image_sha256": digest,
            "pixel_std": 40.0,
            "safety_blocked": False,
            "prompt_alignment_score": 0.35,
            "visual_quality_margin": -0.01,
        },
        image,
        digest,
    )

    assert any("visual quality margin below threshold" in issue for issue in issues)


def test_worker_scores_visual_defects_separately_from_prompt_alignment():
    provider = KaggleBatchImageProvider(worker=FakeWorker(), poll_interval=0)
    source = provider._build_worker_source(
        (SceneImageRequest("scene-1", "elderly scholar at a desk", seed=1),),
        reference_image=b"reference-image-bytes",
        reference_scale=0.75,
    )

    assert "quality_good_text" in source
    assert "quality_bad_texts" in source
    assert "visual_defect_score" in source
    assert "visual_quality_margin" in source
    assert "deformed hands" in source
    assert "warped books" in source



class SemanticFakeWorker:
    def __init__(self):
        self.submissions = []
        self.images = {
            "real": png_bytes(1024, 1024, b"R"),
            "cute3d": png_bytes(1024, 1024, b"C"),
        }

    def submit_script(self, **kwargs):
        self.submissions.append(kwargs)
        return KaggleKernelSubmission("testuser", kwargs["slug"], 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def logs(self, slug):
        return ""

    def download_output_file(self, slug, filename):
        if filename.endswith(".png"):
            return self.images[filename[:-4]]
        if filename != "batch_report.json":
            raise FileNotFoundError(filename)
        reports = {}
        for item_id, profile, subject, style in (
            ("real", "animal_photo_premium", "animal", "photo"),
            ("cute3d", "mascot_premium", "animal", "3d"),
        ):
            image = self.images[item_id]
            reports[item_id] = {
                "prompt": f"one {style} water buffalo",
                "negative_prompt": "bad anatomy",
                "subject_class": subject,
                "style_class": style,
                "subject_name": "water buffalo",
                "semantic_profile": profile,
                "planner_raw": "semantic output",
                "planner_model": "Qwen/Qwen3-0.6B",
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "model": "SG161222/RealVisXL_V5.0",
                "model_variant": "fp16",
                "width": 1024,
                "height": 1024,
                "review_text": '{"p":true,"q":9,"m":9,"s":9,"d":9,"a":9,"c":9,"b":9,"n":1,"x":[],"f":[],"u":[],"i":[]}',
                "hard_gate": {
                    "passed": True,
                    "subject_name": "water buffalo",
                    "requires_full_body": True,
                    "checks": {
                        "subject_ok": True,
                        "style_ok": True,
                        "count_ok": True,
                        "framing_ok": True,
                        "anatomy_ok": True,
                        "no_text_logo": True,
                    },
                    "issues": [],
                },
            }
        return json.dumps({
            "gpu_name": "Tesla T4",
            "model": "SG161222/RealVisXL_V5.0",
            "model_variant": "fp16",
            "items": reports,
        }).encode()


def semantic_provider(worker=None):
    return KaggleSemanticImageBatchProvider(
        worker=worker or SemanticFakeWorker(),
        poll_interval=0,
        profile_positive_constraints={
            "human_photo_premium": "human quality",
            "animal_photo_premium": "animal photo quality",
            "mascot_premium": "3D mascot quality",
            "general_premium": "general quality",
        },
        profile_negative_constraints={
            "human_photo_premium": "bad hands",
            "animal_photo_premium": "bad animal anatomy",
            "mascot_premium": "bad 3D anatomy",
            "general_premium": "bad geometry",
        },
        profile_rubrics={
            "human_photo_premium": ("human anatomy",),
            "animal_photo_premium": ("animal anatomy",),
            "mascot_premium": ("3D mascot quality",),
            "general_premium": ("general quality",),
        },
        profile_dimensions={
            "human_photo_premium": (832, 1216),
            "animal_photo_premium": (1024, 1024),
            "mascot_premium": (1024, 1024),
            "general_premium": (1024, 1024),
        },
    )


def test_semantic_batch_loads_planner_image_model_and_vlm_once_for_all_items():
    provider = semantic_provider()
    items = (
        SemanticImageBatchItem("real", "Tạo ảnh một con voi thật", 10),
        SemanticImageBatchItem("cute3d", "Tạo ảnh một con voi 3D cute", 20),
    )

    source = provider._build_worker_source(items)
    compile(source, "<semantic-batch-worker>", "exec")

    assert source.count('AutoPipelineForText2Image.from_pretrained(CONFIG["model"]') == 1
    assert source.count('Qwen3VLForConditionalGeneration.from_pretrained') == 1
    assert "without a fixed species list" in source
    assert "SUBJECT_CLASS" in source
    assert "STYLE_CLASS" in source
    assert "SUBJECT_NAME" in source
    assert "is_placeholder_prompt" in source
    assert "EXPECTED EXACT SUBJECT/SPECIES/ENTITY" in source
    assert "no_text_logo" in source
    assert "for item in compiled_items:" in source


def test_semantic_batch_returns_two_styles_from_one_model_and_one_submission():
    worker = SemanticFakeWorker()
    provider = semantic_provider(worker)
    result = provider.generate_batch((
        SemanticImageBatchItem("real", "Tạo ảnh trâu thật", 10),
        SemanticImageBatchItem("cute3d", "Tạo ảnh trâu 3D cute", 20),
    ))

    assert len(worker.submissions) == 1
    assert set(result.artifacts) == {"real", "cute3d"}
    assert {artifact.model for artifact in result.artifacts.values()} == {"SG161222/RealVisXL_V5.0"}
    assert result.reports["real"]["semantic_profile"] == "animal_photo_premium"
    assert result.reports["cute3d"]["semantic_profile"] == "mascot_premium"


def test_semantic_batch_rejects_mismatched_profile_maps():
    with pytest.raises(ValueError, match="matching keys"):
        KaggleSemanticImageBatchProvider(
            worker=SemanticFakeWorker(),
            profile_positive_constraints={"animal_photo_premium": "good"},
            profile_negative_constraints={"general_premium": "bad"},
            profile_rubrics={"animal_photo_premium": ("quality",)},
            profile_dimensions={"animal_photo_premium": (1024, 1024)},
        )



def test_semantic_worker_rejects_schema_placeholder_prompt():
    provider = semantic_provider()
    source = provider._build_worker_source((
        SemanticImageBatchItem("real", "Tạo ảnh một con trâu nước thật", 10),
    ))
    compile(source, "<semantic-batch-worker>", "exec")

    assert '"english image description"' in source
    assert "not is_placeholder_prompt(parsed[\"prompt\"])" in source
    assert "valid_subject_name" in source
    assert "subject_name_matches_prompt" in source
    assert "SUBJECT_NAME: <exact English common subject/entity name>" in source


def test_semantic_worker_hard_gate_checks_exact_subject_and_watermark():
    provider = semantic_provider()
    source = provider._build_worker_source((
        SemanticImageBatchItem("cute3d", "Tạo ảnh trâu nước 3D cute toàn thân", 20),
    ))
    compile(source, "<semantic-batch-worker>", "exec")

    assert "the main subject must be exactly the expected species/entity" in source
    assert "logo, watermark, emblem, signature" in source
    assert '"single_model_two_checks": True' in source
    assert '"subject_ok"' in source
    assert '"no_text_logo"' in source
    assert "dedicated watermark/logo/text detector" in source.casefold()
    assert '"branding_gate"' in source
    assert "photorealistic livestock render" in source
    assert '{"clean":false,"findings":["bottom-right logo or text"]}' not in source
    assert "normalize_branding_findings" in source
    assert 'candidate_count = 3 if item["semantic_profile"] == "mascot_premium" else 1' in source
    assert '"candidate_summaries"' in source
    assert "safe_border_crop" in source
    assert "render_width" in source
    assert "central 78 percent" in source
