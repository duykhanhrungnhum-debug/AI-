import hashlib
import json

import pytest

from ai_agent.core.image_model import ImageGenerationRequest
from ai_agent.core.kaggle_image import KaggleImageProvider
from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission


class FakeWorker:
    def __init__(self):
        self.submitted = None
        self.statuses = [
            KaggleKernelStatus("RUNNING"),
            KaggleKernelStatus("COMPLETE"),
        ]
        self.image = b"PNG-BYTES"

    def submit_script(self, **kwargs):
        self.submitted = kwargs
        return KaggleKernelSubmission(
            owner="testuser",
            slug=kwargs["slug"],
            version_number=1,
            kernel_id=1,
            url=None,
        )

    def status(self, slug):
        return self.statuses.pop(0)

    def download_output_file(self, slug, filename):
        if filename == "generated.png":
            return self.image
        if filename == "image_report.json":
            return json.dumps({
                "image_sha256": hashlib.sha256(self.image).hexdigest(),
                "width": 512,
                "height": 512,
                "seed": 42,
                "model": "playgroundai/playground-v2.5-1024px-aesthetic",
                "gpu_name": "Tesla T4",
                "prompt_alignment_score": 0.35,
                "visual_quality_margin": 0.05,
                "visual_defect_score": 0.10,
            }).encode()
        raise FileNotFoundError(filename)


def test_kaggle_image_provider_runs_open_model_and_verifies_evidence(monkeypatch):
    worker = FakeWorker()
    provider = KaggleImageProvider(
        worker=worker,
        poll_interval=0,
        max_poll_attempts=3,
    )

    artifact = provider.generate(ImageGenerationRequest(
        "cinematic haunted house at night",
        negative_prompt="text, watermark",
        width=512,
        height=512,
        seed=42,
    ))

    assert worker.submitted["enable_internet"] is True
    assert worker.submitted["is_private"] is True
    assert worker.submitted["title"] == "Ai Agent Image Worker"
    assert "AutoPipelineForText2Image.from_pretrained" in worker.submitted["source"]
    assert "playgroundai/playground-v2.5-1024px-aesthetic" in worker.submitted["source"]
    assert "EDMDPMSolverMultistepScheduler" in worker.submitted["source"]
    assert '"scheduler": "edm_dpm"' in worker.submitted["source"]
    assert "visual_quality_margin" in worker.submitted["source"]
    assert artifact.data == b"PNG-BYTES"
    assert artifact.provider == "kaggle-gpu-local-model"
    assert artifact.mime_type == "image/png"
    assert "gpu:Tesla T4" in artifact.evidence
    assert any(item.startswith("image_sha256:") for item in artifact.evidence)


def test_kaggle_image_provider_rejects_non_divisible_dimensions():
    provider = KaggleImageProvider(worker=FakeWorker(), poll_interval=0)

    with pytest.raises(ValueError, match="divisible by 8"):
        provider.generate(ImageGenerationRequest("scene", width=513, height=512))


def test_kaggle_image_provider_rejects_hash_mismatch():
    worker = FakeWorker()

    def bad_download(slug, filename):
        if filename == "generated.png":
            return b"BAD-IMAGE"
        return json.dumps({
            "image_sha256": "wrong",
            "width": 512,
            "height": 512,
            "seed": 42,
            "model": "playgroundai/playground-v2.5-1024px-aesthetic",
            "gpu_name": "Tesla T4",
                "prompt_alignment_score": 0.35,
                "visual_quality_margin": 0.05,
                "visual_defect_score": 0.10,
        }).encode()

    worker.download_output_file = bad_download
    provider = KaggleImageProvider(worker=worker, poll_interval=0)

    with pytest.raises(ValueError, match="hash"):
        provider.generate(ImageGenerationRequest("scene", width=512, height=512, seed=42))


def test_kaggle_image_provider_rejects_low_visual_quality_margin():
    worker = FakeWorker()

    def bad_quality(slug, filename):
        if filename == "generated.png":
            return worker.image
        return json.dumps({
            "image_sha256": hashlib.sha256(worker.image).hexdigest(),
            "width": 512,
            "height": 512,
            "seed": 42,
            "model": "playgroundai/playground-v2.5-1024px-aesthetic",
            "gpu_name": "Tesla T4",
            "prompt_alignment_score": 0.35,
            "visual_quality_margin": -0.02,
            "visual_defect_score": 0.30,
        }).encode()

    worker.download_output_file = bad_quality
    provider = KaggleImageProvider(worker=worker, poll_interval=0)

    with pytest.raises(ValueError, match="visual quality margin"):
        provider.generate(ImageGenerationRequest("scene", width=512, height=512, seed=42))


def test_kaggle_image_provider_embeds_profile_specific_quality_critic():
    provider = KaggleImageProvider(
        worker=FakeWorker(),
        poll_interval=0,
        quality_good_text="clean mascot geometry",
        quality_bad_texts=("multiple repeated mascots", "malformed claws"),
    )

    source = provider._build_worker_source(
        ImageGenerationRequest("one crab mascot", width=512, height=512, seed=42)
    )

    assert "clean mascot geometry" in source
    assert "multiple repeated mascots" in source
    assert "malformed claws" in source


def test_kaggle_image_provider_rejects_unknown_scheduler():
    with pytest.raises(ValueError, match="scheduler"):
        KaggleImageProvider(worker=FakeWorker(), scheduler="unknown")


def test_visual_margin_can_be_advisory_when_external_vlm_gate_is_mandatory():
    worker = FakeWorker()

    def low_margin(slug, filename):
        if filename == "generated.png":
            return worker.image
        return json.dumps({
            "image_sha256": hashlib.sha256(worker.image).hexdigest(),
            "width": 512,
            "height": 512,
            "seed": 42,
            "model": "playgroundai/playground-v2.5-1024px-aesthetic",
            "gpu_name": "Tesla T4",
            "prompt_alignment_score": 0.35,
            "visual_quality_margin": 0.001,
            "visual_defect_score": 0.30,
        }).encode()

    worker.download_output_file = low_margin
    provider = KaggleImageProvider(
        worker=worker,
        poll_interval=0,
        enforce_visual_quality_margin=False,
    )

    artifact = provider.generate(
        ImageGenerationRequest("clean single mascot", width=512, height=512, seed=42)
    )

    assert "visual_quality_margin:0.001000" in artifact.evidence
    assert "visual_quality_margin_enforced:False" in artifact.evidence


def test_clip_precheck_can_be_disabled_when_external_vlm_is_authoritative():
    worker = FakeWorker()

    def no_clip_report(slug, filename):
        if filename == "generated.png":
            return worker.image
        return json.dumps({
            "image_sha256": hashlib.sha256(worker.image).hexdigest(),
            "width": 512,
            "height": 512,
            "seed": 42,
            "model": "playgroundai/playground-v2.5-1024px-aesthetic",
            "gpu_name": "Tesla T4",
            "prompt_alignment_score": None,
            "visual_quality_margin": None,
            "visual_defect_score": None,
            "clip_precheck": False,
        }).encode()

    worker.download_output_file = no_clip_report
    provider = KaggleImageProvider(
        worker=worker,
        poll_interval=0,
        enable_clip_precheck=False,
    )
    artifact = provider.generate(
        ImageGenerationRequest("clean single mascot", width=512, height=512, seed=42)
    )

    assert "clip_precheck:False" in artifact.evidence
    source = worker.submitted["source"]
    assert 'if CONFIG["enable_clip_precheck"]:' in source
    assert '"enable_clip_precheck": false' in source.lower()


def test_image_worker_can_keep_pipeline_resident_on_gpu():
    provider = KaggleImageProvider(
        worker=FakeWorker(),
        poll_interval=0,
        enable_clip_precheck=False,
        enable_cpu_offload=False,
    )
    source = provider._build_worker_source(
        ImageGenerationRequest("one dog", width=512, height=512, seed=42)
    )

    assert '"enable_cpu_offload": false' in source.lower()
    assert 'if CONFIG["enable_cpu_offload"]:' in source
    assert 'pipe = pipe.to("cuda")' in source



def test_inline_vlm_can_share_image_worker_gpu_session():
    provider = KaggleImageProvider(
        worker=FakeWorker(),
        poll_interval=0,
        enable_clip_precheck=False,
        enable_cpu_offload=False,
        enable_inline_vlm=True,
        inline_vlm_model="Qwen/Qwen3-VL-2B-Instruct",
        inline_vlm_profile="animal_photo_premium",
        inline_vlm_rubric=("correct canine anatomy", "natural camera realism"),
        inline_vlm_expected_subject_count=1,
        inline_vlm_min_score=9.0,
    )

    source = provider._build_worker_source(
        ImageGenerationRequest("one real dog", width=512, height=512, seed=42)
    )

    assert '"enable_inline_vlm": true' in source.lower()
    assert "Qwen/Qwen3-VL-2B-Instruct" in source
    assert "Qwen3VLForConditionalGeneration.from_pretrained" in source
    assert "del pipe" in source
    assert "torch.cuda.empty_cache()" in source
    assert "inline vision-quality model returned empty review" in source


def test_inline_vlm_requires_profile_and_rubric():
    with pytest.raises(ValueError, match="inline_vlm_profile"):
        KaggleImageProvider(
            worker=FakeWorker(),
            enable_inline_vlm=True,
            inline_vlm_rubric=("quality",),
        )

    with pytest.raises(ValueError, match="inline_vlm_rubric"):
        KaggleImageProvider(
            worker=FakeWorker(),
            enable_inline_vlm=True,
            inline_vlm_profile="animal_photo_premium",
        )



def test_image_worker_enables_hf_xet_high_performance():
    provider = KaggleImageProvider(
        worker=FakeWorker(),
        poll_interval=0,
        enable_clip_precheck=False,
        enable_cpu_offload=False,
    )
    source = provider._build_worker_source(
        ImageGenerationRequest("one dog", width=512, height=512, seed=42)
    )

    assert 'HF_XET_HIGH_PERFORMANCE' in source
    assert 'HF_HUB_DISABLE_TELEMETRY' in source



def test_inline_planner_and_vlm_share_one_generated_worker():
    provider = KaggleImageProvider(
        worker=FakeWorker(),
        poll_interval=0,
        enable_clip_precheck=False,
        enable_cpu_offload=False,
        enable_inline_planner=True,
        inline_planner_raw_command="Tạo ảnh một chú chó Golden Retriever thật ngoài trời",
        inline_planner_positive_constraints="natural realistic dog photo with correct anatomy",
        enable_inline_vlm=True,
        inline_vlm_model="Qwen/Qwen3-VL-2B-Instruct",
        inline_vlm_profile="animal_photo_premium",
        inline_vlm_rubric=("correct canine anatomy", "natural camera realism"),
        inline_vlm_expected_subject_count=1,
        inline_vlm_min_score=9.0,
    )

    source = provider._build_worker_source(
        ImageGenerationRequest(
            "placeholder prompt",
            negative_prompt="bad anatomy",
            width=512,
            height=512,
            seed=42,
        )
    )

    compile(source, "<generated-image-worker>", "exec")
    assert "Qwen/Qwen3-0.6B" in source
    assert "Qwen/Qwen3-1.7B" in source
    assert "inline media planner did not produce usable semantic routing" in source
    assert source.index("compile_prompt(") < source.index("AutoPipelineForText2Image.from_pretrained")
    assert source.index("AutoPipelineForText2Image.from_pretrained") < source.index(
        "Qwen3VLForConditionalGeneration.from_pretrained"
    )


def test_inline_planner_requires_raw_command_and_constraints():
    with pytest.raises(ValueError, match="inline_planner_raw_command"):
        KaggleImageProvider(
            worker=FakeWorker(),
            enable_inline_planner=True,
            inline_planner_positive_constraints="quality",
        )

    with pytest.raises(ValueError, match="inline_planner_positive_constraints"):
        KaggleImageProvider(
            worker=FakeWorker(),
            enable_inline_planner=True,
            inline_planner_raw_command="dog outside",
        )



def test_image_worker_can_request_native_fp16_variant():
    provider = KaggleImageProvider(
        worker=FakeWorker(),
        model="SG161222/RealVisXL_V5.0",
        model_variant="fp16",
        poll_interval=0,
        enable_clip_precheck=False,
        enable_cpu_offload=False,
    )
    source = provider._build_worker_source(
        ImageGenerationRequest("one dog", width=512, height=512, seed=42)
    )

    compile(source, "<generated-image-worker>", "exec")
    assert '"model_variant": "fp16"' in source
    assert 'pipe_kwargs["variant"] = CONFIG["model_variant"]' in source






def test_inline_planner_semantically_routes_unlisted_animal_styles():
    provider = KaggleImageProvider(
        worker=FakeWorker(),
        model="SG161222/RealVisXL_V5.0",
        model_variant="fp16",
        poll_interval=0,
        enable_clip_precheck=False,
        enable_cpu_offload=False,
        enable_inline_planner=True,
        inline_planner_raw_command="Tạo ảnh một con tê giác 3D cute",
        inline_planner_positive_constraints="fallback quality",
        inline_profile_positive_constraints={
            "animal_photo_premium": "natural animal photo",
            "mascot_premium": "premium 3D animal mascot",
            "human_photo_premium": "natural human photo",
            "general_premium": "general premium image",
        },
        inline_profile_negative_constraints={
            "animal_photo_premium": "no CGI",
            "mascot_premium": "no broken anatomy",
            "human_photo_premium": "no bad hands",
            "general_premium": "no artifacts",
        },
        inline_profile_rubrics={
            "animal_photo_premium": ("animal anatomy",),
            "mascot_premium": ("3D anatomy",),
            "human_photo_premium": ("human anatomy",),
            "general_premium": ("coherent geometry",),
        },
        enable_inline_vlm=True,
        inline_vlm_model="Qwen/Qwen3-VL-2B-Instruct",
        inline_vlm_profile="general_premium",
        inline_vlm_rubric=("coherent geometry",),
        inline_vlm_expected_subject_count=1,
        inline_vlm_min_score=9.0,
    )
    source = provider._build_worker_source(
        ImageGenerationRequest("placeholder", width=512, height=512, seed=42)
    )

    compile(source, "<generated-image-worker>", "exec")
    assert "SUBJECT_CLASS: <animal|human|general>" in source
    assert "STYLE_CLASS: <photo|3d|mascot|illustration|general>" in source
    assert 'semantic_profile = "mascot_premium"' in source
    assert '"inline_planner_raw_command"' in source
    assert '("trâu", "buffalo", "water buffalo")' not in source


def test_inline_semantic_profile_maps_require_matching_keys():
    with pytest.raises(ValueError, match="matching keys"):
        KaggleImageProvider(
            worker=FakeWorker(),
            enable_inline_planner=True,
            inline_planner_raw_command="Tạo ảnh con voi thật",
            inline_planner_positive_constraints="fallback",
            inline_profile_positive_constraints={"animal_photo_premium": "photo"},
            inline_profile_negative_constraints={"general_premium": "bad"},
            inline_profile_rubrics={"animal_photo_premium": ("quality",)},
        )
