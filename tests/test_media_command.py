import pytest

from ai_agent.core.media_command import (
    HUMAN_PHOTO_PREMIUM,
    MEDIA_COMMAND_BRAIN_MODEL,
    MASCOT_PREMIUM,
    MediaCommandPlanner,
    benchmark_manifest,
    image_model_config,
)
from ai_agent.core.model import ModelResponse


class FakeModel:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt):
        return ModelResponse(self.text, "fake", "fake-model")


def test_routes_natural_human_command_to_premium_human_profile():
    planner = MediaCommandPlanner()
    plan = planner.plan("Tạo ảnh cô gái đang chạy bộ ngoài trời")

    assert plan.mode == "image"
    assert plan.profile == HUMAN_PHOTO_PREMIUM
    assert "anatomically correct body" in plan.prompt
    assert "deformed hands" in plan.negative_prompt
    assert plan.model_config.model == "SG161222/RealVisXL_V4.0"
    assert plan.model_config.scheduler == "dpm_karras"


def test_routes_mascot_command_to_premium_mascot_profile():
    planner = MediaCommandPlanner()
    plan = planner.plan("Tạo mascot con cua 3D dễ thương")

    assert plan.mode == "image"
    assert plan.profile == MASCOT_PREMIUM
    assert "clean rounded geometry" in plan.prompt
    assert "duplicated limbs" in plan.negative_prompt
    assert "multiple crabs" in plan.negative_prompt
    assert plan.model_config.model == "playgroundai/playground-v2.5-1024px-aesthetic"
    assert plan.model_config.scheduler == "edm_dpm"
    assert plan.model_config.guidance_scale == 3.0


def test_video_command_adds_motion_and_preserves_reference_identity():
    planner = MediaCommandPlanner(FakeModel(
        "PROMPT: the same young woman jogs across a modern glass bridge\n"
        "MOTION: natural jogging motion, subtle hair movement"
    ))
    plan = planner.plan(
        "Dùng nhân vật này làm video cô ấy chạy bộ trên cầu",
        has_reference_image=True,
    )

    assert plan.mode == "video"
    assert plan.reference_required is True
    assert "Preserve the supplied reference identity" in plan.prompt
    assert "identity drift" in plan.negative_prompt
    assert "natural jogging motion" in plan.motion_prompt


def test_model_cannot_remove_deterministic_quality_constraints():
    planner = MediaCommandPlanner(FakeModel("PROMPT: a woman standing outside\nMOTION:"))
    plan = planner.plan("Tạo ảnh người phụ nữ đứng ngoài trời")

    assert "anatomically correct body" in plan.prompt
    assert "five distinct fingers" in plan.prompt
    assert "obvious AI artifacts" in plan.negative_prompt


def test_benchmark_manifest_has_human_and_mascot_acceptance_rubrics():
    manifest = benchmark_manifest()

    assert "human_photo_premium" in manifest["profiles"]
    assert "mascot_premium" in manifest["profiles"]
    assert "correct hand/finger anatomy" in manifest["profiles"]["human_photo_premium"]["must_pass"]
    assert "clean rounded geometry" in manifest["profiles"]["mascot_premium"]["must_pass"]


def test_benchmark_manifest_pins_exact_user_reference_set():
    manifest = benchmark_manifest()
    refs = manifest["reference_set"]

    assert manifest["version"] == 2
    assert len(refs) == 4
    assert {item["profile"] for item in refs} == {"human_photo_premium", "mascot_premium"}
    assert {item["sha256"] for item in refs} == {
        "4306fb1186c907ca568afc4953936b4bb1363373b0edeb78d21f6b9de3a98119",
        "2bae56e56de0cf49d0423884e3d12f6d562cef66520560bd6093970f14d80fdb",
        "eaa4e7f5f1bae39378d87a0e180dd223647ed2ac988b56c76521e3fac807ae1b",
        "00c12e5cb4bd07b164a66cd58b06591f3455d53c6175f23c29aabc31e9fc2ad8",
    }


def test_profile_model_routing_uses_separate_human_and_mascot_stacks():
    human = image_model_config(HUMAN_PHOTO_PREMIUM)
    mascot = image_model_config(MASCOT_PREMIUM)

    assert human.model == "SG161222/RealVisXL_V4.0"
    assert mascot.model == "playgroundai/playground-v2.5-1024px-aesthetic"
    assert human.model != mascot.model
    assert "hands" in human.quality_good_text
    assert "exactly one subject" in mascot.quality_good_text
    assert any("duplicated whole subject" in item for item in mascot.quality_bad_texts)


def test_mascot_critic_distinguishes_valid_appendages_from_broken_geometry():
    mascot = image_model_config(MASCOT_PREMIUM)

    assert any("multiple separate repeated characters" in item for item in mascot.quality_bad_texts)
    assert any("fused disconnected broken claws or legs" in item for item in mascot.quality_bad_texts)
    assert all("duplicated limbs claws appendages" not in item for item in mascot.quality_bad_texts)


def test_command_compiler_accepts_fullwidth_label_separator_and_records_output():
    raw = (
        "PROMPT：one young Asian woman jogging on a modern glass bridge, full body, morning light\n"
        "MOTION："
    )
    plan = MediaCommandPlanner(FakeModel(raw)).plan(
        "Tạo ảnh cô gái châu Á chạy bộ trên cầu kính"
    )

    assert plan.prompt.startswith("one young Asian woman jogging")
    assert plan.compiler_output == raw


def test_command_compiler_rejects_untranslated_vietnamese_prompt_before_image_gpu():
    planner = MediaCommandPlanner(FakeModel(
        "PROMPT: Tạo ảnh cô gái đang chạy bộ ngoài trời trên cầu kính\nMOTION:"
    ))

    with pytest.raises(ValueError, match="translate PROMPT to English"):
        planner.plan("Tạo ảnh cô gái đang chạy bộ ngoài trời trên cầu kính")


def test_command_compiler_rejects_missing_prompt_label_before_image_gpu():
    planner = MediaCommandPlanner(FakeModel(
        "Here is your translated image request: a woman jogging on a bridge"
    ))

    with pytest.raises(ValueError, match="did not return a PROMPT line"):
        planner.plan("Tạo ảnh cô gái đang chạy bộ ngoài trời")


def test_qwen_real_label_typos_use_first_prompt_and_ignore_later_variant():
    raw = (
        "PROMT: A high-quality full-body portrait of a young Asian girl running on a modern glass bridge "
        "in the morning, wearing sportswear in black and jade green, holding a water bottle, natural happy face, "
        "clear from head to toe, correct hand and finger anatomy, natural light, high-end realistic photography style, "
        "one person in the image.\n\n"
        "MOTION:\n\n"
        "PROMP: A close-up alternative that must not replace the first prompt.\n\n"
        "MOTON:"
    )
    plan = MediaCommandPlanner(FakeModel(raw)).plan(
        "Tạo ảnh toàn thân cô gái chạy bộ trên cầu kính"
    )

    assert plan.prompt.startswith("A high-quality full-body portrait")
    assert "close-up alternative" not in plan.prompt
    assert plan.motion_prompt == ""


def test_qwen_spaced_prompt_label_is_normalized_without_using_alternative_description():
    raw = (
        "PROM PT: A high-quality full-body portrait of a young Asian girl running on a modern glass bridge "
        "in the morning, wearing sportswear in black and jade green, holding a water bottle, smiling naturally, "
        "clear from head to toe, correct anatomical hands and fingers, natural light, high-end realistic photography style, "
        "one person in the frame.\n\n"
        "MOTION:\n\n"
        "IMAGE_DESCRIPTION: an alternative description that must not replace PROM PT"
    )
    plan = MediaCommandPlanner(FakeModel(raw)).plan(
        "Tạo ảnh toàn thân cô gái chạy bộ trên cầu kính"
    )

    assert plan.prompt.startswith("A high-quality full-body portrait")
    assert "alternative description" not in plan.prompt


def test_media_command_brain_uses_qwen3_instruct_generation():
    assert MEDIA_COMMAND_BRAIN_MODEL == "Qwen/Qwen3-4B-Instruct-2507"


class CaptureModel(FakeModel):
    def __init__(self, text):
        super().__init__(text)
        self.last_prompt = ""

    def generate(self, prompt):
        self.last_prompt = prompt
        return super().generate(prompt)


def test_command_brain_may_choose_professional_camera_without_changing_story_facts():
    model = CaptureModel("PROMPT: one cute red crab mascot resting on a green lily pad\nMOTION:")
    planner = MediaCommandPlanner(model)
    plan = planner.plan("Tạo mascot con cua đỏ dễ thương nằm trên lá sen")

    assert plan.profile == MASCOT_PREMIUM
    assert "professional production-ready choices" in model.last_prompt
    assert "without changing the scene or subject" in model.last_prompt
    assert "one cute red crab mascot resting on a green lily pad" in plan.prompt
