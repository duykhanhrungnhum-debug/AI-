from ai_agent.core.media_command import (
    HUMAN_PHOTO_PREMIUM,
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
    assert any("duplicated subject" in item for item in mascot.quality_bad_texts)


def test_mascot_critic_distinguishes_valid_appendages_from_broken_geometry():
    mascot = image_model_config(MASCOT_PREMIUM)

    assert any("multiple separate repeated characters" in item for item in mascot.quality_bad_texts)
    assert any("fused disconnected broken claws or legs" in item for item in mascot.quality_bad_texts)
    assert all("duplicated limbs claws appendages" not in item for item in mascot.quality_bad_texts)
