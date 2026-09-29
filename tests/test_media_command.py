from ai_agent.core.media_command import (
    HUMAN_PHOTO_PREMIUM,
    MASCOT_PREMIUM,
    MediaCommandPlanner,
    benchmark_manifest,
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


def test_routes_mascot_command_to_premium_mascot_profile():
    planner = MediaCommandPlanner()
    plan = planner.plan("Tạo mascot con cua 3D dễ thương")

    assert plan.mode == "image"
    assert plan.profile == MASCOT_PREMIUM
    assert "clean rounded geometry" in plan.prompt
    assert "duplicated limbs" in plan.negative_prompt


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
