import pytest

from ai_agent.core.media_command import (
    ANIMAL_PHOTO_PREMIUM,
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
    assert "premium polished cute 3D character render" in plan.prompt
    assert "malformed appendages" in plan.negative_prompt
    assert "multiple characters" in plan.negative_prompt
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


def test_benchmark_manifest_has_human_animal_and_mascot_acceptance_rubrics():
    manifest = benchmark_manifest()

    assert "human_photo_premium" in manifest["profiles"]
    assert "animal_photo_premium" in manifest["profiles"]
    assert "mascot_premium" in manifest["profiles"]
    assert any("hand/finger anatomy" in item for item in manifest["profiles"]["human_photo_premium"]["must_pass"])
    assert any("species-correct" in item for item in manifest["profiles"]["animal_photo_premium"]["must_pass"])
    assert any("species-correct" in item or "3D mascot" in item for item in manifest["profiles"]["mascot_premium"]["must_pass"])


def test_benchmark_manifest_pins_exact_user_reference_set():
    manifest = benchmark_manifest()
    refs = manifest["reference_set"]

    assert manifest["version"] == 4
    assert len(refs) == 5
    assert {item["profile"] for item in refs} == {"human_photo_premium", "animal_photo_premium", "mascot_premium"}
    assert {item["sha256"] for item in refs} == {
        "4306fb1186c907ca568afc4953936b4bb1363373b0edeb78d21f6b9de3a98119",
        "2bae56e56de0cf49d0423884e3d12f6d562cef66520560bd6093970f14d80fdb",
        "eaa4e7f5f1bae39378d87a0e180dd223647ed2ac988b56c76521e3fac807ae1b",
        "00c12e5cb4bd07b164a66cd58b06591f3455d53c6175f23c29aabc31e9fc2ad8",
        "792db3d7f02687715c4893fe0b87a1e189b1185c7ea996ba9d4defe0dc844c74",
    }


def test_profile_model_routing_uses_separate_photo_and_mascot_stacks():
    human = image_model_config(HUMAN_PHOTO_PREMIUM)
    animal = image_model_config(ANIMAL_PHOTO_PREMIUM)
    mascot = image_model_config(MASCOT_PREMIUM)

    assert human.model == "SG161222/RealVisXL_V4.0"
    assert animal.model == "SG161222/RealVisXL_V5.0"
    assert mascot.model == "playgroundai/playground-v2.5-1024px-aesthetic"
    assert animal.model != mascot.model
    assert "hands" in human.quality_good_text
    assert "species-correct anatomy" in animal.quality_good_text
    assert "species-correct rounded anatomy" in mascot.quality_good_text
    assert any("duplicated whole subject" in item for item in mascot.quality_bad_texts)


def test_mascot_critic_distinguishes_valid_appendages_from_broken_geometry():
    mascot = image_model_config(MASCOT_PREMIUM)

    assert any("repeated characters" in item for item in mascot.quality_bad_texts)
    assert any("malformed paws" in item or "impossible anatomy" in item for item in mascot.quality_bad_texts)
    assert any("plastic toy" in item for item in mascot.quality_bad_texts)


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


def test_command_compiler_accepts_boilerplate_english_prompt_without_label():
    planner = MediaCommandPlanner(FakeModel(
        "Here is your translated image request: a woman jogging on a bridge"
    ))

    plan = planner.plan("Tạo ảnh cô gái đang chạy bộ ngoài trời")
    assert plan.prompt.startswith("a woman jogging on a bridge")


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
    assert MEDIA_COMMAND_BRAIN_MODEL == "Qwen/Qwen3-1.7B"


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


def test_benchmark_manifest_keeps_user_rejected_examples_but_generalizes_3d_quality():
    manifest = benchmark_manifest()
    mascot = manifest["profiles"]["mascot_premium"]
    human = manifest["profiles"]["human_photo_premium"]

    assert "2525bafcc74e14444cb64e9c355a882d280417c712a23d1eaadd126aa3c28034" in mascot["known_rejected_sha256"]
    assert "8e38371165a9c1017a1b56a414061f827ff11133cbe23231b7ab5aa800ffe1e9" in human["known_rejected_sha256"]
    assert any("species-correct" in item for item in mascot["must_pass"])
    assert any("plastic-toy" in item for item in mascot["must_pass"])
    assert all("lily pad" not in item for item in mascot["must_pass"])


def test_dog_exam_routes_real_photo_and_cute_3d_to_different_profiles():
    planner = MediaCommandPlanner()
    real = planner.plan("Tạo ảnh một chú chó Golden Retriever thật, ảnh chụp chân thực ngoài trời")
    cute = planner.plan("Tạo ảnh một chú chó Golden Retriever 3D cute")

    assert real.profile == ANIMAL_PHOTO_PREMIUM
    assert real.model_config.model == "SG161222/RealVisXL_V5.0"
    assert "species-correct anatomy" in real.prompt
    assert "cartoon" in real.negative_prompt

    assert cute.profile == MASCOT_PREMIUM
    assert cute.model_config.model == "playgroundai/playground-v2.5-1024px-aesthetic"
    assert "premium polished cute 3D character render" in cute.prompt
    assert "crab" not in cute.prompt.casefold()
    assert "lily pad" not in cute.prompt.casefold()
    assert cute.model_config.inference_steps == 40


def test_command_compiler_accepts_json_prompt_output():
    planner = MediaCommandPlanner(FakeModel(
        '{"prompt":"one photorealistic golden retriever standing on grass in daylight","motion":""}'
    ))
    plan = planner.plan("Tạo ảnh chó Golden Retriever thật ngoài trời")

    assert plan.prompt.startswith("one photorealistic golden retriever")
    assert plan.motion_prompt == ""


def test_command_compiler_accepts_clean_single_english_prompt_without_label():
    planner = MediaCommandPlanner(FakeModel(
        "one premium cute 3D golden retriever puppy mascot with coherent dog anatomy and studio lighting"
    ))
    plan = planner.plan("Tạo ảnh chó Golden Retriever 3D cute")

    assert plan.prompt.startswith("one premium cute 3D golden retriever puppy mascot")
    assert plan.profile == MASCOT_PREMIUM


class FakeBatchModel:
    def __init__(self, texts):
        self.texts = tuple(texts)
        self.calls = 0
        self.batch_calls = 0

    def generate(self, prompt):
        self.calls += 1
        return ModelResponse(self.texts[0], "fake", "fake-model")

    def generate_many(self, prompts):
        self.batch_calls += 1
        responses = tuple(
            ModelResponse(text, "fake", "fake-model")
            for text in self.texts
        )
        return type("Batch", (), {"responses": responses})()


def test_plan_many_uses_one_batch_model_call_for_multiple_commands():
    model = FakeBatchModel((
        "PROMPT: one photorealistic golden retriever on grass\nMOTION:",
        "PROMPT: one cute 3D golden retriever mascot in studio lighting\nMOTION:",
    ))
    planner = MediaCommandPlanner(model)
    plans = planner.plan_many((
        "Tạo ảnh chó Golden Retriever thật ngoài trời",
        "Tạo ảnh chó Golden Retriever 3D cute",
    ))

    assert model.batch_calls == 1
    assert model.calls == 0
    assert plans[0].profile == ANIMAL_PHOTO_PREMIUM
    assert plans[1].profile == MASCOT_PREMIUM
    assert plans[0].prompt.startswith("one photorealistic golden retriever")
    assert plans[1].prompt.startswith("one cute 3D golden retriever")


def test_animal_photo_profile_prefers_natural_unretouched_camera_look():
    plan = MediaCommandPlanner().plan(
        "Tạo ảnh một chú chó Golden Retriever thật, ảnh chụp chân thực ngoài trời"
    )
    cfg = image_model_config(ANIMAL_PHOTO_PREMIUM)
    manifest = benchmark_manifest()["profiles"]["animal_photo_premium"]

    assert "natural unretouched photorealistic animal photograph" in plan.prompt
    assert "four distinct anatomically connected legs" in plan.prompt
    assert "impossible gait" in plan.negative_prompt
    assert "HDR look" in plan.negative_prompt
    assert "oversharpening" in plan.negative_prompt
    assert "advertising retouch" in plan.negative_prompt
    assert cfg.inference_steps == 28
    assert cfg.guidance_scale == 4.0
    assert any("HDR contrast" in item for item in cfg.quality_bad_texts)
    assert any("Photoshop look" in item for item in cfg.quality_bad_texts)
    assert "2e476d56320d8ea3b16488d9c276e76e80fe737984e7f90fd4e98f19271f2589" in manifest["known_rejected_sha256"]
    assert any("real camera" in item for item in manifest["must_pass"])
