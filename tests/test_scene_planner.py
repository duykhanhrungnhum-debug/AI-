import json

import pytest

from ai_agent.core.model import ModelResponse
from ai_agent.core.scene_planner import ScenePlanner


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return ModelResponse(json.dumps(self.payload), "fake", "model")


def test_scene_planner_creates_ordered_widescreen_scenes():
    model = FakeModel({
        "scenes": [
            {
                "scene_id": "s1",
                "narration": "Lan bước vào căn nhà.",
                "image_prompt": "Lan entering an abandoned house at night",
                "negative_prompt": "text, watermark",
            },
            {
                "scene_id": "s2",
                "narration": "Cô nhìn thấy lá thư cũ.",
                "image_prompt": "old sealed letter on dusty table, widescreen composition",
                "negative_prompt": "",
            },
        ]
    })

    plan = ScenePlanner(model, max_scenes=5).plan("Lan bước vào nhà và thấy một lá thư.")

    assert [scene.scene_id for scene in plan.scenes] == ["s1", "s2"]
    assert "16:9 widescreen composition" in plan.scenes[0].image_prompt
    assert "widescreen" in plan.scenes[1].image_prompt
    assert plan.scenes[1].negative_prompt
    assert model.prompts[0].startswith("SCENE_PLANNING")


def test_scene_planner_rejects_duplicate_ids():
    model = FakeModel({
        "scenes": [
            {"scene_id": "same", "narration": "a", "image_prompt": "one", "negative_prompt": ""},
            {"scene_id": "same", "narration": "b", "image_prompt": "two", "negative_prompt": ""},
        ]
    })

    with pytest.raises(ValueError, match="unique"):
        ScenePlanner(model).plan("story")


def test_scene_planner_enforces_scene_limit():
    model = FakeModel({
        "scenes": [
            {"scene_id": f"s{i}", "narration": "n", "image_prompt": "p", "negative_prompt": ""}
            for i in range(3)
        ]
    })

    with pytest.raises(ValueError, match="max_scenes"):
        ScenePlanner(model, max_scenes=2).plan("story")


def test_scene_planner_accepts_json_code_fence():
    payload = {
        "scenes": [
            {
                "scene_id": "s1",
                "narration": "Một cảnh.",
                "image_prompt": "a dark hallway",
                "negative_prompt": "",
            }
        ]
    }

    class FencedModel:
        def generate(self, prompt):
            return ModelResponse(
                "```json\n" + json.dumps(payload) + "\n```",
                "fake",
                "model",
            )

    plan = ScenePlanner(FencedModel()).plan("Một câu chuyện.")
    assert len(plan.scenes) == 1
    assert "16:9 widescreen composition" in plan.scenes[0].image_prompt


def test_scene_planner_supports_vertical_composition():
    model = FakeModel({
        "scenes": [
            {
                "scene_id": "v1",
                "narration": "Một cảnh dọc.",
                "image_prompt": "Vietnamese roadside food stall with two women",
                "negative_prompt": "",
            }
        ]
    })

    plan = ScenePlanner(model).plan(
        "Một cảnh hài ở quầy đồ ăn.",
        composition="9:16 vertical",
    )

    assert "9:16 vertical composition" in plan.scenes[0].image_prompt
    assert "9:16 vertical" in model.prompts[0]
