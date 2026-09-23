"""Convert a verified narrative script into bounded visual scenes for video production."""
from __future__ import annotations

from dataclasses import dataclass
import json

from .invariants import assert_core_invariants
from .model import ModelProvider


@dataclass(frozen=True)
class VisualScene:
    scene_id: str
    narration: str
    image_prompt: str
    negative_prompt: str = "text, watermark, logo, blurry, low quality"


@dataclass(frozen=True)
class VisualScenePlan:
    scenes: tuple[VisualScene, ...]


class ScenePlanner:
    """Plan image scenes from a script without coupling to any YouTube project."""

    def __init__(self, provider: ModelProvider, *, max_scenes: int = 40):
        if max_scenes <= 0:
            raise ValueError("max_scenes must be positive")
        self.provider = provider
        self.max_scenes = max_scenes

    def plan(\n        self,\n        script: str,\n        *,\n        visual_style: str = "cinematic realistic",\n        composition: str = "16:9 widescreen",\n    ) -> VisualScenePlan:\n        assert_core_invariants()
        script = script.strip()
        visual_style = visual_style.strip()
        if not script:
            raise ValueError("script must not be empty")
        if not visual_style:
            raise ValueError("visual_style must not be empty")

        prompt = (
            "SCENE_PLANNING\n"
            "Split the SCRIPT into visual scenes for a narrated video. Preserve story order and character identity. "
            f"Use at most {self.max_scenes} scenes. Each image prompt must describe one still image, use a 16:9 "
            "widescreen composition, and must not invent plot facts not present in the script. "
            f"Visual style: {visual_style}. "
            "Return ONLY a JSON object with key scenes. Each scene must contain scene_id, narration, image_prompt, "
            "and negative_prompt. scene_id values must be unique strings.\n"
            f"SCRIPT:\n{script}"
        )
        response = self.provider.generate(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("scene plan must be valid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
            raise ValueError("scene plan must contain a scenes array")

        raw_scenes = data["scenes"]
        if not raw_scenes:
            raise ValueError("scene plan must contain at least one scene")
        if len(raw_scenes) > self.max_scenes:
            raise ValueError("scene plan exceeds max_scenes")

        scenes: list[VisualScene] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_scenes):
            if not isinstance(item, dict):
                raise ValueError(f"scene {index} must be an object")
            scene_id = str(item.get("scene_id", "")).strip()
            narration = str(item.get("narration", "")).strip()
            image_prompt = str(item.get("image_prompt", "")).strip()
            negative_prompt = str(item.get("negative_prompt", "")).strip()
            if not scene_id or not narration or not image_prompt:
                raise ValueError(f"scene {index} is missing required text")
            if scene_id in seen:
                raise ValueError("scene_id values must be unique")
            seen.add(scene_id)
            if "16:9" not in image_prompt and "widescreen" not in image_prompt.casefold():
                image_prompt = image_prompt + ", 16:9 widescreen composition"
            scenes.append(
                VisualScene(
                    scene_id=scene_id,
                    narration=narration,
                    image_prompt=image_prompt,
                    negative_prompt=negative_prompt or "text, watermark, logo, blurry, low quality",
                )
            )
        return VisualScenePlan(tuple(scenes))
