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
    motion_prompt: str = ""
    continuity_anchor: str = ""
    camera: str = ""
    lighting: str = ""
    palette: str = ""


@dataclass(frozen=True)
class VisualScenePlan:
    scenes: tuple[VisualScene, ...]
    continuity_bible: str = ""


class ScenePlanner:
    """Plan image/video scenes while carrying continuity between shots."""

    def __init__(self, provider: ModelProvider, *, max_scenes: int = 40):
        if max_scenes <= 0:
            raise ValueError("max_scenes must be positive")
        self.provider = provider
        self.max_scenes = max_scenes

    def plan(
        self,
        script: str,
        *,
        visual_style: str = "cinematic realistic",
        composition: str = "16:9 widescreen",
    ) -> VisualScenePlan:
        assert_core_invariants()
        script = script.strip()
        visual_style = visual_style.strip()
        composition = composition.strip()
        if not script:
            raise ValueError("script must not be empty")
        if not visual_style:
            raise ValueError("visual_style must not be empty")
        if not composition:
            raise ValueError("composition must not be empty")

        prompt = (
            "SCENE_PLANNING\n"
            "Split the SCRIPT into visual scenes for a narrated video. Preserve story order and character identity. "
            f"Use at most {self.max_scenes} scenes. Each image prompt must describe one still image, use a {composition} "
            "composition, and must not invent plot facts not present in the script. "
            f"Visual style: {visual_style}. "
            "First build one compact continuity_bible describing recurring character identity, wardrobe, important "
            "objects, location traits, time-of-day, lighting logic and stable palette. Reuse those facts across scenes. "
            "For every scene separate WHAT STAYS from WHAT CHANGES: continuity_anchor contains only details that must "
            "remain stable from neighboring scenes; motion_prompt contains only the intended subject/camera/environment "
            "motion for the shot. Keep motion simple enough for a short generated clip. "
            "Return ONLY a JSON object with keys continuity_bible and scenes. Each scene must contain scene_id, narration, "
            "image_prompt and negative_prompt; it may also contain motion_prompt, continuity_anchor, camera, lighting, "
            "and palette. scene_id values must be unique strings. Keep prompts concise and concrete.\n"
            f"SCRIPT:\n{script}"
        )
        response = self.provider.generate(prompt)
        raw = response.text.strip()
        fence = chr(96) * 3
        if raw.startswith(fence):
            lines = raw.splitlines()
            if lines and lines[0].startswith(fence):
                lines = lines[1:]
            if lines and lines[-1].strip() == fence:
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

        continuity_bible = self._clean(data.get("continuity_bible"))
        scenes: list[VisualScene] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_scenes):
            if not isinstance(item, dict):
                raise ValueError(f"scene {index} must be an object")
            scene_id = self._clean(item.get("scene_id"))
            narration = self._clean(item.get("narration"))
            image_prompt = self._clean(item.get("image_prompt"))
            negative_prompt = self._clean(item.get("negative_prompt"))
            motion_prompt = self._clean(item.get("motion_prompt"))
            continuity_anchor = self._clean(item.get("continuity_anchor"))
            camera = self._clean(item.get("camera"))
            lighting = self._clean(item.get("lighting"))
            palette = self._clean(item.get("palette"))
            if not scene_id or not narration or not image_prompt:
                raise ValueError(f"scene {index} is missing required text")
            if scene_id in seen:
                raise ValueError("scene_id values must be unique")
            seen.add(scene_id)

            if composition.casefold() not in image_prompt.casefold():
                image_prompt = image_prompt + f", {composition} composition"

            locks = []
            if continuity_bible:
                locks.append(f"Continuity bible: {continuity_bible}")
            if continuity_anchor:
                locks.append(f"Continuity lock: {continuity_anchor}")
            if camera:
                locks.append(f"Camera: {camera}")
            if lighting:
                locks.append(f"Lighting: {lighting}")
            if palette:
                locks.append(f"Palette: {palette}")
            if locks:
                image_prompt = image_prompt.rstrip(" .") + ". " + ". ".join(locks) + "."

            scenes.append(
                VisualScene(
                    scene_id=scene_id,
                    narration=narration,
                    image_prompt=image_prompt,
                    negative_prompt=negative_prompt or "text, watermark, logo, blurry, low quality",
                    motion_prompt=motion_prompt,
                    continuity_anchor=continuity_anchor,
                    camera=camera,
                    lighting=lighting,
                    palette=palette,
                )
            )
        return VisualScenePlan(tuple(scenes), continuity_bible=continuity_bible)

    @staticmethod
    def _clean(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value).strip()
