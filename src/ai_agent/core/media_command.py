"""Natural-language media command planning with premium quality profiles."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

from .model import ModelProvider


MediaMode = Literal["image", "video"]
QualityProfileName = Literal["human_photo_premium", "mascot_premium", "general_premium"]

MEDIA_COMMAND_BRAIN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class MediaQualityProfile:
    name: QualityProfileName
    positive_constraints: str
    negative_constraints: str
    width: int
    height: int


HUMAN_PHOTO_PREMIUM = MediaQualityProfile(
    name="human_photo_premium",
    positive_constraints=(
        "premium photorealistic image, natural face and expression, anatomically correct body, "
        "natural shoulders elbows wrists and hands, five distinct fingers per visible hand, "
        "realistic skin hair and clothing texture, clean background geometry, physically plausible objects, "
        "professional composition and lighting, crisp detail, immediately usable final image"
    ),
    negative_constraints=(
        "bad anatomy, deformed hands, malformed fingers, fused fingers, extra fingers, missing fingers, "
        "extra limbs, twisted arms, distorted face, asymmetrical eyes, warped body, melted clothing, "
        "warped objects, broken perspective, duplicate person, obvious AI artifacts, text, watermark, logo"
    ),
    width=832,
    height=1216,
)


MASCOT_PREMIUM = MediaQualityProfile(
    name="mascot_premium",
    positive_constraints=(
        "premium polished 3D mascot asset, EXACTLY ONE main mascot subject, one body only, clean rounded geometry, "
        "coherent symmetrical appendages, crisp silhouette and edges, expressive readable face, detailed clean materials, "
        "professional studio lighting, balanced centered composition, production-ready final asset"
    ),
    negative_constraints=(
        "multiple mascots, multiple crabs, duplicate character, repeated subject, crowd, duplicated body, "
        "malformed appendages, duplicated limbs, extra claws, extra legs, asymmetrical broken geometry, melted shape, "
        "warped shell, messy outline, muddy texture, distorted face, inconsistent eyes, obvious AI artifacts, "
        "text, watermark, logo"
    ),
    width=1024,
    height=1024,
)


GENERAL_PREMIUM = MediaQualityProfile(
    name="general_premium",
    positive_constraints=(
        "premium production-ready image, coherent geometry, crisp details, professional composition and lighting, "
        "clean materials and textures, no obvious AI artifacts"
    ),
    negative_constraints=(
        "warped geometry, duplicated objects, melted details, broken perspective, blurry, noisy, unfinished, "
        "obvious AI artifacts, text, watermark, logo"
    ),
    width=1024,
    height=1024,
)


@dataclass(frozen=True)
class ImageModelConfig:
    model: str
    inference_steps: int
    guidance_scale: float
    scheduler: str
    quality_good_text: str
    quality_bad_texts: tuple[str, ...]


HUMAN_IMAGE_MODEL = ImageModelConfig(
    model="SG161222/RealVisXL_V4.0",
    inference_steps=28,
    guidance_scale=5.5,
    scheduler="dpm_karras",
    quality_good_text=(
        "premium photorealistic professional photograph, natural face and expression, correct human anatomy, "
        "realistic hands with coherent fingers, plausible shoulders elbows wrists and body proportions, "
        "clean clothing skin hair, physically plausible objects and background geometry, crisp production-ready detail"
    ),
    quality_bad_texts=(
        "bad AI portrait with deformed hands, fused fingers, extra or missing fingers, twisted wrists or arms",
        "bad AI human image with distorted face, asymmetrical eyes, warped body, extra limbs or broken anatomy",
        "bad AI photograph with melted clothing, warped furniture or objects, broken perspective and obvious artifacts",
    ),
)


MASCOT_IMAGE_MODEL = ImageModelConfig(
    model="playgroundai/playground-v2.5-1024px-aesthetic",
    inference_steps=30,
    guidance_scale=3.0,
    scheduler="edm_dpm",
    quality_good_text=(
        "premium polished cute 3D mascot asset, exactly one subject, clean rounded geometry, coherent symmetrical "
        "appendages, crisp silhouette, expressive clean eyes and mouth, detailed materials, studio lighting, "
        "production-ready sticker or character asset"
    ),
    quality_bad_texts=(
        "bad mascot image with multiple separate repeated characters, duplicated whole subject or crowd of copies",
        "bad 3D mascot with fused disconnected broken claws or legs, melted joints, impossible attachment points, warped body",
        "bad mascot asset with distorted inconsistent eyes or mouth, messy silhouette, muddy textures and obvious AI artifacts",
    ),
)


GENERAL_IMAGE_MODEL = ImageModelConfig(
    model="playgroundai/playground-v2.5-1024px-aesthetic",
    inference_steps=30,
    guidance_scale=3.0,
    scheduler="edm_dpm",
    quality_good_text=(
        "premium production-ready image, coherent geometry, crisp detail, professional composition and lighting, "
        "clean materials and textures, no obvious AI artifacts"
    ),
    quality_bad_texts=(
        "bad AI image with warped geometry, duplicated subjects or objects and broken perspective",
        "low quality blurry noisy unfinished image with melted details and obvious AI artifacts",
    ),
)


def image_model_config(profile: MediaQualityProfile | QualityProfileName) -> ImageModelConfig:
    name = profile.name if isinstance(profile, MediaQualityProfile) else profile
    if name == HUMAN_PHOTO_PREMIUM.name:
        return HUMAN_IMAGE_MODEL
    if name == MASCOT_PREMIUM.name:
        return MASCOT_IMAGE_MODEL
    return GENERAL_IMAGE_MODEL


@dataclass(frozen=True)
class MediaCommandPlan:
    mode: MediaMode
    profile: MediaQualityProfile
    model_config: ImageModelConfig
    prompt: str
    negative_prompt: str
    motion_prompt: str = ""
    reference_required: bool = False
    compiler_output: str = ""


class MediaCommandPlanner:
    """Turn an ordinary user command into an executable premium media plan.

    The model is used only to rewrite the user's intent into concise English visual
    instructions. Routing and quality constraints remain deterministic so a model
    response cannot silently weaken the acceptance standard.
    """

    def __init__(self, provider: ModelProvider | None = None):
        self.provider = provider

    def plan(self, command: str, *, has_reference_image: bool = False) -> MediaCommandPlan:
        command = command.strip()
        if not command:
            raise ValueError("media command must not be empty")

        mode = self._infer_mode(command)
        profile = self._infer_profile(command)
        rewritten_prompt = command
        motion_prompt = ""
        compiler_output = ""

        if self.provider is not None:
            compiler_prompt = (
                "MEDIA_COMMAND_COMPILE\n"
                "Translate and compile the user's request into concise ENGLISH visual instructions. "
                "The PROMPT value MUST be English even when USER_COMMAND is Vietnamese or another language. "
                "Preserve every requested subject count, person/object, action, location, clothing, held object, "
                "camera/framing, time of day and visual style. Do not invent or remove story facts. "
                "When camera/framing/lighting are not specified, choose professional production-ready choices that best "
                "express the user's intent without changing the scene or subject. Resolve pronouns and implied references "
                "from the command conservatively. Keep identity/reference instructions if present. Do not lower quality requirements. "
                "Return exactly two plain-text labeled lines and no Markdown or commentary. "
                "Return ONE image description only; do not propose alternatives, variants, close-ups or second prompts:\n"
                "PROMPT: <one complete concise English still-image description>\n"
                "MOTION: <English motion only when video is requested; otherwise leave empty>\n"
                f"USER_COMMAND: {command}"
            )
            compiler_output = self.provider.generate(compiler_prompt).text.strip()
            parsed = self._parse_labeled(compiler_output)
            rewritten_prompt = parsed.get("PROMPT", "").strip()
            motion_prompt = parsed.get("MOTION", "").strip()
            if not rewritten_prompt:
                raise ValueError(
                    "media command compiler did not return a PROMPT line; refusing to send an uncompiled command to the image model"
                )
            if self._looks_vietnamese(rewritten_prompt):
                raise ValueError(
                    "media command compiler did not translate PROMPT to English; refusing image generation"
                )

        positive = profile.positive_constraints
        prompt = rewritten_prompt.rstrip(" .") + ". Quality requirements: " + positive + "."
        negatives = profile.negative_constraints

        if has_reference_image:
            prompt += (
                " Preserve the supplied reference identity and distinctive features exactly unless the user explicitly "
                "asks to change them."
            )
            negatives += ", changed identity, different person, face drift, identity drift"

        if mode == "video" and not motion_prompt:
            motion_prompt = "natural physically plausible subject and environment motion, stable identity, no morphing"

        return MediaCommandPlan(
            mode=mode,
            profile=profile,
            model_config=image_model_config(profile),
            prompt=prompt,
            negative_prompt=negatives,
            motion_prompt=motion_prompt,
            reference_required=has_reference_image,
            compiler_output=compiler_output,
        )

    @staticmethod
    def _infer_mode(command: str) -> MediaMode:
        text = command.casefold()
        video_terms = (
            "video", "clip", "animate", "animation", "chuyển động", "làm phim", "tạo phim", "i2v"
        )
        return "video" if any(term in text for term in video_terms) else "image"

    @staticmethod
    def _infer_profile(command: str) -> MediaQualityProfile:
        text = command.casefold()
        mascot_terms = (
            "mascot", "sticker", "cute 3d", "3d cute", "chibi", "hoạt hình 3d",
            "nhân vật dễ thương", "con cua", "crab mascot"
        )
        human_terms = (
            "người", "cô gái", "phụ nữ", "đàn ông", "chàng trai", "cô ấy", "anh ấy",
            "woman", "girl", "man", "person", "portrait", "selfie", "full body", "chân dung"
        )
        if any(term in text for term in mascot_terms):
            return MASCOT_PREMIUM
        if any(term in text for term in human_terms):
            return HUMAN_PHOTO_PREMIUM
        return GENERAL_PREMIUM

    @staticmethod
    def _parse_labeled(raw: str) -> dict[str, str]:
        values: dict[str, str] = {}
        normalized = raw.replace("：", ":")
        aliases = {
            "PROMPT": "PROMPT",
            "PROMT": "PROMPT",
            "PROMP": "PROMPT",
            "MOTION": "MOTION",
            "MOTON": "MOTION",
            "MOTIN": "MOTION",
        }
        for line in normalized.splitlines():
            line = line.strip().lstrip("-*# ").strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = "".join(
                character
                for character in key.strip().upper()
                if character.isalpha()
            )
            canonical = aliases.get(key)
            if canonical is not None and canonical not in values:
                values[canonical] = value.strip().strip(chr(96)).strip()
        return values

    @staticmethod
    def _looks_vietnamese(text: str) -> bool:
        lowered = text.casefold()
        vietnamese_chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
        if any(char in vietnamese_chars for char in lowered):
            return True
        common_words = (
            " cô gái ", " phụ nữ ", " đàn ông ", " chạy bộ ", " ngoài trời ", " buổi sáng ",
            " trang phục ", " chai nước ", " gương mặt ", " toàn thân ", " chỉ một ",
        )
        padded = " " + lowered + " "
        return sum(term in padded for term in common_words) >= 2


def benchmark_manifest() -> dict:
    """Acceptance rubric tied to the exact user-supplied premium reference set.

    Only immutable hashes/metadata are stored here; the raw personal examples are
    not committed to the repository. This prevents later tests from silently
    substituting a different benchmark set.
    """
    return {
        "version": 3,
        "reference_set": [
            {
                "profile": HUMAN_PHOTO_PREMIUM.name,
                "sha256": "4306fb1186c907ca568afc4953936b4bb1363373b0edeb78d21f6b9de3a98119",
                "width": 679,
                "height": 1536,
                "role": "full-body outdoor fitness photo quality reference",
            },
            {
                "profile": HUMAN_PHOTO_PREMIUM.name,
                "sha256": "2bae56e56de0cf49d0423884e3d12f6d562cef66520560bd6093970f14d80fdb",
                "width": 679,
                "height": 1536,
                "role": "close-up selfie portrait quality reference",
            },
            {
                "profile": MASCOT_PREMIUM.name,
                "sha256": "eaa4e7f5f1bae39378d87a0e180dd223647ed2ac988b56c76521e3fac807ae1b",
                "width": 679,
                "height": 1536,
                "role": "sleepy 3D crab mascot quality reference",
            },
            {
                "profile": MASCOT_PREMIUM.name,
                "sha256": "00c12e5cb4bd07b164a66cd58b06591f3455d53c6175f23c29aabc31e9fc2ad8",
                "width": 679,
                "height": 1536,
                "role": "joyful 3D crab mascot quality reference",
            },
        ],
        "profiles": {
            HUMAN_PHOTO_PREMIUM.name: {
                "must_pass": [
                    "premium commercial-photo finish comparable to the user's accepted human references",
                    "natural East Asian facial proportions and expression; no doll-like, plastic or distorted face",
                    "correct hand/finger anatomy with five distinct plausible fingers when visible",
                    "plausible body proportions, shoulders, elbows, wrists, hips, knees and feet",
                    "clean realistic clothing, skin and hair texture without melted or waxy surfaces",
                    "straight coherent bridge/building/railing geometry and physically plausible held objects",
                    "natural daylight, believable depth and premium photographic detail rather than synthetic CGI gloss",
                    "no obvious AI artifacts, duplicate body parts, warped props, pseudo-text or broken perspective",
                    "production-ready overall appearance at the same quality bar as the supplied benchmark set",
                ],
                "known_rejected_sha256": [
                    "8e38371165a9c1017a1b56a414061f827ff11133cbe23231b7ab5aa800ffe1e9",
                ],
            },
            MASCOT_PREMIUM.name: {
                "must_pass": [
                    "match the user's accepted cute 3D crab benchmark visual language, not merely generic 3D quality",
                    "squat low wide crab silhouette resting naturally on a broad green lily pad; NOT a tall spherical balloon body",
                    "two large rounded frontal claws integrated close to the body; NOT thin raised scissor-like claws",
                    "coherent crab legs and two short antennae; no duplicated, missing, detached or implausibly attached appendages",
                    "organic bumpy red-orange shell surface with soft cream underside/claw tips; NOT shiny plastic-toy material",
                    "cute face integrated into crab anatomy with expressive eyes/mouth; no human-like eyebrows, nose or teeth unless requested",
                    "clean sticker/cutout presentation with a crisp white outline around the crab and lily pad, black or transparent-style background",
                    "broad green lily pad with clean shape and visible water droplets when appropriate",
                    "soft polished studio-quality 3D lighting and materials without dark glossy toy-render look",
                    "production-ready overall appearance at the same quality bar as the supplied sleepy/joyful crab references",
                ],
                "known_rejected_sha256": [
                    "2525bafcc74e14444cb64e9c355a882d280417c712a23d1eaadd126aa3c28034",
                ],
            },
        },
    }
