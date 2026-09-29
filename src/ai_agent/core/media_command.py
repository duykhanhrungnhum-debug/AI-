"""Natural-language media command planning with premium quality profiles."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

from .model import ModelProvider


MediaMode = Literal["image", "video"]
QualityProfileName = Literal["human_photo_premium", "mascot_premium", "general_premium"]


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
        "premium polished 3D mascot asset, clean rounded geometry, coherent symmetrical appendages, "
        "crisp silhouette and edges, expressive readable face, detailed clean materials, professional lighting, "
        "balanced composition, production-ready final asset"
    ),
    negative_constraints=(
        "malformed appendages, duplicated limbs, asymmetrical broken geometry, melted shape, warped shell, "
        "messy outline, muddy texture, distorted face, inconsistent eyes, obvious AI artifacts, text, watermark, logo"
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
class MediaCommandPlan:
    mode: MediaMode
    profile: MediaQualityProfile
    prompt: str
    negative_prompt: str
    motion_prompt: str = ""
    reference_required: bool = False


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

        if self.provider is not None:
            prompt = (
                "MEDIA_COMMAND_COMPILE\n"
                "Convert the user's request into concise English visual instructions without adding people, objects, "
                "actions or story facts that the user did not request. Keep identity/reference instructions if present. "
                "Do not lower quality requirements. Return exactly two labeled lines and no Markdown:\n"
                "PROMPT: <one concise still-image description>\n"
                "MOTION: <only requested or natural minimal motion; empty if image-only>\n"
                f"USER_COMMAND: {command}"
            )
            raw = self.provider.generate(prompt).text.strip()
            parsed = self._parse_labeled(raw)
            rewritten_prompt = parsed.get("PROMPT", "").strip() or command
            motion_prompt = parsed.get("MOTION", "").strip()

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
            prompt=prompt,
            negative_prompt=negatives,
            motion_prompt=motion_prompt,
            reference_required=has_reference_image,
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
        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            if key in {"PROMPT", "MOTION"}:
                values[key] = value.strip()
        return values


def benchmark_manifest() -> dict:
    """Acceptance rubric tied to the exact user-supplied premium reference set.

    Only immutable hashes/metadata are stored here; the raw personal examples are
    not committed to the repository. This prevents later tests from silently
    substituting a different benchmark set.
    """
    return {
        "version": 2,
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
                    "natural face and expression",
                    "correct hand/finger anatomy",
                    "plausible body proportions and joints",
                    "clean clothing/skin/hair detail",
                    "clean background geometry and objects",
                    "no obvious AI artifacts",
                    "production-ready overall appearance",
                ],
            },
            MASCOT_PREMIUM.name: {
                "must_pass": [
                    "clean rounded geometry",
                    "coherent appendages and symmetry",
                    "clean expressive eyes/mouth",
                    "crisp silhouette/cutout-ready edges",
                    "clean materials and lighting",
                    "no malformed or duplicated parts",
                    "production-ready overall appearance",
                ],
            },
        },
    }
