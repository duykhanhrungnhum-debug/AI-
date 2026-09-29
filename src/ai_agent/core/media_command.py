"""Natural-language media command planning with premium quality profiles."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

from .model import ModelProvider


MediaMode = Literal["image", "video"]
QualityProfileName = Literal["human_photo_premium", "animal_photo_premium", "mascot_premium", "general_premium"]

MEDIA_COMMAND_FAST_MODEL = "Qwen/Qwen3-0.6B"
MEDIA_COMMAND_BRAIN_MODEL = "Qwen/Qwen3-1.7B"


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


ANIMAL_PHOTO_PREMIUM = MediaQualityProfile(
    name="animal_photo_premium",
    positive_constraints=(
        "natural unretouched photorealistic animal photograph, exactly one main animal subject unless requested otherwise, "
        "species-correct anatomy and proportions, natural eyes nose mouth ears paws legs and tail, realistic irregular fur texture, "
        "four distinct anatomically connected legs with plausible shoulders hips elbows hocks and paws when the species has four legs, "
        "natural readable limb separation with no fused or ambiguous paws, biomechanically plausible gait, weight bearing and ground contact, "
        "ordinary natural daylight, believable camera exposure and lens rendering, moderate contrast and saturation, realistic depth of field, "
        "subtle microcontrast, small natural imperfections, clean coherent background geometry, authentic camera-captured look rather than polished advertising retouch"
    ),
    negative_constraints=(
        "cartoon, CGI, 3D render, plastic fur, airbrushed fur, beauty retouch, commercial advertising retouch, HDR look, "
        "oversharpening, excessive microcontrast, excessive contrast, oversaturated colors, artificial glow, fake bokeh, "
        "perfectly smoothed surfaces, malformed paws, fused toes, fused legs, ambiguous paw count, extra legs, missing legs, duplicated limbs, "
        "misplaced joints, impossible gait, floating paws, warped face, asymmetrical eyes, deformed muzzle, broken tail, melted fur, "
        "duplicate animal, broken perspective, obvious AI artifacts, text, watermark, logo"
    ),
    width=1024,
    height=1024,
)


MASCOT_PREMIUM = MediaQualityProfile(
    name="mascot_premium",
    positive_constraints=(
        "premium polished cute 3D character render matching the user's accepted mascot benchmark quality, "
        "EXACTLY ONE main character unless requested otherwise, species-correct but appealing rounded proportions, "
        "clean coherent limbs/ears/paws/tail or other species appendages, expressive eyes and friendly readable face, "
        "tactile detailed materials with soft organic surface variation rather than cheap plastic, "
        "clean silhouette, polished studio-quality lighting, balanced composition, premium animation-film character quality, "
        "production-ready final asset"
    ),
    negative_constraints=(
        "multiple characters, duplicate character, repeated subject, crowd, duplicated body, malformed appendages, "
        "extra legs, missing legs, fused paws, detached ears, duplicated tail, broken joints, melted shape, warped body, "
        "cheap shiny plastic toy material, flat muddy texture, distorted face, inconsistent eyes, human anatomy on an animal, "
        "obvious AI artifacts, text, watermark, logo"
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


ANIMAL_IMAGE_MODEL = ImageModelConfig(
    model="SG161222/RealVisXL_V5.0",
    inference_steps=28,
    guidance_scale=4.0,
    scheduler="dpm_karras",
    quality_good_text=(
        "natural unretouched camera photograph of an animal, species-correct anatomy, realistic eyes muzzle ears paws legs and tail, "
        "four distinct anatomically connected legs and natural separated paws when applicable, correct joints and biomechanically plausible gait, "
        "irregular natural fur texture, physically plausible pose, ordinary daylight, believable lens rendering and exposure, "
        "moderate contrast and saturation, subtle realistic detail, authentic non-commercial pet-photo appearance"
    ),
    quality_bad_texts=(
        "bad AI animal with malformed paws, extra or missing legs, duplicated limbs, fused toes or legs, misplaced joints or impossible gait",
        "bad AI animal with distorted eyes, muzzle, ears or tail, melted fur, plastic CGI texture or impossible anatomy",
        "overprocessed pet photo with HDR contrast, oversharpening, excessive saturation, airbrushed fur, artificial glow or fake bokeh",
        "commercial advertising retouch with unnaturally perfect fur, exaggerated microcontrast, polished Photoshop look or synthetic depth",
        "bad AI photograph with duplicate subject, warped background, broken perspective, text watermark or obvious artifacts",
    ),
)


UNIFIED_ANIMAL_STYLE_IMAGE_MODEL = ImageModelConfig(
    model="SG161222/RealVisXL_V5.0",
    inference_steps=28,
    guidance_scale=4.0,
    scheduler="dpm_karras",
    quality_good_text=(
        "premium production-ready animal image matching the requested visual style, species-correct anatomy and proportions, "
        "coherent limbs joints face eyes ears horns paws hooves wings fins or tail as applicable, clean silhouette, "
        "high-quality natural or 3D materials appropriate to the request, coherent lighting and geometry"
    ),
    quality_bad_texts=(
        "bad AI animal with malformed anatomy, extra or missing limbs, fused appendages, misplaced joints or impossible pose",
        "bad AI animal with distorted face eyes muzzle beak horns ears tail wings fins paws or hooves",
        "image that ignores the requested photo versus 3D style, broken geometry, duplicated subject, text watermark or obvious artifacts",
    ),
)


MASCOT_IMAGE_MODEL = ImageModelConfig(
    model="playgroundai/playground-v2.5-1024px-aesthetic",
    inference_steps=40,
    guidance_scale=3.0,
    scheduler="edm_dpm",
    quality_good_text=(
        "premium polished cute 3D character render, exactly one subject, species-correct rounded anatomy, coherent limbs paws ears and tail, "
        "expressive clean eyes and face, tactile detailed fur or surface materials, soft studio lighting, clean silhouette, "
        "high-end animation-film mascot quality, production-ready character asset"
    ),
    quality_bad_texts=(
        "bad mascot image with repeated characters, duplicated whole subject, crowd of copies or extra body parts",
        "bad 3D animal with malformed paws, extra or missing legs, broken joints, detached ears, duplicated tail or impossible anatomy",
        "bad glossy plastic toy character with distorted face, inconsistent eyes, flat muddy texture or cheap render look",
        "bad mascot asset with melted geometry, messy silhouette, warped body and obvious AI artifacts",
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
    if name in {ANIMAL_PHOTO_PREMIUM.name, MASCOT_PREMIUM.name}:
        return UNIFIED_ANIMAL_STYLE_IMAGE_MODEL
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

        compiler_output = ""
        if self.provider is not None:
            compiler_output = self.provider.generate(self._compiler_prompt(command)).text.strip()
        return self._finalize_plan(
            command,
            compiler_output=compiler_output,
            has_reference_image=has_reference_image,
        )

    def plan_many(
        self,
        commands: tuple[str, ...] | list[str],
        *,
        has_reference_images: tuple[bool, ...] | list[bool] | None = None,
    ) -> tuple[MediaCommandPlan, ...]:
        """Compile many commands with one model batch when the provider supports it."""
        commands = tuple(command.strip() for command in commands)
        if not commands or any(not command for command in commands):
            raise ValueError("media commands must contain non-empty text")

        if has_reference_images is None:
            reference_flags = (False,) * len(commands)
        else:
            reference_flags = tuple(bool(value) for value in has_reference_images)
            if len(reference_flags) != len(commands):
                raise ValueError("has_reference_images length must match commands")

        outputs = ("",) * len(commands)
        if self.provider is not None:
            prompts = tuple(self._compiler_prompt(command) for command in commands)
            generate_many = getattr(self.provider, "generate_many", None)
            if callable(generate_many):
                batch = generate_many(prompts)
                responses = tuple(batch.responses)
                if len(responses) != len(commands):
                    raise ValueError("media command compiler batch size mismatch")
                outputs = tuple(response.text.strip() for response in responses)
            else:
                outputs = tuple(
                    self.provider.generate(prompt).text.strip()
                    for prompt in prompts
                )

        return tuple(
            self._finalize_plan(
                command,
                compiler_output=compiler_output,
                has_reference_image=reference_flag,
            )
            for command, compiler_output, reference_flag
            in zip(commands, outputs, reference_flags, strict=True)
        )

    @staticmethod
    def _compiler_prompt(command: str) -> str:
        return (
            "MEDIA_COMMAND_COMPILE\n"
            "Translate and compile the user's request into concise ENGLISH visual instructions. "
            "The PROMPT value MUST be English even when USER_COMMAND is Vietnamese or another language. "
            "Preserve every requested subject count, person/object, action, location, clothing, held object, "
            "camera/framing, time of day and visual style. Do not invent or remove story facts. "
            "For full-body or moving animal subjects, preserve the requested action while choosing a natural view with readable limb and paw anatomy and minimal severe self-occlusion. "
            "When camera/framing/lighting are not specified, choose professional production-ready choices that best "
            "express the user's intent without changing the scene or subject. Resolve pronouns and implied references "
            "from the command conservatively. Keep identity/reference instructions if present. Do not lower quality requirements. "
            "Classify the request semantically instead of matching a fixed list of species names. "
            "SUBJECT_CLASS must be one of animal, human, general. STYLE_CLASS must be one of photo, 3d, mascot, illustration, general. "
            "Return exactly four plain-text labeled lines and no Markdown or commentary. "
            "Return ONE image description only; do not propose alternatives, variants, close-ups or second prompts:\n"
            "SUBJECT_CLASS: <animal|human|general>\n"
            "STYLE_CLASS: <photo|3d|mascot|illustration|general>\n"
            "PROMPT: <one complete concise English still-image description>\n"
            "MOTION: <English motion only when video is requested; otherwise leave empty>\n"
            f"USER_COMMAND: {command}"
        )

    def _finalize_plan(
        self,
        command: str,
        *,
        compiler_output: str,
        has_reference_image: bool,
    ) -> MediaCommandPlan:
        mode = self._infer_mode(command)
        parsed: dict[str, str] = {}
        rewritten_prompt = command
        motion_prompt = ""

        if compiler_output:
            parsed = self._parse_compiler_output(compiler_output)
            rewritten_prompt = parsed.get("PROMPT", "").strip()
            motion_prompt = parsed.get("MOTION", "").strip()
            if not rewritten_prompt:
                raise ValueError(
                    "media command compiler did not return a usable English image prompt"
                )
            if self._looks_vietnamese(rewritten_prompt):
                raise ValueError(
                    "media command compiler did not translate PROMPT to English; refusing image generation"
                )

        profile = self._profile_from_semantics(
            parsed.get("SUBJECT_CLASS", ""),
            parsed.get("STYLE_CLASS", ""),
            fallback_command=command,
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

    @classmethod
    def _profile_from_semantics(
        cls,
        subject_class: str,
        style_class: str,
        *,
        fallback_command: str,
    ) -> MediaQualityProfile:
        subject = subject_class.strip().casefold()
        style = style_class.strip().casefold()
        if style in {"3d", "mascot"}:
            return MASCOT_PREMIUM
        if subject == "animal" and style == "photo":
            return ANIMAL_PHOTO_PREMIUM
        if subject == "human" and style == "photo":
            return HUMAN_PHOTO_PREMIUM
        if subject == "animal":
            return ANIMAL_PHOTO_PREMIUM
        if subject == "human":
            return HUMAN_PHOTO_PREMIUM
        return cls._infer_profile_fallback(fallback_command)

    @staticmethod
    def _infer_profile(command: str) -> MediaQualityProfile:
        return MediaCommandPlanner._infer_profile_fallback(command)

    @staticmethod
    def _infer_profile_fallback(command: str) -> MediaQualityProfile:
        """Conservative fallback used only when semantic compiler labels are absent."""
        text = command.casefold()
        mascot_terms = (
            "mascot", "sticker", "cute 3d", "3d cute", "chibi", "hoạt hình 3d",
            "nhân vật dễ thương", "3d dễ thương"
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

    @classmethod
    def _parse_compiler_output(cls, raw: str) -> dict[str, str]:
        """Accept Qwen compiler output in JSON, labeled, or clean English prose form."""
        raw = raw.strip()
        if not raw:
            return {}

        cleaned = raw.strip().strip("`").strip()
        if cleaned.casefold().startswith("json"):
            cleaned = cleaned[4:].lstrip("\n :")

        # JSON first: otherwise a JSON key such as {"prompt": ...} can be
        # accidentally interpreted by the permissive labeled-line parser.
        candidates = [cleaned]
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if 0 <= first_brace < last_brace:
            candidates.insert(0, cleaned[first_brace:last_brace + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            prompt = ""
            for key in ("PROMPT", "prompt", "image_prompt", "imagePrompt", "description"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    prompt = value.strip()
                    break
            motion = ""
            for key in ("MOTION", "motion", "motion_prompt", "motionPrompt"):
                value = payload.get(key)
                if isinstance(value, str):
                    motion = value.strip()
                    break
            if prompt:
                return {"PROMPT": prompt, "MOTION": motion}

        labeled = cls._parse_labeled(cleaned)
        if labeled.get("PROMPT"):
            return labeled

        lines = [
            line.strip().lstrip("-*#> ").strip()
            for line in cleaned.splitlines()
            if line.strip()
        ]
        boilerplate_prefixes = (
            "here is", "here's", "sure", "certainly", "of course",
            "image prompt", "visual prompt", "description",
        )

        # Qwen may answer: "Here is your translated image request: <prompt>".
        if len(lines) == 1 and lines[0].casefold().startswith(boilerplate_prefixes):
            if ":" in lines[0]:
                tail = lines[0].split(":", 1)[1].strip()
                if len(tail) >= 20 and not cls._looks_vietnamese(tail):
                    return {"PROMPT": tail, "MOTION": ""}

        usable = [
            line for line in lines
            if len(line) >= 20
            and not line.casefold().startswith(boilerplate_prefixes)
            and not line.casefold().startswith(("motion:", "motion："))
        ]
        if len(usable) == 1 and not cls._looks_vietnamese(usable[0]):
            return {"PROMPT": usable[0], "MOTION": ""}

        if (
            20 <= len(cleaned) <= 1800
            and "\n" not in cleaned
            and not cleaned.casefold().startswith(boilerplate_prefixes)
            and not cls._looks_vietnamese(cleaned)
        ):
            return {"PROMPT": cleaned, "MOTION": ""}

        return {}


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
        "version": 4,
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
            {
                "profile": ANIMAL_PHOTO_PREMIUM.name,
                "sha256": "792db3d7f02687715c4893fe0b87a1e189b1185c7ea996ba9d4defe0dc844c74",
                "width": 1536,
                "height": 1536,
                "role": "natural unretouched outdoor dog photograph realism reference",
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
            ANIMAL_PHOTO_PREMIUM.name: {
                "must_pass": [
                    "natural unretouched camera-photo realism comparable to the user's accepted outdoor dog reference",
                    "must look captured by a real camera rather than digitally painted, HDR-processed, beauty-retouched or advertising-polished",
                    "species-correct head, muzzle, ears, eyes, torso, legs, paws and tail",
                    "natural irregular fur texture with subtle variation; no airbrushed, plastic, waxy or uniformly perfect fur",
                    "ordinary believable daylight and exposure with moderate contrast, saturation and microcontrast",
                    "realistic lens rendering and depth of field; no artificial glow, fake bokeh or excessive edge sharpening",
                    "physically plausible pose, weight-bearing and ground contact",
                    "clean coherent background and perspective with natural photographic depth",
                    "no extra/missing/fused limbs or paws, warped face, duplicate subject or obvious AI artifacts",
                    "small natural imperfections are acceptable and preferred over overprocessed perfection",
                ],
                "known_rejected_sha256": [
                    "2e476d56320d8ea3b16488d9c276e76e80fe737984e7f90fd4e98f19271f2589",
                ],
            },
            MASCOT_PREMIUM.name: {
                "must_pass": [
                    "match the user's accepted premium cute 3D mascot quality bar, not merely generic 3D technical correctness",
                    "species-correct appealing silhouette and proportions with coherent appendages",
                    "clean expressive eyes and face integrated naturally into the animal/character anatomy",
                    "detailed tactile fur/shell/skin/material surface; NOT cheap shiny plastic-toy material",
                    "clean paws/feet/legs/ears/tail or equivalent appendages with no duplication, fusion or detachment",
                    "soft polished studio-quality 3D lighting with strong depth and readable form",
                    "clean silhouette and background separation suitable for a finished character asset",
                    "no melted geometry, duplicate parts, malformed anatomy, muddy textures or obvious AI artifacts",
                    "production-ready overall appearance at the same quality bar as the supplied 3D mascot references",
                ],
                "known_rejected_sha256": [
                    "2525bafcc74e14444cb64e9c355a882d280417c712a23d1eaadd126aa3c28034",
                ],
            },
        },
    }
