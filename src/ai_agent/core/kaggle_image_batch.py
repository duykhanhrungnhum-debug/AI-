"""Batch image generation for media-production workloads.

One Kaggle GPU session loads the open model once, renders many scenes, verifies
every downloaded image against the worker manifest, and can retry only failed
scenes with new seeds. This is optimized for episodic/video pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import base64
import json
import struct
import textwrap
import time
import zipfile

from .image_model import ImageArtifact
from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker


@dataclass(frozen=True)
class SceneImageRequest:
    scene_id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 768
    height: int = 432
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id is required")
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if self.width <= 0 or self.height <= 0 or self.width % 8 or self.height % 8:
            raise ValueError("dimensions must be positive and divisible by 8")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class SceneImageResult:
    scene_id: str
    artifact: ImageArtifact
    verified: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class BatchImageResult:
    scenes: tuple[SceneImageResult, ...]
    rounds: int

    @property
    def verified(self) -> bool:
        return bool(self.scenes) and all(scene.verified for scene in self.scenes)

    @property
    def failed_scene_ids(self) -> tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes if not scene.verified)


@dataclass
class KaggleBatchImageProvider:
    worker: KaggleGpuWorker
    model: str = "SG161222/RealVisXL_V4.0"
    ip_adapter_model: str = "h94/IP-Adapter"
    ip_adapter_weight: str = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    identity_threshold: float = 0.60
    prompt_alignment_threshold: float = 0.22
    visual_quality_margin_threshold: float = 0.015
    kernel_slug: str = "ai-agent-image-batch"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    inference_steps: int = 28
    guidance_scale: float = 5.5
    min_pixel_std: float = 8.0
    provider: str = "kaggle-gpu-local-model-batch"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if self.inference_steps <= 0 or self.min_pixel_std < 0:
            raise ValueError("inference_steps and min_pixel_std must be valid")
        if not -1.0 <= self.identity_threshold <= 1.0:
            raise ValueError("identity_threshold must be in [-1, 1]")
        if not -1.0 <= self.prompt_alignment_threshold <= 1.0:
            raise ValueError("prompt_alignment_threshold must be in [-1, 1]")
        if not -2.0 <= self.visual_quality_margin_threshold <= 2.0:
            raise ValueError("visual_quality_margin_threshold must be in [-2, 2]")

    def generate_batch(
        self,
        requests: tuple[SceneImageRequest, ...] | list[SceneImageRequest],
        *,
        reference_image: bytes | None = None,
        reference_scale: float = 0.75,
    ) -> BatchImageResult:
        assert_core_invariants()
        requests = tuple(requests)
        self._validate_requests(requests)
        if reference_image is not None and not reference_image:
            raise ValueError("reference_image must not be empty")
        if reference_image is not None and len(reference_image) > 8_000_000:
            raise ValueError("reference_image exceeds 8 MB")
        if not 0.50 <= reference_scale <= 0.95:
            raise ValueError("reference_scale must be between 0.50 and 0.95")
        source = self._build_worker_source(
            requests,
            reference_image=reference_image,
            reference_scale=reference_scale,
        )
        title = self.kernel_slug.replace("-", " ").title()
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=title,
            source=source,
            enable_internet=True,
            is_private=True,
        )
        self._wait()

        archive = self.worker.download_output_file(self.kernel_slug, "images.zip")
        report_bytes = self.worker.download_output_file(self.kernel_slug, "batch_report.json")
        report = self._parse_report(report_bytes)
        entries = report.get("scenes")
        if not isinstance(entries, list):
            raise ValueError("batch report does not contain scenes")
        if report.get("model") != self.model:
            raise ValueError("batch report model does not match configured model")
        if reference_image is not None:
            expected_reference = sha256(reference_image).hexdigest()
            if report.get("reference_sha256") != expected_reference:
                raise ValueError("batch report reference image hash mismatch")
        gpu_name = str(report.get("gpu_name", "")).strip()
        if not gpu_name:
            raise ValueError("batch report does not contain GPU evidence")

        by_id = {str(item.get("scene_id")): item for item in entries if isinstance(item, dict)}
        results: list[SceneImageResult] = []
        hashes: dict[str, list[int]] = {}

        with zipfile.ZipFile(BytesIO(archive), "r") as zipped:
            for index, request in enumerate(requests):
                entry = by_id.get(request.scene_id)
                if entry is None:
                    raise ValueError(f"batch report missing scene: {request.scene_id}")
                filename = str(entry.get("filename", ""))
                if not filename:
                    raise ValueError(f"batch report missing filename: {request.scene_id}")
                try:
                    image = zipped.read(filename)
                except KeyError as exc:
                    raise ValueError(f"batch archive missing file: {filename}") from exc

                digest = sha256(image).hexdigest()
                issues = self._verify_scene(request, entry, image, digest)
                hashes.setdefault(digest, []).append(index)
                artifact = ImageArtifact(
                    data=image,
                    mime_type="image/png",
                    provider=self.provider,
                    model=self.model,
                    evidence=(
                        f"kaggle_kernel:{submission.ref}",
                        f"scene_id:{request.scene_id}",
                        f"image_sha256:{digest}",
                        f"dimensions:{request.width}x{request.height}",
                        f"seed:{entry.get('seed')}",
                        f"pixel_std:{entry.get('pixel_std')}",
                        *((f"identity_score:{float(entry.get('identity_score')):.6f}", f"reference_scale:{float(entry.get('reference_scale')):.3f}") if entry.get("identity_score") is not None else ()),
                        *((f"prompt_alignment_score:{float(entry.get('prompt_alignment_score')):.6f}",) if entry.get("prompt_alignment_score") is not None else ()),
                        *((f"visual_quality_margin:{float(entry.get('visual_quality_margin')):.6f}",) if entry.get("visual_quality_margin") is not None else ()),
                        *((f"visual_defect_score:{float(entry.get('visual_defect_score')):.6f}",) if entry.get("visual_defect_score") is not None else ()),
                        f"safety_blocked:{bool(entry.get('safety_blocked'))}",
                        f"gpu:{gpu_name}",
                        f"model:{self.model}",
                    ),
                )
                results.append(
                    SceneImageResult(
                        scene_id=request.scene_id,
                        artifact=artifact,
                        verified=not issues,
                        issues=tuple(issues),
                    )
                )

        duplicate_indexes = {
            index
            for indexes in hashes.values()
            if len(indexes) > 1
            for index in indexes
        }
        if duplicate_indexes:
            updated: list[SceneImageResult] = []
            for index, result in enumerate(results):
                if index in duplicate_indexes:
                    issues = tuple(dict.fromkeys((*result.issues, "duplicate image content")))
                    updated.append(replace(result, verified=False, issues=issues))
                else:
                    updated.append(result)
            results = updated

        return BatchImageResult(tuple(results), rounds=1)

    def generate_with_retries(
        self,
        requests: tuple[SceneImageRequest, ...] | list[SceneImageRequest],
        *,
        max_rounds: int = 2,
        reference_image: bytes | None = None,
        reference_scale: float = 0.75,
    ) -> BatchImageResult:
        """Retry only failed scenes and refine the next request from verifier feedback."""
        assert_core_invariants()
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        original = tuple(requests)
        self._validate_requests(original)

        latest: dict[str, SceneImageResult] = {}
        pending = original
        rounds = 0
        seen_hashes: dict[str, set[str]] = {request.scene_id: set() for request in original}

        while pending and rounds < max_rounds:
            batch = self.generate_batch(
                pending,
                reference_image=reference_image,
                reference_scale=min(0.92, reference_scale + rounds * 0.05),
            )
            rounds += 1
            next_pending: list[SceneImageRequest] = []

            request_by_id = {request.scene_id: request for request in pending}
            for result in batch.scenes:
                prior = seen_hashes[result.scene_id]
                digest = self._evidence_value(result.artifact.evidence, "image_sha256:")
                issues = list(result.issues)
                if digest in prior and not result.verified:
                    issues.append("retry loop detected: repeated image")
                prior.add(digest)
                result = replace(
                    result,
                    verified=(result.verified and not issues),
                    issues=tuple(dict.fromkeys(issues)),
                )
                latest[result.scene_id] = result

                if not result.verified and rounds < max_rounds:
                    request = request_by_id[result.scene_id]
                    next_pending.append(
                        self._refined_retry_request(
                            request,
                            result.issues,
                            round_index=rounds,
                        )
                    )

            pending = tuple(next_pending)

        ordered = tuple(latest[request.scene_id] for request in original)
        return BatchImageResult(ordered, rounds=rounds)

    @staticmethod
    def _refined_retry_request(
        request: SceneImageRequest,
        issues: tuple[str, ...],
        *,
        round_index: int,
    ) -> SceneImageRequest:
        """Translate verifier failures into a bounded, safer next attempt."""
        issue_text = " | ".join(issues).casefold()
        refinements: list[str] = []
        negatives: list[str] = []

        if "identity similarity" in issue_text or "identity" in issue_text:
            refinements.append(
                "STRICT IDENTITY LOCK: preserve the reference person's face shape, age, hair, "
                "body proportions, costume identity and distinctive features"
            )
            negatives.extend(("different person", "changed face", "face drift", "identity drift"))

        if "prompt alignment" in issue_text:
            refinements.append(
                "STRICT SCENE MATCH: depict exactly the requested subject, action, location, objects and time of day; "
                "do not add unrelated people or events"
            )
            negatives.extend(("unrelated scene", "wrong location", "unrequested people", "wrong action"))

        if "visual quality margin" in issue_text or "visual anatomy quality" in issue_text:
            refinements.append(
                "STRICT HUMAN QUALITY: natural human anatomy, coherent shoulders and elbows, two anatomically plausible hands "
                "when visible, five distinct fingers per visible hand, natural wrists, undistorted face, physically plausible "
                "books/furniture and clean object geometry"
            )
            negatives.extend((
                "bad anatomy", "deformed hands", "malformed hands", "fused fingers", "extra fingers",
                "missing fingers", "extra limbs", "twisted arms", "distorted face", "warped objects",
                "broken perspective", "AI artifacts",
            ))

        if "duplicate image" in issue_text or "retry loop" in issue_text:
            refinements.append(
                "Use a clearly different pose, staging or camera framing while preserving the locked identity "
                "and story facts"
            )
            negatives.append("same composition as previous attempt")

        if "lacks visual variation" in issue_text:
            refinements.append(
                "Increase readable visual structure with clear foreground, subject and background separation, "
                "natural texture and lighting contrast"
            )
            negatives.extend(("flat blank image", "featureless background"))

        if "safety checker blocked" in issue_text:
            refinements.append(
                "Keep the scene non-graphic and safe; remove explicit injury, sexual or otherwise unsafe visual detail "
                "while preserving the harmless story intent"
            )

        if not refinements:
            refinements.append(
                "Preserve all story and continuity facts, improve prompt adherence, anatomy and visual clarity"
            )

        prompt = request.prompt.rstrip(" .") + ". Retry refinement: " + ". ".join(refinements) + "."
        negative_prompt = request.negative_prompt
        if negatives:
            extra = ", ".join(dict.fromkeys(negatives))
            negative_prompt = (negative_prompt.rstrip(" ,") + ", " + extra).strip(" ,")

        return replace(
            request,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=request.seed + max(1, round_index),
        )

    def _wait(self) -> None:
        for _ in range(self.max_poll_attempts):
            status = self.worker.status(self.kernel_slug)
            if status.terminal:
                if not status.successful:
                    logs = ""
                    try:
                        logs = self.worker.logs(self.kernel_slug)
                    except Exception:
                        pass
                    detail = (status.failure_message or logs or status.status).strip()
                    if len(detail) > 8000:
                        detail = "...[tail]\n" + detail[-8000:]
                    raise RuntimeError(f"Kaggle image batch failed: {detail}")
                return
            if self.poll_interval:
                time.sleep(self.poll_interval)
        raise TimeoutError("Timed out waiting for Kaggle image batch")

    @staticmethod
    def _validate_requests(requests: tuple[SceneImageRequest, ...]) -> None:
        if not requests:
            raise ValueError("at least one scene is required")
        ids = [request.scene_id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("scene_id values must be unique")

    def _verify_scene(
        self,
        request: SceneImageRequest,
        entry: dict,
        image: bytes,
        digest: str,
    ) -> list[str]:
        issues: list[str] = []
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            issues.append("output is not a PNG")
        if entry.get("image_sha256") != digest:
            issues.append("image hash mismatch")

        try:
            width, height = self._png_dimensions(image)
        except ValueError:
            width, height = 0, 0
            issues.append("invalid PNG header")
        if width != request.width or height != request.height:
            issues.append(
                f"dimension mismatch: expected {request.width}x{request.height}, got {width}x{height}"
            )

        try:
            pixel_std = float(entry.get("pixel_std", 0))
        except (TypeError, ValueError):
            pixel_std = 0.0
        if pixel_std < self.min_pixel_std:
            issues.append(f"image lacks visual variation: pixel_std={pixel_std:.3f}")
        if entry.get("safety_blocked") is True:
            issues.append("safety checker blocked generated image")
        if entry.get("identity_score") is not None:
            try:
                identity_score = float(entry.get("identity_score"))
            except (TypeError, ValueError):
                identity_score = -1.0
            if identity_score < self.identity_threshold:
                issues.append(
                    f"identity similarity below threshold: {identity_score:.4f} < {self.identity_threshold:.4f}"
                )

        try:
            prompt_alignment = float(entry.get("prompt_alignment_score"))
        except (TypeError, ValueError):
            prompt_alignment = -1.0
        if prompt_alignment < self.prompt_alignment_threshold:
            issues.append(
                f"prompt alignment below threshold: {prompt_alignment:.4f} < {self.prompt_alignment_threshold:.4f}"
            )

        try:
            visual_quality_margin = float(entry.get("visual_quality_margin"))
        except (TypeError, ValueError):
            visual_quality_margin = -2.0
        if visual_quality_margin < self.visual_quality_margin_threshold:
            issues.append(
                "visual quality margin below threshold: "
                f"{visual_quality_margin:.4f} < {self.visual_quality_margin_threshold:.4f}"
            )
        return issues

    @staticmethod
    def _png_dimensions(data: bytes) -> tuple[int, int]:
        if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("invalid PNG")
        if data[12:16] != b"IHDR":
            raise ValueError("PNG missing IHDR")
        return struct.unpack(">II", data[16:24])

    @staticmethod
    def _parse_report(data: bytes) -> dict:
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("batch report is invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("batch report must be an object")
        return parsed

    @staticmethod
    def _evidence_value(evidence: tuple[str, ...], prefix: str) -> str:
        for item in evidence:
            if item.startswith(prefix):
                return item[len(prefix):]
        return ""

    def _build_worker_source(
        self,
        requests: tuple[SceneImageRequest, ...],
        *,
        reference_image: bytes | None = None,
        reference_scale: float = 0.75,
    ) -> str:
        config = {
            "model": self.model,
            "ip_adapter_model": self.ip_adapter_model,
            "ip_adapter_weight": self.ip_adapter_weight,
            "reference_b64": base64.b64encode(reference_image).decode("ascii") if reference_image else None,
            "reference_sha256": sha256(reference_image).hexdigest() if reference_image else None,
            "reference_scale": reference_scale,
            "identity_threshold": self.identity_threshold,
            "prompt_alignment_threshold": self.prompt_alignment_threshold,
            "visual_quality_margin_threshold": self.visual_quality_margin_threshold,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "scenes": [
                {
                    "scene_id": request.scene_id,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": request.width,
                    "height": request.height,
                    "seed": request.seed,
                }
                for request in requests
            ],
        }
        config_json = json.dumps(config, ensure_ascii=False)
        return textwrap.dedent(
            f"""
            from __future__ import annotations

            import base64
            from hashlib import sha256
            from io import BytesIO
            import json
            import subprocess
            import sys
            from pathlib import Path
            import zipfile

            CONFIG = json.loads({config_json!r})

            try:
                import numpy as np
                import torch
                import torch.nn.functional as F
                from PIL import Image
                from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
                from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "diffusers<1", "transformers<5", "accelerate<2", "safetensors", "numpy", "Pillow",
                ])
                import numpy as np
                import torch
                import torch.nn.functional as F
                from PIL import Image
                from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
                from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            adapter_weight = str(CONFIG.get("ip_adapter_weight") or "")
            is_sdxl = "sdxl" in adapter_weight.casefold() or "xl" in str(CONFIG["model"]).casefold()
            image_encoder = None
            if CONFIG.get("reference_b64") and is_sdxl:
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    CONFIG["ip_adapter_model"],
                    subfolder="models/image_encoder",
                    torch_dtype=torch.float16,
                )
            pipe_kwargs = {{"torch_dtype": torch.float16}}
            if image_encoder is not None:
                pipe_kwargs["image_encoder"] = image_encoder
            pipe = AutoPipelineForText2Image.from_pretrained(
                CONFIG["model"],
                **pipe_kwargs,
            )
            if "face" in adapter_weight.casefold():
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )

            reference_image = None
            reference_embedding = None
            clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            clip_vision = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).eval()
            clip_text = CLIPTextModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).eval()

            def image_embedding(image):
                values = clip_image_processor(images=image, return_tensors="pt").pixel_values
                with torch.no_grad():
                    vector = clip_vision(pixel_values=values).image_embeds[0].float()
                return F.normalize(vector, dim=0)

            def text_embedding(text):
                values = clip_tokenizer(
                    [text], return_tensors="pt", padding=True, truncation=True
                )
                with torch.no_grad():
                    vector = clip_text(**values).text_embeds[0].float()
                return F.normalize(vector, dim=0)

            if CONFIG.get("reference_b64"):
                reference_bytes = base64.b64decode(CONFIG["reference_b64"])
                if sha256(reference_bytes).hexdigest() != CONFIG["reference_sha256"]:
                    raise RuntimeError("reference image hash mismatch inside Kaggle worker")
                reference_image = Image.open(BytesIO(reference_bytes)).convert("RGB")
                pipe.load_ip_adapter(
                    CONFIG["ip_adapter_model"],
                    subfolder="sdxl_models" if is_sdxl else "models",
                    weight_name=CONFIG["ip_adapter_weight"],
                )
                pipe.set_ip_adapter_scale(float(CONFIG["reference_scale"]))
                reference_embedding = image_embedding(reference_image)

            # SDXL + IP-Adapter can exceed T4 VRAM when kept fully resident.
            # CPU offload trades a little latency for much safer memory use.
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

            quality_good_text = (
                "high quality realistic photograph, natural human anatomy, coherent shoulders elbows wrists, "
                "realistic hands and fingers, undistorted face, physically plausible body, clean detailed objects"
            )
            quality_bad_texts = (
                "bad AI generated image with deformed hands, malformed fingers, fused fingers, extra fingers, missing fingers",
                "bad AI generated image with extra limbs, twisted arms, broken anatomy, distorted face",
                "bad AI generated image with warped books, melted furniture, broken perspective, obvious visual artifacts",
            )
            quality_good_embedding = text_embedding(quality_good_text)
            quality_bad_embeddings = tuple(text_embedding(text) for text in quality_bad_texts)

            output_dir = Path("/kaggle/working/batch_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            reports = []

            for index, scene in enumerate(CONFIG["scenes"]):
                generator = torch.Generator(device="cpu").manual_seed(int(scene["seed"]))
                generation = {{
                    "prompt": scene["prompt"],
                    "negative_prompt": scene["negative_prompt"] or None,
                    "width": int(scene["width"]),
                    "height": int(scene["height"]),
                    "num_inference_steps": int(CONFIG["inference_steps"]),
                    "guidance_scale": float(CONFIG["guidance_scale"]),
                    "generator": generator,
                }}
                if reference_image is not None:
                    generation["ip_adapter_image"] = reference_image
                result = pipe(**generation)
                image = result.images[0]
                safety_blocked = None
                detected = getattr(result, "nsfw_content_detected", None)
                if isinstance(detected, (list, tuple)) and detected:
                    safety_blocked = bool(detected[0])
                generated_embedding = image_embedding(image)
                identity_score = None
                if reference_embedding is not None:
                    identity_score = float(torch.dot(reference_embedding, generated_embedding).item())
                prompt_alignment_score = float(
                    torch.dot(text_embedding(scene["prompt"]), generated_embedding).item()
                )
                visual_quality_score = float(
                    torch.dot(quality_good_embedding, generated_embedding).item()
                )
                visual_defect_score = max(
                    float(torch.dot(bad_embedding, generated_embedding).item())
                    for bad_embedding in quality_bad_embeddings
                )
                visual_quality_margin = visual_quality_score - visual_defect_score
                filename = f"scene_{{index:04d}}.png"
                path = output_dir / filename
                image.save(path, format="PNG")
                data = path.read_bytes()
                pixels = np.asarray(image, dtype=np.float32)

                reports.append({{
                    "scene_id": scene["scene_id"],
                    "filename": filename,
                    "image_sha256": sha256(data).hexdigest(),
                    "prompt_sha256": sha256(scene["prompt"].encode("utf-8")).hexdigest(),
                    "width": image.width,
                    "height": image.height,
                    "seed": int(scene["seed"]),
                    "pixel_mean": float(pixels.mean()),
                    "pixel_std": float(pixels.std()),
                    "pixel_min": float(pixels.min()),
                    "pixel_max": float(pixels.max()),
                    "identity_score": identity_score,
                    "prompt_alignment_score": prompt_alignment_score,
                    "visual_quality_score": visual_quality_score,
                    "visual_defect_score": visual_defect_score,
                    "visual_quality_margin": visual_quality_margin,
                    "reference_scale": float(CONFIG["reference_scale"]) if reference_image is not None else None,
                    "safety_blocked": safety_blocked,
                }})

            archive_path = Path("/kaggle/working/images.zip")
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(output_dir.glob("*.png")):
                    archive.write(path, arcname=path.name)

            batch_report = {{
                "model": CONFIG["model"],
                "reference_sha256": CONFIG.get("reference_sha256"),
                "gpu_name": gpu_name,
                "scene_count": len(reports),
                "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
                "scenes": reports,
            }}
            Path("/kaggle/working/batch_report.json").write_text(
                json.dumps(batch_report, indent=2) + "\\n",
                encoding="utf-8",
            )
            print("AI_AGENT_IMAGE_BATCH_OK")
            print(json.dumps(batch_report, indent=2))
            """
        ).strip() + "\n"



@dataclass(frozen=True)
class SemanticImageBatchItem:
    item_id: str
    command: str
    seed: int
    fallback_profile: str = "general_premium"

    def __post_init__(self) -> None:
        if not self.item_id.strip() or not self.command.strip():
            raise ValueError("item_id and command are required")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class SemanticImageBatchOutput:
    artifacts: dict[str, ImageArtifact]
    reports: dict[str, dict]
    batch_report: dict


@dataclass
class KaggleSemanticImageBatchProvider:
    """Semantic planner -> one image model load -> one VLM load for the whole batch."""

    worker: KaggleGpuWorker
    model: str = "SG161222/RealVisXL_V5.0"
    model_variant: str | None = "fp16"
    kernel_slug: str = "ai-agent-semantic-image-batch"
    poll_interval: float = 3.0
    max_poll_attempts: int = 600
    inference_steps: int = 28
    guidance_scale: float = 4.0
    planner_fast_model: str = "Qwen/Qwen3-0.6B"
    planner_final_model: str = "Qwen/Qwen3-1.7B"
    vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    vlm_min_score: float = 9.0
    profile_positive_constraints: dict[str, str] | None = None
    profile_negative_constraints: dict[str, str] | None = None
    profile_rubrics: dict[str, tuple[str, ...]] | None = None
    profile_dimensions: dict[str, tuple[int, int]] | None = None
    provider: str = "kaggle-gpu-semantic-image-batch"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if not 0 <= self.vlm_min_score <= 10:
            raise ValueError("vlm_min_score must be in [0, 10]")
        maps = (
            self.profile_positive_constraints,
            self.profile_negative_constraints,
            self.profile_rubrics,
            self.profile_dimensions,
        )
        if any(value is None for value in maps):
            raise ValueError("semantic profile maps are required")
        keys = set(self.profile_positive_constraints or {})
        if not keys:
            raise ValueError("semantic profile maps must not be empty")
        if any(set(value or {}) != keys for value in maps[1:]):
            raise ValueError("semantic profile maps must have matching keys")

    def generate_batch(
        self,
        items: tuple[SemanticImageBatchItem, ...] | list[SemanticImageBatchItem],
    ) -> SemanticImageBatchOutput:
        assert_core_invariants()
        items = tuple(items)
        if not items:
            raise ValueError("at least one semantic image item is required")
        ids = [item.item_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic image item ids must be unique")

        source = self._build_worker_source(items)
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=self.kernel_slug.replace("-", " ").title(),
            source=source,
            enable_internet=True,
            is_private=True,
        )
        for _ in range(self.max_poll_attempts):
            status = self.worker.status(self.kernel_slug)
            if status.terminal:
                if not status.successful:
                    logs = ""
                    try:
                        logs = self.worker.logs(self.kernel_slug)
                    except Exception:
                        pass
                    detail = (status.failure_message or logs or status.status).strip()
                    if len(detail) > 8000:
                        detail = "...[tail]\n" + detail[-8000:]
                    raise RuntimeError(f"Kaggle semantic image batch failed: {detail}")
                break
            if self.poll_interval:
                time.sleep(self.poll_interval)
        else:
            raise TimeoutError("Timed out waiting for Kaggle semantic image batch")

        report = self._parse_report(
            self.worker.download_output_file(self.kernel_slug, "batch_report.json")
        )
        if report.get("model") != self.model:
            raise ValueError("semantic batch report model mismatch")
        reports = report.get("items")
        if not isinstance(reports, dict):
            raise ValueError("semantic batch report has no item map")
        gpu_name = str(report.get("gpu_name", "")).strip()
        if not gpu_name:
            raise ValueError("semantic batch report has no GPU evidence")

        artifacts: dict[str, ImageArtifact] = {}
        normalized_reports: dict[str, dict] = {}
        for item in items:
            entry = reports.get(item.item_id)
            if not isinstance(entry, dict):
                raise ValueError(f"semantic batch report missing item: {item.item_id}")
            image = self.worker.download_output_file(
                self.kernel_slug,
                f"{item.item_id}.png",
            )
            digest = sha256(image).hexdigest()
            if entry.get("image_sha256") != digest:
                raise ValueError(f"semantic batch hash mismatch: {item.item_id}")
            try:
                width, height = KaggleBatchImageProvider._png_dimensions(image)
            except ValueError as exc:
                raise ValueError(f"semantic batch invalid PNG: {item.item_id}") from exc
            if width != int(entry.get("width", 0)) or height != int(entry.get("height", 0)):
                raise ValueError(f"semantic batch dimensions mismatch: {item.item_id}")
            profile = str(entry.get("semantic_profile", "")).strip()
            if profile not in set(self.profile_rubrics or {}):
                raise ValueError(f"semantic batch profile invalid: {item.item_id}")
            normalized_reports[item.item_id] = entry
            artifacts[item.item_id] = ImageArtifact(
                data=image,
                mime_type="image/png",
                provider=self.provider,
                model=self.model,
                evidence=(
                    f"kaggle_kernel:{submission.ref}",
                    f"item_id:{item.item_id}",
                    f"image_sha256:{digest}",
                    f"dimensions:{width}x{height}",
                    f"seed:{item.seed}",
                    f"semantic_profile:{profile}",
                    f"subject_class:{entry.get('subject_class')}",
                    f"style_class:{entry.get('style_class')}",
                    f"planner_model:{entry.get('planner_model')}",
                    f"vlm_model:{self.vlm_model}",
                    f"gpu:{gpu_name}",
                    f"model:{self.model}",
                    f"model_variant:{self.model_variant or 'default'}",
                ),
            )
        return SemanticImageBatchOutput(
            artifacts=artifacts,
            reports=normalized_reports,
            batch_report=report,
        )

    @staticmethod
    def _parse_report(data: bytes) -> dict:
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("semantic batch report is invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("semantic batch report must be an object")
        return parsed

    def _build_worker_source(
        self,
        items: tuple[SemanticImageBatchItem, ...],
    ) -> str:
        config = {
            "model": self.model,
            "model_variant": self.model_variant,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "planner_fast_model": self.planner_fast_model,
            "planner_final_model": self.planner_final_model,
            "vlm_model": self.vlm_model,
            "vlm_min_score": self.vlm_min_score,
            "profile_positive_constraints": self.profile_positive_constraints,
            "profile_negative_constraints": self.profile_negative_constraints,
            "profile_rubrics": {
                key: list(value)
                for key, value in (self.profile_rubrics or {}).items()
            },
            "profile_dimensions": {
                key: list(value)
                for key, value in (self.profile_dimensions or {}).items()
            },
            "items": [
                {
                    "item_id": item.item_id,
                    "command": item.command,
                    "seed": item.seed,
                    "fallback_profile": item.fallback_profile,
                }
                for item in items
            ],
        }
        config_json = json.dumps(config, ensure_ascii=False)
        template = r"""
from __future__ import annotations

from hashlib import sha256
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIG = json.loads(__CONFIG_JSON__)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    import torch
    from PIL import Image
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "diffusers<1", "transformers>=4.57,<5", "accelerate<2", "safetensors", "Pillow<13",
    ])
    import torch
    from PIL import Image
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available")

started = time.perf_counter()
gpu_name = torch.cuda.get_device_name(0)


def normalize_subject(value):
    token = (value or "").strip().casefold().replace("_", " ").replace("-", " ")
    if token in {"animal", "animals", "creature", "creatures", "pet", "wildlife"}:
        return "animal"
    if token in {"human", "person", "people", "man", "woman"}:
        return "human"
    if token in {"general", "object", "scene", "other"}:
        return "general"
    return ""


def normalize_style(value):
    token = (value or "").strip().casefold().replace("_", " ").replace("-", " ")
    if token in {"photo", "photograph", "photorealistic", "realistic photo", "real photo"}:
        return "photo"
    if token in {"3d", "3 d", "3d render", "cgi", "cg"}:
        return "3d"
    if token in {"mascot", "3d mascot"}:
        return "mascot"
    if token in {"illustration", "drawing", "painted", "painting"}:
        return "illustration"
    if token in {"general", "other"}:
        return "general"
    return ""


def parse_compiled_result(raw):
    raw = (raw or "").strip().strip(chr(96)).strip()
    result = {"prompt": "", "subject_class": "", "style_class": "", "subject_name": ""}
    if not raw:
        return result
    cleaned = raw
    if cleaned.casefold().startswith("json"):
        cleaned = cleaned[4:].lstrip("\n :")
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    candidates = [cleaned]
    if 0 <= first_brace < last_brace:
        candidates.insert(0, cleaned[first_brace:last_brace + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        result["prompt"] = str(
            payload.get("PROMPT")
            or payload.get("prompt")
            or payload.get("image_prompt")
            or payload.get("description")
            or ""
        ).strip()
        result["subject_class"] = normalize_subject(
            str(payload.get("SUBJECT_CLASS") or payload.get("subject_class") or "")
        )
        result["style_class"] = normalize_style(
            str(payload.get("STYLE_CLASS") or payload.get("style_class") or "")
        )
        result["subject_name"] = str(
            payload.get("SUBJECT_NAME")
            or payload.get("subject_name")
            or payload.get("SPECIES")
            or payload.get("species")
            or payload.get("ENTITY")
            or payload.get("entity")
            or ""
        ).strip()
        if result["prompt"]:
            return result

    aliases = {
        "PROMPT": "prompt",
        "PROMT": "prompt",
        "PROMP": "prompt",
        "DESCRIPTION": "prompt",
        "SUBJECTCLASS": "subject_class",
        "SUBJECT": "subject_class",
        "STYLECLASS": "style_class",
        "STYLE": "style_class",
        "SUBJECTNAME": "subject_name",
        "SPECIES": "subject_name",
        "ENTITY": "subject_name",
    }
    lines = []
    for line in cleaned.replace("：", ":").splitlines():
        line = line.strip().lstrip("-*#> ").strip()
        if not line:
            continue
        lines.append(line)
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = "".join(ch for ch in key.strip().upper() if ch.isalpha())
        canonical = aliases.get(key)
        if canonical and value.strip() and not result[canonical]:
            result[canonical] = value.strip().strip(chr(96)).strip()

    result["subject_class"] = normalize_subject(result["subject_class"])
    result["style_class"] = normalize_style(result["style_class"])

    if not result["prompt"]:
        boilerplate = (
            "here is", "here's", "sure", "certainly", "output", "result",
            "subject_class", "style_class", "user_command",
        )
        usable = []
        for line in lines:
            lowered = line.casefold()
            if lowered.startswith(boilerplate):
                if ":" in line and lowered.startswith(("here is", "here's")):
                    tail = line.split(":", 1)[1].strip()
                    if len(tail) >= 20:
                        usable.append(tail)
                continue
            if len(line) >= 20:
                usable.append(line)
        if usable:
            result["prompt"] = max(usable, key=len).strip().strip(chr(96)).strip()

    return result


def parse_class_pair(raw):
    text = (raw or "").strip().casefold()
    if not text:
        return "", ""
    cleaned = text.replace("subject_class", "").replace("style_class", "")
    cleaned = cleaned.replace(":", " ").replace("|", " ").replace(",", " ").replace("/", " ")
    tokens = [token.strip() for token in cleaned.split() if token.strip()]
    subject = ""
    style = ""
    for token in tokens:
        if not subject:
            subject = normalize_subject(token)
        if not style:
            style = normalize_style(token)
    return subject, style


def looks_vietnamese(text):
    chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    return any(ch in chars for ch in text.casefold())


def is_placeholder_prompt(text):
    normalized = " ".join((text or "").strip().casefold().split())
    if not normalized:
        return True
    placeholders = {
        "english image description",
        "english prompt",
        "image description",
        "complete concise english image prompt",
        "english image prompt",
        "<english image description>",
        "<english image prompt>",
    }
    if normalized in placeholders:
        return True
    if normalized.startswith(("english image description", "image description here", "prompt goes here")):
        return True
    if "<" in normalized or ">" in normalized:
        return True
    return False


def valid_subject_name(value):
    name = " ".join((value or "").strip().split())
    if not (2 <= len(name) <= 100):
        return False
    if looks_vietnamese(name):
        return False
    if is_placeholder_prompt(name):
        return False
    return True


def choose_profile(subject, style, fallback):
    if style in ("3d", "mascot"):
        return "mascot_premium"
    if subject == "animal" and style == "photo":
        return "animal_photo_premium"
    if subject == "human" and style == "photo":
        return "human_photo_premium"
    if subject == "animal":
        return "animal_photo_premium"
    if subject == "human":
        return "human_photo_premium"
    return fallback if fallback in CONFIG["profile_rubrics"] else "general_premium"


def compile_items(model_name, pending):
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    ready = time.perf_counter()
    outputs = {}
    valid_subjects = {"animal", "human", "general"}
    valid_styles = {"photo", "3d", "mascot", "illustration", "general"}

    def generate_text(instruction, max_new_tokens):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.03,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        del generated, inputs
        return text

    for item in pending:
        gen_started = time.perf_counter()
        instruction = (
            "MEDIA_COMMAND_COMPILE\n"
            "Understand USER_COMMAND semantically without a fixed species list. "
            "Translate it into ONE concise ENGLISH still-image description and preserve the exact species, subject count, action, location and requested style. "
            "SUBJECT_CLASS must be exactly animal, human, or general. "
            "STYLE_CLASS must be exactly photo, 3d, mascot, illustration, or general. "
            "Return exactly three lines and no commentary:\n"
            "SUBJECT_CLASS: animal|human|general\n"
            "STYLE_CLASS: photo|3d|mascot|illustration|general\n"
            "PROMPT: English image description\n"
            "USER_COMMAND: " + item["command"]
        )
        raw = generate_text(instruction, 180)
        parsed = parse_compiled_result(raw)
        retry_raw = ""
        class_raw = ""
        retries = 0

        prompt_valid = (
            len(parsed["prompt"]) >= 20
            and not looks_vietnamese(parsed["prompt"])
            and "USER_COMMAND" not in parsed["prompt"]
        )
        if not prompt_valid:
            retries += 1
            retry_raw = generate_text(
                "Translate this image request into one complete concise ENGLISH image prompt. "
                "Preserve the exact species, count, action, location and requested photo/3D style. "
                "Return ONLY the English prompt sentence and nothing else. REQUEST: " + item["command"],
                140,
            )
            retry_parsed = parse_compiled_result(retry_raw)
            retry_prompt = (retry_parsed["prompt"] or retry_raw).strip().strip(chr(96)).strip()
            if (
                len(retry_prompt) >= 20
                and not looks_vietnamese(retry_prompt)
                and "USER_COMMAND" not in retry_prompt
            ):
                parsed["prompt"] = retry_prompt
                prompt_valid = True

        classes_valid = (
            parsed["subject_class"] in valid_subjects
            and parsed["style_class"] in valid_styles
        )
        routing_fallback_used = False
        if prompt_valid and not classes_valid:
            retries += 1
            class_raw = generate_text(
                "Classify the image request semantically. Do not name the species. "
                "Return ONLY two tokens separated by a vertical bar: "
                "first token animal, human, or general; second token photo, 3d, mascot, illustration, or general. "
                "REQUEST: " + item["command"] + "\nENGLISH_PROMPT: " + parsed["prompt"],
                24,
            )
            subject, style = parse_class_pair(class_raw)
            if subject in valid_subjects:
                parsed["subject_class"] = subject
            if style in valid_styles:
                parsed["style_class"] = style
            classes_valid = (
                parsed["subject_class"] in valid_subjects
                and parsed["style_class"] in valid_styles
            )

        if prompt_valid and not classes_valid:
            routing_fallback_used = True
            fallback = item.get("fallback_profile") or "general_premium"
            parsed["subject_class"] = (
                "human" if fallback == "human_photo_premium"
                else "animal" if fallback == "animal_photo_premium"
                else "general"
            )
            parsed["style_class"] = (
                "3d" if fallback == "mascot_premium"
                else "photo" if fallback in {"human_photo_premium", "animal_photo_premium"}
                else "general"
            )
            classes_valid = True

        valid = prompt_valid and classes_valid
        outputs[item["item_id"]] = {
            "valid": valid,
            "raw": raw,
            "retry_raw": retry_raw,
            "class_raw": class_raw,
            "parsed": parsed,
            "routing_fallback_used": routing_fallback_used,
            "retry_count": retries,
            "generate_seconds": round(time.perf_counter() - gen_started, 3),
        }

    timing = {
        "model_load_seconds": round(ready - load_started, 3),
        "total_seconds": round(time.perf_counter() - load_started, 3),
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, timing


planner_outputs, planner_fast_timing = compile_items(CONFIG["planner_fast_model"], CONFIG["items"])
failed_ids = {
    item["item_id"]
    for item in CONFIG["items"]
    if not planner_outputs[item["item_id"]]["valid"]
}
planner_final_timing = None
if failed_ids:
    pending = [item for item in CONFIG["items"] if item["item_id"] in failed_ids]
    fallback_outputs, planner_final_timing = compile_items(CONFIG["planner_final_model"], pending)
    planner_outputs.update(fallback_outputs)

compiled_items = []
for item in CONFIG["items"]:
    entry = planner_outputs[item["item_id"]]
    if not entry["valid"]:
        raise RuntimeError("semantic planner failed for item " + item["item_id"])
    parsed = entry["parsed"]
    profile = choose_profile(
        parsed["subject_class"],
        parsed["style_class"],
        item["fallback_profile"],
    )
    positive = CONFIG["profile_positive_constraints"][profile]
    negative = CONFIG["profile_negative_constraints"][profile]
    width, height = CONFIG["profile_dimensions"][profile]
    compiled_items.append({
        **item,
        "prompt": parsed["prompt"].rstrip(" .") + ". Quality requirements: " + positive.rstrip(" .") + ".",
        "negative_prompt": negative,
        "subject_class": parsed["subject_class"],
        "style_class": parsed["style_class"],
        "semantic_profile": profile,
        "planner_raw": entry["raw"],
        "planner_retry_raw": entry.get("retry_raw", ""),
        "planner_class_raw": entry.get("class_raw", ""),
        "routing_fallback_used": bool(entry.get("routing_fallback_used")),
        "planner_retry_count": int(entry.get("retry_count", 0)),
        "planner_model": CONFIG["planner_final_model"] if item["item_id"] in failed_ids else CONFIG["planner_fast_model"],
        "planner_generate_seconds": entry["generate_seconds"],
        "width": int(width),
        "height": int(height),
    })

image_load_started = time.perf_counter()
pipe_kwargs = {"torch_dtype": torch.float16}
if CONFIG.get("model_variant"):
    pipe_kwargs["variant"] = CONFIG["model_variant"]
pipe = AutoPipelineForText2Image.from_pretrained(CONFIG["model"], **pipe_kwargs)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
)
pipe = pipe.to("cuda")
pipe.enable_vae_slicing()
image_ready = time.perf_counter()

image_reports = {}
for item in compiled_items:
    generate_started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(int(item["seed"]))
    result = pipe(
        prompt=item["prompt"],
        negative_prompt=item["negative_prompt"] or None,
        width=item["width"],
        height=item["height"],
        num_inference_steps=int(CONFIG["inference_steps"]),
        guidance_scale=float(CONFIG["guidance_scale"]),
        generator=generator,
    )
    image = result.images[0]
    output_path = Path("/kaggle/working") / (item["item_id"] + ".png")
    image.save(output_path, format="PNG")
    data = output_path.read_bytes()
    image_reports[item["item_id"]] = {
        **item,
        "image_sha256": sha256(data).hexdigest(),
        "model": CONFIG["model"],
        "model_variant": CONFIG.get("model_variant"),
        "image_generation_seconds": round(time.perf_counter() - generate_started, 3),
    }

del pipe
gc.collect()
torch.cuda.empty_cache()

qa_load_started = time.perf_counter()
qa_model = Qwen3VLForConditionalGeneration.from_pretrained(
    CONFIG["vlm_model"],
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
qa_processor = AutoProcessor.from_pretrained(CONFIG["vlm_model"])
qa_model.eval()
qa_ready = time.perf_counter()

for item in compiled_items:
    image = Image.open(Path("/kaggle/working") / (item["item_id"] + ".png")).convert("RGB")
    image.thumbnail((1024, 1024))
    width, height = image.size
    views = [
        image,
        image.crop((0, 0, width, max(1, height // 2))),
        image.crop((0, height // 2, width, height)),
    ]
    rubric = "\n".join(
        "- " + value
        for value in CONFIG["profile_rubrics"][item["semantic_profile"]]
    )
    instruction = (
        "You are the strict final visual-quality inspector for a production image pipeline. "
        "Judge only what is visible. PROFILE: " + item["semantic_profile"]
        + "; ORIGINAL PROMPT: " + item["prompt"]
        + "; EXPECTED MAIN SUBJECT COUNT: 1; QUALITY RUBRIC: " + rubric + ". "
        "Inspect anatomy/geometry, appendages, face/eyes, materials, lighting, perspective, texture continuity, background coherence and AI artifacts. "
        "For animal photo profile require real-camera realism and correct species anatomy. "
        "For mascot profile require the requested premium 3D style, species-correct anatomy, clean appendages and non-cheap materials. "
        "Score 0-10 for q=quality,m=prompt match,s=structure,d=detail,a=aesthetic,c=composition,b=benchmark. "
        "Set p=true ONLY if every score is at least " + str(CONFIG["vlm_min_score"])
        + ", subject count is correct, and x/f/u are empty. If uncertain about a critical region, p=false. "
        "Return ONLY compact JSON: "
        '{"p":true,"q":9,"m":9,"s":9,"d":9,"a":9,"c":9,"b":9,"n":1,"x":[],"f":[],"u":[],"i":[]}'
    )
    content = [{"type": "image", "image": view} for view in views]
    content.append({"type": "text", "text": instruction})
    qa_inputs = qa_processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(qa_model.device)
    review_started = time.perf_counter()
    with torch.inference_mode():
        generated = qa_model.generate(
            **qa_inputs,
            max_new_tokens=180,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.03,
        )
    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(qa_inputs.input_ids, generated)
    ]
    review_text = qa_processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if not review_text:
        raise RuntimeError("empty VLM review for " + item["item_id"])
    image_reports[item["item_id"]]["review_text"] = review_text
    image_reports[item["item_id"]]["qa_review_seconds"] = round(
        time.perf_counter() - review_started,
        3,
    )

report = {
    "gpu_name": gpu_name,
    "model": CONFIG["model"],
    "model_variant": CONFIG.get("model_variant"),
    "planner_fast_timing": planner_fast_timing,
    "planner_final_timing": planner_final_timing,
    "image_model_load_seconds": round(image_ready - image_load_started, 3),
    "qa_model_load_seconds": round(qa_ready - qa_load_started, 3),
    "worker_total_seconds": round(time.perf_counter() - started, 3),
    "items": image_reports,
}
Path("/kaggle/working/batch_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print("AI_AGENT_SEMANTIC_IMAGE_BATCH_OK")
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
        return textwrap.dedent(template).replace(
            "__CONFIG_JSON__",
            repr(config_json),
        ).strip() + "\n"
