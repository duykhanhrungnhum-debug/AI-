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
