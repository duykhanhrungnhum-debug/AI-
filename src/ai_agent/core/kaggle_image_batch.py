"""Batch image generation for media-production workloads.

One Kaggle GPU session loads the open model once, renders many scenes, verifies
every downloaded image against the worker manifest, and can retry only failed
scenes with new seeds. This is optimized for episodic/video pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
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
    model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    kernel_slug: str = "ai-agent-image-batch"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    inference_steps: int = 16
    guidance_scale: float = 7.0
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

    def generate_batch(self, requests: tuple[SceneImageRequest, ...] | list[SceneImageRequest]) -> BatchImageResult:
        assert_core_invariants()
        requests = tuple(requests)
        self._validate_requests(requests)
        source = self._build_worker_source(requests)
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
    ) -> BatchImageResult:
        """Retry only failed scenes; each retry changes the seed to avoid loops."""
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
            batch = self.generate_batch(pending)
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
                    next_pending.append(replace(request, seed=request.seed + rounds))

            pending = tuple(next_pending)

        ordered = tuple(latest[request.scene_id] for request in original)
        return BatchImageResult(ordered, rounds=rounds)

    def _wait(self) -> None:
        for _ in range(self.max_poll_attempts):
            status = self.worker.status(self.kernel_slug)
            if status.terminal:
                if not status.successful:
                    raise RuntimeError(
                        f"Kaggle image batch failed: {status.status} {status.failure_message}"
                    )
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

    def _build_worker_source(self, requests: tuple[SceneImageRequest, ...]) -> str:
        config = {
            "model": self.model,
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

            from hashlib import sha256
            import json
            import subprocess
            import sys
            from pathlib import Path
            import zipfile

            CONFIG = json.loads({config_json!r})

            try:
                import numpy as np
                import torch
                from diffusers import StableDiffusionPipeline
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "diffusers<1", "transformers<5", "accelerate<2", "safetensors", "numpy",
                ])
                import numpy as np
                import torch
                from diffusers import StableDiffusionPipeline

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            pipe = StableDiffusionPipeline.from_pretrained(
                CONFIG["model"],
                torch_dtype=torch.float16,
            )
            pipe.enable_attention_slicing()
            pipe = pipe.to("cuda")

            output_dir = Path("/kaggle/working/batch_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            reports = []

            for index, scene in enumerate(CONFIG["scenes"]):
                generator = torch.Generator(device="cuda").manual_seed(int(scene["seed"]))
                result = pipe(
                    prompt=scene["prompt"],
                    negative_prompt=scene["negative_prompt"] or None,
                    width=int(scene["width"]),
                    height=int(scene["height"]),
                    num_inference_steps=int(CONFIG["inference_steps"]),
                    guidance_scale=float(CONFIG["guidance_scale"]),
                    generator=generator,
                )
                image = result.images[0]
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
                }})

            archive_path = Path("/kaggle/working/images.zip")
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(output_dir.glob("*.png")):
                    archive.write(path, arcname=path.name)

            batch_report = {{
                "model": CONFIG["model"],
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
