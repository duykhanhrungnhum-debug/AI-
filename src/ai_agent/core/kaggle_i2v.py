"""On-demand image-to-video generation for verified scene keyframes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import base64
import json
import textwrap
import time
import zipfile

from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker


@dataclass(frozen=True)
class SceneImageToVideoRequest:
    scene_id: str
    image: bytes
    width: int = 512
    height: int = 288
    num_frames: int = 14
    fps: int = 7
    seed: int = 0
    motion_bucket_id: int = 96
    noise_aug_strength: float = 0.02

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id is required")
        if not self.image:
            raise ValueError("image is required")
        if len(self.image) > 8 * 1024 * 1024:
            raise ValueError("image must not exceed 8MB")
        if self.width <= 0 or self.height <= 0 or self.width % 8 or self.height % 8:
            raise ValueError("video dimensions must be positive and divisible by 8")
        if self.num_frames <= 1:
            raise ValueError("num_frames must be greater than 1")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.motion_bucket_id <= 0:
            raise ValueError("motion_bucket_id must be positive")
        if not 0 <= self.noise_aug_strength <= 0.20:
            raise ValueError("noise_aug_strength must be between 0 and 0.20")


@dataclass(frozen=True)
class SceneImageToVideoArtifact:
    data: bytes
    mime_type: str
    provider: str
    model: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class SceneImageToVideoResult:
    scene_id: str
    artifact: SceneImageToVideoArtifact
    verified: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class BatchImageToVideoResult:
    scenes: tuple[SceneImageToVideoResult, ...]
    rounds: int

    @property
    def verified(self) -> bool:
        return bool(self.scenes) and all(scene.verified for scene in self.scenes)

    @property
    def failed_scene_ids(self) -> tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes if not scene.verified)


@dataclass
class KaggleBatchImageToVideoProvider:
    """Animate verified keyframes with Stable Video Diffusion on a Kaggle GPU."""

    worker: KaggleGpuWorker
    model: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    kernel_slug: str = "ai-agent-reference-i2v"
    poll_interval: float = 20.0
    max_poll_attempts: int = 180
    min_video_bytes: int = 20_000
    identity_threshold: float = 0.55
    min_motion_delta: float = 1.0
    provider: str = "kaggle-gpu-image-to-video"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if self.min_video_bytes <= 0:
            raise ValueError("min_video_bytes must be positive")
        if not 0 < self.identity_threshold <= 1:
            raise ValueError("identity_threshold must be in (0, 1]")
        if self.min_motion_delta < 0:
            raise ValueError("min_motion_delta must be non-negative")

    def generate_batch(
        self,
        requests: tuple[SceneImageToVideoRequest, ...] | list[SceneImageToVideoRequest],
    ) -> BatchImageToVideoResult:
        assert_core_invariants()
        requests = tuple(requests)
        self._validate_requests(requests)

        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=self.kernel_slug.replace("-", " ").title(),
            source=self._build_worker_source(requests),
            enable_internet=True,
            enable_gpu=True,
            is_private=True,
        )
        self._wait()

        archive = self.worker.download_output_file(self.kernel_slug, "i2v_videos.zip")
        report_bytes = self.worker.download_output_file(self.kernel_slug, "i2v_batch_report.json")
        report = self._parse_report(report_bytes)
        if report.get("model") != self.model:
            raise ValueError("I2V report model does not match configured model")
        gpu_name = str(report.get("gpu_name", "")).strip()
        if not gpu_name:
            raise ValueError("I2V report does not contain GPU evidence")

        entries = report.get("scenes")
        if not isinstance(entries, list):
            raise ValueError("I2V report does not contain scenes")
        by_id = {str(item.get("scene_id")): item for item in entries if isinstance(item, dict)}
        results: list[SceneImageToVideoResult] = []
        hashes: dict[str, list[int]] = {}

        with zipfile.ZipFile(BytesIO(archive), "r") as zipped:
            for index, request in enumerate(requests):
                entry = by_id.get(request.scene_id)
                if entry is None:
                    raise ValueError(f"I2V report missing scene: {request.scene_id}")
                filename = str(entry.get("filename", "")).strip()
                if not filename:
                    raise ValueError(f"I2V report missing filename: {request.scene_id}")
                try:
                    video = zipped.read(filename)
                except KeyError as exc:
                    raise ValueError(f"I2V archive missing file: {filename}") from exc

                digest = sha256(video).hexdigest()
                issues = self._verify_scene(request, entry, video, digest)
                hashes.setdefault(digest, []).append(index)
                artifact = SceneImageToVideoArtifact(
                    data=video,
                    mime_type="video/mp4",
                    provider=self.provider,
                    model=self.model,
                    duration_seconds=float(entry.get("duration_seconds", 0) or 0),
                    width=int(entry.get("width", 0) or 0),
                    height=int(entry.get("height", 0) or 0),
                    fps=float(entry.get("fps", 0) or 0),
                    evidence=(
                        f"kaggle_kernel:{submission.ref}",
                        f"scene_id:{request.scene_id}",
                        f"video_sha256:{digest}",
                        f"input_image_sha256:{sha256(request.image).hexdigest()}",
                        f"dimensions:{entry.get('width')}x{entry.get('height')}",
                        f"duration_seconds:{float(entry.get('duration_seconds', 0) or 0):.3f}",
                        f"fps:{float(entry.get('fps', 0) or 0):.3f}",
                        f"seed:{entry.get('seed')}",
                        f"motion_bucket_id:{entry.get('motion_bucket_id')}",
                        f"noise_aug_strength:{float(entry.get('noise_aug_strength', 0) or 0):.4f}",
                        f"first_frame_similarity:{float(entry.get('first_frame_similarity', 0) or 0):.6f}",
                        f"last_frame_similarity:{float(entry.get('last_frame_similarity', 0) or 0):.6f}",
                        f"motion_delta:{float(entry.get('motion_delta', 0) or 0):.6f}",
                        f"gpu:{gpu_name}",
                        f"model:{self.model}",
                    ),
                )
                results.append(SceneImageToVideoResult(
                    scene_id=request.scene_id,
                    artifact=artifact,
                    verified=not issues,
                    issues=tuple(issues),
                ))

        duplicates = {
            idx for indexes in hashes.values() if len(indexes) > 1 for idx in indexes
        }
        if duplicates:
            updated: list[SceneImageToVideoResult] = []
            for index, result in enumerate(results):
                if index in duplicates:
                    issues = tuple(dict.fromkeys((*result.issues, "duplicate video content")))
                    updated.append(replace(result, verified=False, issues=issues))
                else:
                    updated.append(result)
            results = updated
        return BatchImageToVideoResult(tuple(results), rounds=1)

    def generate_with_retries(
        self,
        requests: tuple[SceneImageToVideoRequest, ...] | list[SceneImageToVideoRequest],
        *,
        max_rounds: int = 2,
    ) -> BatchImageToVideoResult:
        assert_core_invariants()
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        original = tuple(requests)
        self._validate_requests(original)

        latest: dict[str, SceneImageToVideoResult] = {}
        pending = original
        rounds = 0
        seen_hashes: dict[str, set[str]] = {r.scene_id: set() for r in original}

        while pending and rounds < max_rounds:
            batch = self.generate_batch(pending)
            rounds += 1
            next_pending: list[SceneImageToVideoRequest] = []
            request_by_id = {r.scene_id: r for r in pending}
            for result in batch.scenes:
                digest = self._evidence_value(result.artifact.evidence, "video_sha256:")
                issues = list(result.issues)
                if digest in seen_hashes[result.scene_id] and not result.verified:
                    issues.append("retry loop detected: repeated video")
                seen_hashes[result.scene_id].add(digest)
                result = replace(
                    result,
                    verified=(result.verified and not issues),
                    issues=tuple(dict.fromkeys(issues)),
                )
                latest[result.scene_id] = result

                if not result.verified and rounds < max_rounds:
                    request = request_by_id[result.scene_id]
                    identity_drift = any("identity similarity" in issue for issue in issues)
                    low_motion = any("insufficient motion" in issue for issue in issues)
                    noise = request.noise_aug_strength
                    bucket = request.motion_bucket_id
                    if identity_drift:
                        noise = max(0.0, noise - 0.01)
                        bucket = max(48, bucket - 24)
                    elif low_motion:
                        bucket = min(180, bucket + 24)
                    next_pending.append(replace(
                        request,
                        seed=request.seed + rounds,
                        noise_aug_strength=noise,
                        motion_bucket_id=bucket,
                    ))
            pending = tuple(next_pending)

        return BatchImageToVideoResult(
            tuple(latest[r.scene_id] for r in original),
            rounds=rounds,
        )

    def _verify_scene(self, request, entry, video: bytes, digest: str) -> list[str]:
        issues: list[str] = []
        if len(video) < self.min_video_bytes:
            issues.append(f"video file too small: {len(video)} bytes")
        if len(video) < 12 or video[4:8] != b"ftyp":
            issues.append("output is not a recognized MP4")
        if entry.get("video_sha256") != digest:
            issues.append("video hash mismatch")
        if entry.get("input_image_sha256") != sha256(request.image).hexdigest():
            issues.append("input image hash mismatch")
        if int(entry.get("width", 0) or 0) != request.width:
            issues.append("video width mismatch")
        if int(entry.get("height", 0) or 0) != request.height:
            issues.append("video height mismatch")
        first_similarity = float(entry.get("first_frame_similarity", 0) or 0)
        last_similarity = float(entry.get("last_frame_similarity", 0) or 0)
        if min(first_similarity, last_similarity) < self.identity_threshold:
            issues.append(
                "identity similarity below threshold: "
                f"first={first_similarity:.3f}, last={last_similarity:.3f}, "
                f"threshold={self.identity_threshold:.3f}"
            )
        motion_delta = float(entry.get("motion_delta", 0) or 0)
        if motion_delta < self.min_motion_delta:
            issues.append(
                f"insufficient motion: delta={motion_delta:.3f}, "
                f"threshold={self.min_motion_delta:.3f}"
            )
        if float(entry.get("duration_seconds", 0) or 0) <= 0:
            issues.append("video duration is invalid")
        return issues

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
                    raise RuntimeError(f"Kaggle I2V batch failed: {detail}")
                return
            if self.poll_interval:
                time.sleep(self.poll_interval)
        raise TimeoutError("Timed out waiting for Kaggle I2V batch")

    @staticmethod
    def _validate_requests(requests) -> None:
        if not requests:
            raise ValueError("at least one I2V scene is required")
        ids = [r.scene_id for r in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("scene_id values must be unique")

    @staticmethod
    def _parse_report(data: bytes) -> dict:
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("I2V report is invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("I2V report must be an object")
        return parsed

    @staticmethod
    def _evidence_value(evidence: tuple[str, ...], prefix: str) -> str:
        for item in evidence:
            if item.startswith(prefix):
                return item[len(prefix):]
        return ""

    def _build_worker_source(self, requests) -> str:
        config = {
            "model": self.model,
            "scenes": [
                {
                    "scene_id": r.scene_id,
                    "image_b64": base64.b64encode(r.image).decode("ascii"),
                    "input_image_sha256": sha256(r.image).hexdigest(),
                    "width": r.width,
                    "height": r.height,
                    "num_frames": r.num_frames,
                    "fps": r.fps,
                    "seed": r.seed,
                    "motion_bucket_id": r.motion_bucket_id,
                    "noise_aug_strength": r.noise_aug_strength,
                }
                for r in requests
            ],
        }
        config_json = json.dumps(config, separators=(",", ":"))
        return textwrap.dedent(
            f"""
            from __future__ import annotations

            import base64
            import gc
            from hashlib import sha256
            from io import BytesIO
            import json
            from pathlib import Path
            import subprocess
            import sys
            import zipfile

            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q",
                "diffusers==0.35.2", "transformers>=4.46,<5", "accelerate>=1,<2",
                "safetensors>=0.4", "imageio>=2.34", "imageio-ffmpeg>=0.5",
            ], check=True)

            import numpy as np
            import torch
            import torch.nn.functional as F
            from PIL import Image
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import export_to_video
            from transformers import CLIPImageProcessor, CLIPVisionModel

            CONFIG = json.loads({json.dumps(config_json)})
            output_dir = Path("/kaggle/working/i2v")
            output_dir.mkdir(parents=True, exist_ok=True)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            if not gpu_name:
                raise RuntimeError("CUDA GPU is required")

            pipe = StableVideoDiffusionPipeline.from_pretrained(
                CONFIG["model"],
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe.enable_model_cpu_offload()
            pipe.unet.enable_forward_chunking()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").eval()

            reports = []
            for index, scene in enumerate(CONFIG["scenes"]):
                raw = base64.b64decode(scene["image_b64"])
                if sha256(raw).hexdigest() != scene["input_image_sha256"]:
                    raise RuntimeError("input image checksum mismatch")
                image = Image.open(BytesIO(raw)).convert("RGB")
                image = image.resize((int(scene["width"]), int(scene["height"])), Image.Resampling.LANCZOS)
                generator = torch.Generator(device="cpu").manual_seed(int(scene["seed"]))
                frames = pipe(
                    image,
                    num_frames=int(scene["num_frames"]),
                    decode_chunk_size=2,
                    generator=generator,
                    fps=int(scene["fps"]),
                    motion_bucket_id=int(scene["motion_bucket_id"]),
                    noise_aug_strength=float(scene["noise_aug_strength"]),
                ).frames[0]

                filename = f"scene_{{index:04d}}.mp4"
                path = output_dir / filename
                export_to_video(frames, str(path), fps=int(scene["fps"]))

                pixels = processor(images=[image, frames[0], frames[-1]], return_tensors="pt").pixel_values
                with torch.no_grad():
                    pooled = clip(pixel_values=pixels).pooler_output
                    pooled = F.normalize(pooled, dim=-1)
                first_similarity = float((pooled[0] * pooled[1]).sum().item())
                last_similarity = float((pooled[0] * pooled[2]).sum().item())
                motion_delta = float(np.abs(
                    np.asarray(frames[0], dtype=np.float32)
                    - np.asarray(frames[-1], dtype=np.float32)
                ).mean())

                data = path.read_bytes()
                reports.append({{
                    "scene_id": scene["scene_id"],
                    "filename": filename,
                    "video_sha256": sha256(data).hexdigest(),
                    "input_image_sha256": scene["input_image_sha256"],
                    "width": int(scene["width"]),
                    "height": int(scene["height"]),
                    "fps": float(scene["fps"]),
                    "duration_seconds": float(len(frames) / int(scene["fps"])),
                    "seed": int(scene["seed"]),
                    "motion_bucket_id": int(scene["motion_bucket_id"]),
                    "noise_aug_strength": float(scene["noise_aug_strength"]),
                    "first_frame_similarity": first_similarity,
                    "last_frame_similarity": last_similarity,
                    "motion_delta": motion_delta,
                }})

                del pooled, pixels, frames
                gc.collect()
                torch.cuda.empty_cache()

            del pipe, clip, processor
            gc.collect()
            torch.cuda.empty_cache()

            archive_path = Path("/kaggle/working/i2v_videos.zip")
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for path in sorted(output_dir.glob("*.mp4")):
                    archive.write(path, arcname=path.name)
            report = {{
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "scene_count": len(reports),
                "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
                "scenes": reports,
            }}
            Path("/kaggle/working/i2v_batch_report.json").write_text(
                json.dumps(report, indent=2) + "\\n", encoding="utf-8"
            )
            print("AI_AGENT_I2V_BATCH_OK")
            print(json.dumps(report, indent=2))
            """
        ).strip() + "\n"
