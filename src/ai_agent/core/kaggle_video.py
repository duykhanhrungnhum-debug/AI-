"""On-demand batch text-to-video generation on a self-controlled Kaggle GPU worker."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
import textwrap
import time
import zipfile

from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker


@dataclass(frozen=True)
class SceneVideoRequest:
    scene_id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 480
    num_frames: int = 17
    fps: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id is required")
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("video dimensions must be positive and divisible by 16")
        if self.num_frames <= 0 or (self.num_frames - 1) % 4:
            raise ValueError("num_frames must follow Wan's 4*k+1 rule")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class SceneVideoArtifact:
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
class SceneVideoResult:
    scene_id: str
    artifact: SceneVideoArtifact
    verified: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class BatchVideoResult:
    scenes: tuple[SceneVideoResult, ...]
    rounds: int

    @property
    def verified(self) -> bool:
        return bool(self.scenes) and all(scene.verified for scene in self.scenes)

    @property
    def failed_scene_ids(self) -> tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes if not scene.verified)


@dataclass
class KaggleBatchVideoProvider:
    """Generate short motion clips with an open-weight Wan model on Kaggle GPU."""

    worker: KaggleGpuWorker
    model: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    kernel_slug: str = "ai-agent-video-batch"
    poll_interval: float = 20.0
    max_poll_attempts: int = 180
    inference_steps: int = 20
    guidance_scale: float = 5.0
    min_video_bytes: int = 20_000
    provider: str = "kaggle-gpu-local-video-model"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if self.inference_steps <= 0 or self.guidance_scale <= 0:
            raise ValueError("inference_steps and guidance_scale must be positive")
        if self.min_video_bytes <= 0:
            raise ValueError("min_video_bytes must be positive")

    def generate_batch(
        self,
        requests: tuple[SceneVideoRequest, ...] | list[SceneVideoRequest],
    ) -> BatchVideoResult:
        assert_core_invariants()
        requests = tuple(requests)
        self._validate_requests(requests)

        source = self._build_worker_source(requests)
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=self.kernel_slug.replace("-", " ").title(),
            source=source,
            enable_internet=True,
            enable_gpu=True,
            is_private=True,
        )
        self._wait()

        archive = self.worker.download_output_file(self.kernel_slug, "videos.zip")
        report_bytes = self.worker.download_output_file(self.kernel_slug, "video_batch_report.json")
        report = self._parse_report(report_bytes)

        if report.get("model") != self.model:
            raise ValueError("video batch report model does not match configured model")
        gpu_name = str(report.get("gpu_name", "")).strip()
        if not gpu_name:
            raise ValueError("video batch report does not contain GPU evidence")
        entries = report.get("scenes")
        if not isinstance(entries, list):
            raise ValueError("video batch report does not contain scenes")

        by_id = {str(item.get("scene_id")): item for item in entries if isinstance(item, dict)}
        results: list[SceneVideoResult] = []
        hashes: dict[str, list[int]] = {}

        with zipfile.ZipFile(BytesIO(archive), "r") as zipped:
            for index, request in enumerate(requests):
                entry = by_id.get(request.scene_id)
                if entry is None:
                    raise ValueError(f"video batch report missing scene: {request.scene_id}")
                filename = str(entry.get("filename", "")).strip()
                if not filename:
                    raise ValueError(f"video batch report missing filename: {request.scene_id}")
                try:
                    video = zipped.read(filename)
                except KeyError as exc:
                    raise ValueError(f"video batch archive missing file: {filename}") from exc

                digest = sha256(video).hexdigest()
                issues = self._verify_scene(request, entry, video, digest)
                hashes.setdefault(digest, []).append(index)

                artifact = SceneVideoArtifact(
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
                        f"dimensions:{entry.get('width')}x{entry.get('height')}",
                        f"duration_seconds:{float(entry.get('duration_seconds', 0) or 0):.3f}",
                        f"fps:{float(entry.get('fps', 0) or 0):.3f}",
                        f"seed:{entry.get('seed')}",
                        f"gpu:{gpu_name}",
                        f"model:{self.model}",
                    ),
                )
                results.append(
                    SceneVideoResult(
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
            updated: list[SceneVideoResult] = []
            for index, result in enumerate(results):
                if index in duplicate_indexes:
                    issues = tuple(dict.fromkeys((*result.issues, "duplicate video content")))
                    updated.append(replace(result, verified=False, issues=issues))
                else:
                    updated.append(result)
            results = updated

        return BatchVideoResult(tuple(results), rounds=1)

    def generate_with_retries(
        self,
        requests: tuple[SceneVideoRequest, ...] | list[SceneVideoRequest],
        *,
        max_rounds: int = 2,
    ) -> BatchVideoResult:
        assert_core_invariants()
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        original = tuple(requests)
        self._validate_requests(original)

        latest: dict[str, SceneVideoResult] = {}
        pending = original
        rounds = 0
        seen_hashes: dict[str, set[str]] = {request.scene_id: set() for request in original}

        while pending and rounds < max_rounds:
            batch = self.generate_batch(pending)
            rounds += 1
            next_pending: list[SceneVideoRequest] = []
            request_by_id = {request.scene_id: request for request in pending}

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
                    next_pending.append(replace(request, seed=request.seed + rounds))

            pending = tuple(next_pending)

        return BatchVideoResult(
            tuple(latest[request.scene_id] for request in original),
            rounds=rounds,
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
                    raise RuntimeError(f"Kaggle video batch failed: {detail[:4000]}")
                return
            if self.poll_interval:
                time.sleep(self.poll_interval)
        raise TimeoutError("Timed out waiting for Kaggle video batch")

    @staticmethod
    def _validate_requests(requests: tuple[SceneVideoRequest, ...]) -> None:
        if not requests:
            raise ValueError("at least one video scene is required")
        ids = [request.scene_id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("scene_id values must be unique")

    def _verify_scene(
        self,
        request: SceneVideoRequest,
        entry: dict,
        video: bytes,
        digest: str,
    ) -> list[str]:
        issues: list[str] = []
        if len(video) < self.min_video_bytes:
            issues.append(f"video file too small: {len(video)} bytes")
        if len(video) < 12 or video[4:8] != b"ftyp":
            issues.append("output is not a recognized MP4")
        if entry.get("video_sha256") != digest:
            issues.append("video hash mismatch")
        if int(entry.get("width", 0) or 0) != request.width:
            issues.append("video width mismatch")
        if int(entry.get("height", 0) or 0) != request.height:
            issues.append("video height mismatch")
        try:
            fps = float(entry.get("fps", 0) or 0)
            duration = float(entry.get("duration_seconds", 0) or 0)
        except (TypeError, ValueError):
            fps = 0.0
            duration = 0.0
        if fps <= 0:
            issues.append("video fps is invalid")
        if duration <= 0:
            issues.append("video duration is invalid")
        expected = request.num_frames / request.fps
        if duration > 0 and abs(duration - expected) > max(1.0, expected * 0.5):
            issues.append(
                f"video duration mismatch: expected about {expected:.3f}s, got {duration:.3f}s"
            )
        return issues

    @staticmethod
    def _parse_report(data: bytes) -> dict:
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("video batch report is invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("video batch report must be an object")
        return parsed

    @staticmethod
    def _evidence_value(evidence: tuple[str, ...], prefix: str) -> str:
        for item in evidence:
            if item.startswith(prefix):
                return item[len(prefix):]
        return ""

    def _build_worker_source(self, requests: tuple[SceneVideoRequest, ...]) -> str:
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
                    "num_frames": request.num_frames,
                    "fps": request.fps,
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
                import torch
                from diffusers import AutoencoderKLWan, WanPipeline
                from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
                from diffusers.utils import export_to_video
                import imageio_ffmpeg
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "diffusers>=0.36,<1", "transformers>=4.46,<5",
                    "accelerate>=1,<2", "safetensors", "sentencepiece",
                    "ftfy", "imageio", "imageio-ffmpeg",
                ])
                import torch
                from diffusers import AutoencoderKLWan, WanPipeline
                from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
                from diffusers.utils import export_to_video
                import imageio_ffmpeg

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            model_id = CONFIG["model"]
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
            )
            pipe = WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config,
                flow_shift=3.0,
            )
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()

            output_dir = Path("/kaggle/working/batch_videos")
            output_dir.mkdir(parents=True, exist_ok=True)
            reports = []

            for index, scene in enumerate(CONFIG["scenes"]):
                generator = torch.Generator(device="cuda").manual_seed(int(scene["seed"]))
                frames = pipe(
                    prompt=scene["prompt"],
                    negative_prompt=scene["negative_prompt"] or None,
                    height=int(scene["height"]),
                    width=int(scene["width"]),
                    num_frames=int(scene["num_frames"]),
                    num_inference_steps=int(CONFIG["inference_steps"]),
                    guidance_scale=float(CONFIG["guidance_scale"]),
                    generator=generator,
                ).frames[0]

                filename = f"scene_{{index:04d}}.mp4"
                path = output_dir / filename
                export_to_video(frames, str(path), fps=int(scene["fps"]))

                data = path.read_bytes()
                reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
                metadata = next(reader)
                reader.close()
                size = metadata.get("size") or (0, 0)
                actual_fps = float(metadata.get("fps") or scene["fps"])
                duration = float(metadata.get("duration") or (scene["num_frames"] / scene["fps"]))

                reports.append({{
                    "scene_id": scene["scene_id"],
                    "filename": filename,
                    "video_sha256": sha256(data).hexdigest(),
                    "prompt_sha256": sha256(scene["prompt"].encode("utf-8")).hexdigest(),
                    "width": int(size[0]),
                    "height": int(size[1]),
                    "fps": actual_fps,
                    "duration_seconds": duration,
                    "seed": int(scene["seed"]),
                    "num_frames": int(scene["num_frames"]),
                }})

                del frames
                torch.cuda.empty_cache()

            archive_path = Path("/kaggle/working/videos.zip")
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
            Path("/kaggle/working/video_batch_report.json").write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
            print("AI_AGENT_VIDEO_BATCH_OK")
            print(json.dumps(report, indent=2))
            """
        ).strip() + "\\n"
