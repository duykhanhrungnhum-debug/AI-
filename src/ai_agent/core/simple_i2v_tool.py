"""Bounded image-to-video backend used only inside AIKA's Video Tool.

One invocation submits one Kaggle GPU job, produces one MP4 and one structured
report, and never performs a hidden GPU retry. AIKA and MediaToolBroker never
see Kaggle/model orchestration details.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import textwrap
import time
from uuid import uuid4

from .kaggle_worker import KaggleGpuWorker


@dataclass(frozen=True)
class SimpleI2VArtifact:
    data: bytes
    mime_type: str
    provider: str
    model: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    elapsed_seconds: float
    evidence: tuple[str, ...]


@dataclass
class SimpleKaggleI2VTool:
    worker: KaggleGpuWorker
    kernel_slug: str
    model: str = "Lightricks/LTX-Video"
    inference_steps: int = 30
    guidance_scale: float = 3.0
    poll_interval: float = 5.0
    max_wait_seconds: float = 900.0
    min_video_bytes: int = 20_000
    provider: str = "ltx-image-to-video"

    def generate(
        self,
        prompt: str,
        image: bytes,
        *,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
    ) -> SimpleI2VArtifact:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("video prompt is required")
        if not image:
            raise ValueError("reference image is required")
        if len(image) > 12 * 1024 * 1024:
            raise ValueError("reference image must not exceed 12MB")
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("LTX video dimensions must be positive and divisible by 32")
        if num_frames <= 1 or (num_frames - 1) % 8:
            raise ValueError("LTX num_frames must follow the 8*k+1 temporal grid")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")

        run_token = uuid4().hex
        input_sha = sha256(image).hexdigest()
        source = self._worker_source(
            run_token=run_token,
            prompt=prompt,
            image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
        )
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=self.kernel_slug,
            source=source,
            machine_shape="NvidiaTeslaT4",
            enable_internet=True,
            enable_gpu=True,
            is_private=True,
        )
        canonical_slug = submission.slug
        report = self._wait_for_report(run_token, canonical_slug)
        video = self.worker.download_output_file(canonical_slug, "video_tool.mp4")

        if len(video) < self.min_video_bytes:
            raise RuntimeError(f"video artifact too small: {len(video)} bytes")
        if len(video) < 12 or video[4:8] != b"ftyp":
            raise RuntimeError("video artifact is not a recognized MP4")
        digest = sha256(video).hexdigest()
        if report.get("video_sha256") != digest:
            raise RuntimeError("video artifact hash mismatch")
        if report.get("input_image_sha256") != input_sha:
            raise RuntimeError("reference image hash mismatch")
        if report.get("model") != self.model:
            raise RuntimeError("video report model mismatch")
        if int(report.get("num_frames", 0) or 0) != num_frames:
            raise RuntimeError("video report frame count mismatch")
        if int(report.get("width", 0) or 0) != width or int(report.get("height", 0) or 0) != height:
            raise RuntimeError("video report dimensions mismatch")

        return SimpleI2VArtifact(
            data=video,
            mime_type="video/mp4",
            provider=self.provider,
            model=self.model,
            duration_seconds=float(report.get("duration_seconds", 0) or 0),
            width=int(report.get("width", 0) or 0),
            height=int(report.get("height", 0) or 0),
            fps=float(report.get("fps", 0) or 0),
            elapsed_seconds=float(report.get("generation_seconds", 0) or 0),
            evidence=(
                f"kaggle_kernel:{submission.ref}",
                f"run_token:{run_token}",
                f"input_image_sha256:{input_sha}",
                f"video_sha256:{digest}",
                f"gpu:{report.get('gpu_name', '')}",
                f"dimensions:{width}x{height}",
                f"frames:{num_frames}",
                f"fps:{report.get('fps')}",
                f"steps:{report.get('inference_steps')}",
                f"first_frame_similarity:{report.get('first_frame_similarity')}",
                f"last_frame_similarity:{report.get('last_frame_similarity')}",
                f"motion_delta:{report.get('motion_delta')}",
                f"generation_seconds:{report.get('generation_seconds')}",
            ),
        )

    def _wait_for_report(self, run_token: str, kernel_slug: str) -> dict:
        deadline = time.monotonic() + self.max_wait_seconds
        latest_token = ""
        while time.monotonic() < deadline:
            try:
                raw = self.worker.download_output_file(kernel_slug, "video_tool_report.json")
                parsed = json.loads(raw.decode("utf-8"))
            except (FileNotFoundError, ValueError, json.JSONDecodeError, RuntimeError):
                parsed = None

            if isinstance(parsed, dict):
                latest_token = str(parsed.get("run_token", ""))
                if latest_token == run_token:
                    if parsed.get("status") != "success":
                        stage = str(parsed.get("stage", "unknown"))
                        error = str(parsed.get("error", "video worker failed"))
                        raise RuntimeError(
                            f"AIKA I2V worker failed run_token={run_token} stage={stage}: {error}"
                        )
                    return parsed

            try:
                status = self.worker.status(kernel_slug)
                if status.terminal and not status.successful:
                    detail = status.failure_message or status.status
                    raise RuntimeError(
                        f"AIKA I2V Kaggle job failed run_token={run_token}: {detail}"
                    )
            except RuntimeError:
                raise
            except Exception:
                pass

            if self.poll_interval:
                time.sleep(self.poll_interval)

        stale = f" latest_artifact_token={latest_token}" if latest_token else ""
        raise TimeoutError(
            f"AIKA I2V tool timeout run_token={run_token} slug={kernel_slug}{stale}"
        )

    def _worker_source(
        self,
        *,
        run_token: str,
        prompt: str,
        image: bytes,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
    ) -> str:
        import base64

        config = json.dumps(
            {
                "run_token": run_token,
                "model": self.model,
                "prompt": prompt,
                "image_b64": base64.b64encode(image).decode("ascii"),
                "input_image_sha256": sha256(image).hexdigest(),
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "seed": seed,
                "inference_steps": self.inference_steps,
                "guidance_scale": self.guidance_scale,
            },
            ensure_ascii=False,
        )
        template = r'''
from __future__ import annotations
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CONFIG = json.loads(__CONFIG__)
REPORT_PATH = Path("/kaggle/working/video_tool_report.json")
TOTAL_STARTED = time.perf_counter()
STATE = {"stage": "bootstrap"}


def write_report(payload):
    payload = dict(payload)
    payload.setdefault("run_token", CONFIG["run_token"])
    payload.setdefault("model", CONFIG["model"])
    payload.setdefault("inference_steps", int(CONFIG["inference_steps"]))
    payload.setdefault("total_seconds", round(time.perf_counter() - TOTAL_STARTED, 3))
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def frame_array(frame):
    import numpy as np
    if hasattr(frame, "convert"):
        frame = frame.convert("RGB")
        return np.asarray(frame, dtype=np.float32)
    arr = np.asarray(frame, dtype=np.float32)
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    return arr


try:
    STATE["stage"] = "dependencies"
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "diffusers==0.40.0", "transformers==5.18.0", "accelerate>=1,<2",
        "safetensors", "sentencepiece", "Pillow", "imageio", "imageio-ffmpeg",
    ])

    import numpy as np
    import torch
    from PIL import Image
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")
    gpu_name = torch.cuda.get_device_name(0)

    STATE["stage"] = "reference_image"
    image_bytes = base64.b64decode(CONFIG["image_b64"], validate=True)
    if sha256(image_bytes).hexdigest() != CONFIG["input_image_sha256"]:
        raise RuntimeError("reference image hash mismatch before inference")
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((int(CONFIG["width"]), int(CONFIG["height"])), Image.Resampling.LANCZOS)
    input_pixels = np.asarray(image, dtype=np.float32)

    STATE["stage"] = "model_load"
    pipe = LTXImageToVideoPipeline.from_pretrained(
        CONFIG["model"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    negative_prompt = (
        "worst quality, low quality, inconsistent motion, jittery, blurry, distorted, "
        "deformed, duplicated subjects, fused subjects, morphing identity, text, watermark"
    )
    generator = torch.Generator(device="cpu").manual_seed(int(CONFIG["seed"]))

    STATE["stage"] = "inference"
    started = time.perf_counter()
    with torch.inference_mode():
        frames = pipe(
            image=image,
            prompt=CONFIG["prompt"],
            negative_prompt=negative_prompt,
            width=int(CONFIG["width"]),
            height=int(CONFIG["height"]),
            num_frames=int(CONFIG["num_frames"]),
            frame_rate=int(CONFIG["fps"]),
            num_inference_steps=int(CONFIG["inference_steps"]),
            guidance_scale=float(CONFIG["guidance_scale"]),
            generator=generator,
        ).frames[0]
    generation_seconds = time.perf_counter() - started
    if len(frames) != int(CONFIG["num_frames"]):
        raise RuntimeError(f"unexpected generated frame count: {len(frames)}")

    STATE["stage"] = "technical_qa"
    first = frame_array(frames[0])
    last = frame_array(frames[-1])
    if first.shape != input_pixels.shape:
        raise RuntimeError(f"first frame shape mismatch: {first.shape} != {input_pixels.shape}")
    first_similarity = 1.0 - float(np.mean(np.abs(first - input_pixels)) / 255.0)
    last_similarity = 1.0 - float(np.mean(np.abs(last - input_pixels)) / 255.0)
    motion_delta = float(np.mean(np.abs(last - first)))
    if not np.isfinite(first_similarity) or not np.isfinite(last_similarity) or not np.isfinite(motion_delta):
        raise RuntimeError("non-finite video QA metric")
    if motion_delta < 0.25:
        raise RuntimeError(f"generated video is effectively static: motion_delta={motion_delta:.4f}")

    STATE["stage"] = "export"
    output = Path("/kaggle/working/video_tool.mp4")
    export_to_video(frames, str(output), fps=int(CONFIG["fps"]))
    data = output.read_bytes()
    if len(data) < 20_000 or len(data) < 12 or data[4:8] != b"ftyp":
        raise RuntimeError("generated video artifact is invalid")

    report = {
        "run_token": CONFIG["run_token"],
        "status": "success",
        "stage": "done",
        "model": CONFIG["model"],
        "gpu_name": gpu_name,
        "precision": "fp16_model_cpu_offload",
        "input_image_sha256": CONFIG["input_image_sha256"],
        "video_sha256": sha256(data).hexdigest(),
        "width": int(CONFIG["width"]),
        "height": int(CONFIG["height"]),
        "fps": float(CONFIG["fps"]),
        "num_frames": int(CONFIG["num_frames"]),
        "duration_seconds": float(CONFIG["num_frames"]) / float(CONFIG["fps"]),
        "inference_steps": int(CONFIG["inference_steps"]),
        "seed": int(CONFIG["seed"]),
        "generation_seconds": round(generation_seconds, 3),
        "first_frame_similarity": round(first_similarity, 6),
        "last_frame_similarity": round(last_similarity, 6),
        "motion_delta": round(motion_delta, 6),
        "quality_verified": False,
        "qa_required": "visual_semantic_identity_motion",
    }
    write_report(report)
    print("AIKA_I2V_TECHNICAL_OK_PENDING_VISUAL_QA")
    print(json.dumps(report, ensure_ascii=False))
except Exception as exc:
    failed = {
        "run_token": CONFIG["run_token"],
        "status": "failed",
        "stage": STATE.get("stage", "unknown"),
        "model": CONFIG["model"],
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-8000:],
        "input_image_sha256": CONFIG.get("input_image_sha256", ""),
    }
    try:
        if "torch" in globals() and torch.cuda.is_available():
            failed["gpu_name"] = torch.cuda.get_device_name(0)
            failed["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            failed["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception:
        pass
    write_report(failed)
    print("AIKA_I2V_FAIL")
    print(json.dumps(failed, ensure_ascii=False))
    raise
'''
        return textwrap.dedent(template.replace("__CONFIG__", repr(config)))
