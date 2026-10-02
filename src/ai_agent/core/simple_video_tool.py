"""Minimal one-shot video provider for AIKA.

One request creates one short MP4. The provider owns only execution,
verification, and a bounded result wait. It does not plan scenes, narrate,
or silently retry GPU work.
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
class SimpleVideoArtifact:
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
class SimpleKaggleVideoTool:
    worker: KaggleGpuWorker
    kernel_slug: str = "ai-agent-video-tool"
    model: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    inference_steps: int = 40
    guidance_scale: float = 5.0
    poll_interval: float = 5.0
    max_wait_seconds: float = 900.0
    min_video_bytes: int = 20_000
    provider: str = "kaggle-video-tool"

    def generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
    ) -> SimpleVideoArtifact:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("video prompt is required")
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("video dimensions must be positive and divisible by 16")
        if num_frames <= 0 or (num_frames - 1) % 4:
            raise ValueError("num_frames must follow Wan's 4*k+1 rule")
        if fps <= 0:
            raise ValueError("fps must be positive")

        run_token = uuid4().hex
        source = self._worker_source(
            run_token=run_token,
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
        )
        # Kaggle links a kernel title to its slug. The caller already provides a
        # unique per-job slug, so keep title and slug identical to avoid server
        # normalization creating a different kernel handle.
        kernel_title = self.kernel_slug
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=kernel_title,
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
        if report.get("model") != self.model:
            raise RuntimeError("video report model mismatch")

        return SimpleVideoArtifact(
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
                f"video_sha256:{digest}",
                f"gpu:{report.get('gpu_name', '')}",
                f"dimensions:{report.get('width')}x{report.get('height')}",
                f"fps:{report.get('fps')}",
                f"steps:{report.get('inference_steps')}",
                f"precision:{report.get('precision')}",
                f"generation_seconds:{report.get('generation_seconds')}",
            ),
        )

    def _wait_for_report(self, run_token: str, kernel_slug: str) -> dict:
        """Wait for this exact run token and surface worker failures immediately."""
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
                            f"AIKA video worker failed run_token={run_token} stage={stage}: {error}"
                        )
                    return parsed

            if self.poll_interval:
                time.sleep(self.poll_interval)

        detail = ""
        try:
            metadata = self.worker.output_metadata(kernel_slug)
            if isinstance(metadata, dict):
                detail = str(metadata.get("logNullable") or metadata.get("log") or "")[-3000:]
        except Exception:
            pass
        stale = f" latest_artifact_token={latest_token}" if latest_token else ""
        suffix = f" log_tail={detail}" if detail else ""
        raise TimeoutError(
            f"AIKA video tool timeout run_token={run_token} slug={kernel_slug}{stale}{suffix}"
        )

    def _worker_source(
        self,
        *,
        run_token: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
    ) -> str:
        config = json.dumps(
            {
                "run_token": run_token,
                "model": self.model,
                "prompt": prompt,
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
from hashlib import sha256
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

CONFIG = json.loads(__CONFIG__)
REPORT_PATH = Path("/kaggle/working/video_tool_report.json")
STATE = {"stage": "bootstrap"}
TOTAL_STARTED = time.perf_counter()


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


try:
    STATE["stage"] = "dependencies"
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "diffusers==0.40.0", "transformers==5.18.0",
        "accelerate>=1,<2", "safetensors", "sentencepiece", "ftfy",
        "imageio", "imageio-ffmpeg",
    ])

    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    model_id = CONFIG["model"]
    gpu_name = torch.cuda.get_device_name(0)
    device = torch.device("cuda")

    STATE["stage"] = "model_load"
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=3.0,
    )
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    negative_prompt = (
        "overall gray, raw noise, worst quality, low quality, blurred details, "
        "static image, subtitles, text, watermark, deformed, disfigured, "
        "fused subjects, duplicated subjects"
    )

    STATE["stage"] = "prompt_encode"
    with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=CONFIG["prompt"],
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=float(CONFIG["guidance_scale"]) > 1.0,
            num_videos_per_prompt=1,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

    STATE["stage"] = "cuda_prepare"
    pipe.transformer.to(device=device, dtype=torch.float32)
    pipe.vae.to(device=device, dtype=torch.float32)
    prompt_embeds = prompt_embeds.to(device=device, dtype=torch.float32)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=torch.float32)

    generator = torch.Generator(device=device).manual_seed(int(CONFIG["seed"]))
    STATE["stage"] = "inference"
    started = time.perf_counter()
    with torch.inference_mode():
        frames = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=int(CONFIG["height"]),
            width=int(CONFIG["width"]),
            num_frames=int(CONFIG["num_frames"]),
            num_inference_steps=int(CONFIG["inference_steps"]),
            guidance_scale=float(CONFIG["guidance_scale"]),
            generator=generator,
        ).frames[0]
    generation_seconds = time.perf_counter() - started

    STATE["stage"] = "export"
    output = Path("/kaggle/working/video_tool.mp4")
    export_to_video(frames, str(output), fps=int(CONFIG["fps"]))
    data = output.read_bytes()
    if len(data) < 20_000 or len(data) < 12 or data[4:8] != b"ftyp":
        raise RuntimeError("generated video artifact is invalid")

    samples = []
    for index in sorted({0, len(frames) // 2, len(frames) - 1}):
        samples.append(np.asarray(frames[index], dtype=np.float32))
    pixels = np.concatenate([sample.reshape(-1, 3) for sample in samples], axis=0)
    channel_spread = float(np.mean(np.max(pixels, axis=1) - np.min(pixels, axis=1)))
    luminance_std = float(np.std(np.mean(pixels, axis=1)))

    report = {
        "run_token": CONFIG["run_token"],
        "status": "success",
        "stage": "done",
        "model": CONFIG["model"],
        "gpu_name": gpu_name,
        "precision": "transformer_fp32_vae_fp32_text_bf16_cpu",
        "video_sha256": sha256(data).hexdigest(),
        "width": int(CONFIG["width"]),
        "height": int(CONFIG["height"]),
        "fps": float(CONFIG["fps"]),
        "duration_seconds": float(CONFIG["num_frames"]) / float(CONFIG["fps"]),
        "generation_seconds": round(generation_seconds, 3),
        "num_frames": int(CONFIG["num_frames"]),
        "inference_steps": int(CONFIG["inference_steps"]),
        "seed": int(CONFIG["seed"]),
        "channel_spread": round(channel_spread, 3),
        "luminance_std": round(luminance_std, 3),
    }
    write_report(report)
    print("AIKA_SIMPLE_VIDEO_TOOL_OK")
    print(json.dumps(report, ensure_ascii=False))
except Exception as exc:
    failed = {
        "run_token": CONFIG["run_token"],
        "status": "failed",
        "stage": STATE.get("stage", "unknown"),
        "model": CONFIG["model"],
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-8000:],
        "inference_steps": int(CONFIG["inference_steps"]),
    }
    try:
        if "torch" in globals() and torch.cuda.is_available():
            failed["gpu_name"] = torch.cuda.get_device_name(0)
            failed["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            failed["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception:
        pass
    write_report(failed)
    print("AIKA_SIMPLE_VIDEO_TOOL_FAIL")
    print(json.dumps(failed, ensure_ascii=False))
    raise
'''
        return textwrap.dedent(template.replace("__CONFIG__", repr(config)))
