from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import textwrap
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]

IMAGE_MODEL = "black-forest-labs/FLUX.2-klein-4B"
VIDEO_MODEL = "Wan-AI/Wan2.1-T2V-1.3B"
WAN_REPO_COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
REALESRGAN_REPO_COMMIT = "a4abfb2979a7bbff3f69f58f58ae324608821e27"


def _load_worker_class():
    path = ROOT / "local-ai" / "kaggle-worker.py"
    spec = importlib.util.spec_from_file_location("money_agent_kaggle_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Kaggle worker from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KaggleWorker


def _wait_report(worker, slug: str, filename: str, run_token: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status = ""
    while time.monotonic() < deadline:
        try:
            raw = worker.download(slug, filename)
            report = json.loads(raw.decode("utf-8"))
            if isinstance(report, dict) and report.get("run_token") == run_token:
                if report.get("status") == "failed":
                    raise RuntimeError(
                        f"{report.get('stage', 'unknown')}: "
                        f"{report.get('error', 'worker failed')}"
                    )
                if report.get("status") == "success":
                    return report
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        status = worker.status(slug)
        last_status = status.status
        if status.terminal and not status.successful:
            detail = status.failure_message or status.status
            try:
                logs = worker.logs(slug)
                if logs.strip():
                    detail = logs[-8000:]
            except Exception:
                pass
            raise RuntimeError(detail)
        time.sleep(worker.poll_interval)
    raise TimeoutError(f"Kaggle job timed out; last_status={last_status}")


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _text_source(config: dict[str, Any]) -> str:
    raw = json.dumps(config, ensure_ascii=False)
    template = r"""
from hashlib import sha256
import json, os, subprocess, sys, traceback
from pathlib import Path
CONFIG = json.loads(__CONFIG__)
REPORT = Path("/kaggle/working/manager_text_report.json")
def save(data):
    out = dict(data); out.setdefault("run_token", CONFIG["run_token"])
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
try:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
            "transformers<5", "accelerate<2", "safetensors", "sentencepiece"])
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model"], torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    rendered = tokenizer.apply_chat_template(
        CONFIG["messages"], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=int(CONFIG["max_new_tokens"]),
        do_sample=False,
        repetition_penalty=1.05,
    )
    new_tokens = generated[:, inputs.input_ids.shape[1]:]
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    if not text:
        raise RuntimeError("model returned empty text")
    save({
        "status":"success",
        "model":CONFIG["model"],
        "gpu_name":torch.cuda.get_device_name(0),
        "text":text,
        "sha256":sha256(text.encode("utf-8")).hexdigest(),
    })
except Exception as exc:
    save({
        "status":"failed",
        "stage":"text",
        "error":f"{type(exc).__name__}: {exc}",
        "traceback":traceback.format_exc()[-8000:],
    })
    raise
"""
    return textwrap.dedent(template).replace("__CONFIG__", repr(raw)).strip() + "\n"


def _image_source(config: dict[str, Any]) -> str:
    raw = json.dumps(config, ensure_ascii=False)
    template = r"""
from hashlib import sha256
import json, os, subprocess, sys, time, traceback
from pathlib import Path
CONFIG = json.loads(__CONFIG__)
REPORT = Path("/kaggle/working/manager_image_report.json")
OUTPUT = Path("/kaggle/working/manager_image.png")
def save(data):
    out = dict(data); out.setdefault("run_token", CONFIG["run_token"])
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
try:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "diffusers>=0.36", "transformers>=4.57,<5", "accelerate>=1,<2",
        "safetensors", "sentencepiece", "Pillow<13"
    ])
    import torch
    from diffusers import DiffusionPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    started = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        CONFIG["model"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    generator = torch.Generator(device="cpu").manual_seed(int(CONFIG["seed"]))
    image = pipe(
        prompt=CONFIG["prompt"],
        width=int(CONFIG["width"]),
        height=int(CONFIG["height"]),
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=generator,
    ).images[0]
    image.save(OUTPUT, "PNG")
    elapsed = time.perf_counter() - started
    raw_bytes = OUTPUT.read_bytes()
    save({
        "status":"success",
        "stage":"flux2_klein_4b",
        "model":CONFIG["model"],
        "gpu_name":torch.cuda.get_device_name(0),
        "seconds":round(elapsed, 3),
        "sha256":sha256(raw_bytes).hexdigest(),
    })
except Exception as exc:
    save({
        "status":"failed",
        "stage":"flux2_klein_4b",
        "error":f"{type(exc).__name__}: {exc}",
        "traceback":traceback.format_exc()[-12000:],
    })
    raise
"""
    return textwrap.dedent(template).replace("__CONFIG__", repr(raw)).strip() + "\n"


def _video_source(config: dict[str, Any]) -> str:
    raw = json.dumps(config, ensure_ascii=False)
    template = r"""
from hashlib import sha256
import gc, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

CONFIG = json.loads(__CONFIG__)
WORK = Path("/kaggle/working")
REPORT = WORK / "manager_video_report.json"
WAN_RAW = WORK / "manager_video_wan.mp4"
OUTPUT = WORK / "manager_video.mp4"
WAN_DIR = WORK / "Wan2.1"
WAN_MODEL_DIR = WORK / "Wan2.1-T2V-1.3B"
ESR_DIR = WORK / "Real-ESRGAN"
ESR_OUT = WORK / "realesrgan-output"

def save(data):
    out = dict(data)
    out.setdefault("run_token", CONFIG["run_token"])
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def run(cmd, cwd=None):
    subprocess.check_call(cmd, cwd=cwd)

try:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    report = {
        "status": "running",
        "stage": "setup",
        "video_model": CONFIG["video_model"],
        "upscaler": "RealESRGAN_x2plus",
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    }
    save(report)

    setup_started = time.perf_counter()
    run([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "huggingface_hub>=0.28,<1",
        "transformers>=4.49,<5",
        "accelerate>=1.1,<2",
        "diffusers>=0.31,<1",
        "tokenizers>=0.20.3,<1",
        "opencv-python-headless>=4.9",
        "imageio", "imageio-ffmpeg", "easydict", "ftfy", "dashscope",
        "ffmpeg-python", "numpy<2", "safetensors"
    ])

    if WAN_DIR.exists():
        shutil.rmtree(WAN_DIR)
    run(["git", "clone", "--quiet", "--filter=blob:none", "https://github.com/Wan-Video/Wan2.1.git", str(WAN_DIR)])
    run(["git", "-C", str(WAN_DIR), "checkout", "--quiet", CONFIG["wan_commit"]])

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=CONFIG["video_model"],
        local_dir=str(WAN_MODEL_DIR),
        local_dir_use_symlinks=False,
    )

    setup_seconds = time.perf_counter() - setup_started
    report.update({"stage": "wan_generate", "setup_seconds": round(setup_seconds, 3)})
    save(report)

    wan_started = time.perf_counter()
    run([
        sys.executable, str(WAN_DIR / "generate.py"),
        "--task", "t2v-1.3B",
        "--size", CONFIG["wan_size"],
        "--frame_num", str(CONFIG["num_frames"]),
        "--ckpt_dir", str(WAN_MODEL_DIR),
        "--offload_model", "True",
        "--t5_cpu",
        "--sample_steps", str(CONFIG["sample_steps"]),
        "--sample_shift", "8",
        "--sample_guide_scale", "6",
        "--base_seed", str(CONFIG["seed"]),
        "--save_file", str(WAN_RAW),
        "--prompt", CONFIG["prompt"],
    ], cwd=str(WAN_DIR))
    wan_seconds = time.perf_counter() - wan_started

    if not WAN_RAW.exists() or WAN_RAW.stat().st_size < 20000:
        raise RuntimeError("Wan2.1 produced no valid MP4")

    gc.collect()
    torch.cuda.empty_cache()

    report.update({
        "stage": "realesrgan_x2",
        "wan_seconds": round(wan_seconds, 3),
        "wan_bytes": WAN_RAW.stat().st_size,
    })
    save(report)

    if ESR_DIR.exists():
        shutil.rmtree(ESR_DIR)
    run(["git", "clone", "--quiet", "--filter=blob:none", "https://github.com/xinntao/Real-ESRGAN.git", str(ESR_DIR)])
    run(["git", "-C", str(ESR_DIR), "checkout", "--quiet", CONFIG["realesrgan_commit"]])
    run([sys.executable, "-m", "pip", "install", "--quiet", "basicsr>=1.4.2", "facexlib>=0.3.0", "gfpgan>=1.3.8"])
    run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(ESR_DIR)])

    ESR_OUT.mkdir(parents=True, exist_ok=True)
    upscale_started = time.perf_counter()
    run([
        sys.executable, str(ESR_DIR / "inference_realesrgan_video.py"),
        "-i", str(WAN_RAW),
        "-n", "RealESRGAN_x2plus",
        "-o", str(ESR_OUT),
        "-s", str(CONFIG["upscale_factor"]),
        "-t", "256",
        "--fps", str(CONFIG["fps"]),
        "--suffix", "x2",
    ], cwd=str(ESR_DIR))
    upscale_seconds = time.perf_counter() - upscale_started

    candidates = list(ESR_OUT.rglob("*.mp4"))
    if not candidates:
        raise FileNotFoundError("Real-ESRGAN produced no MP4")
    upscaled = max(candidates, key=lambda p: p.stat().st_mtime)

    finalize_started = time.perf_counter()
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(upscaled),
        "-vf", "crop=1080:1920:(iw-1080)/2:(ih-1920)/2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(OUTPUT),
    ])
    finalize_seconds = time.perf_counter() - finalize_started

    raw_bytes = OUTPUT.read_bytes()
    if len(raw_bytes) < 20000 or raw_bytes[4:8] != b"ftyp":
        raise RuntimeError("invalid final MP4 output")

    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(OUTPUT)
    ], text=True)
    meta = json.loads(probe)["streams"][0]
    if int(meta["width"]) != 1080 or int(meta["height"]) != 1920:
        raise RuntimeError(f"final video is not 1080x1920: {meta}")

    total_seconds = setup_seconds + wan_seconds + upscale_seconds + finalize_seconds
    save({
        "status": "success",
        "stage": "wan_1_3b_plus_realesrgan_x2",
        "video_model": CONFIG["video_model"],
        "upscaler": "RealESRGAN_x2plus",
        "gpu_name": torch.cuda.get_device_name(0),
        "setup_seconds": round(setup_seconds, 3),
        "wan_seconds": round(wan_seconds, 3),
        "upscale_seconds": round(upscale_seconds, 3),
        "finalize_seconds": round(finalize_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "frames": int(CONFIG["num_frames"]),
        "fps": int(CONFIG["fps"]),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": len(raw_bytes),
        "sha256": sha256(raw_bytes).hexdigest(),
    })
except Exception as exc:
    save({
        "status": "failed",
        "stage": "wan_1_3b_plus_realesrgan_x2",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-16000:],
    })
    raise
"""
    return textwrap.dedent(template).replace("__CONFIG__", repr(raw)).strip() + "\n"

class KaggleTools:
    def __init__(self) -> None:
        Worker = _load_worker_class()
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        if not username or not token:
            raise RuntimeError("KAGGLE_USERNAME and KAGGLE_API_TOKEN are required")
        self.worker = Worker(
            token,
            username,
            timeout=120,
            poll_interval=float(os.environ.get("KAGGLE_POLL_INTERVAL_SECONDS", "5")),
            submission_retry_attempts=1,
        )
        self.machine_shape = os.environ.get("KAGGLE_MACHINE_SHAPE", "NvidiaTeslaT4").strip() or None
        self.timeout = float(os.environ.get("KAGGLE_MANAGER_TIMEOUT_SECONDS", "3600"))
        self.text_model = os.environ.get("MONEY_TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    def think(self, system: str, user: str) -> str:
        token = uuid4().hex
        slug = f"manager-text-{token[:12]}"
        payload = {
            "run_token": token,
            "model": self.text_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_new_tokens": 1200,
        }
        self.worker.submit(
            slug=slug,
            title="Money Agent Manager Brain",
            source=_text_source(payload),
            machine_shape=self.machine_shape,
        )
        report = _wait_report(self.worker, slug, "manager_text_report.json", token, self.timeout)
        return str(report["text"]).strip()

    def generate_image(self, prompt: str, output_dir: str | Path = "artifacts") -> Path:
        token = uuid4().hex
        slug = f"flux2-image-{token[:12]}"
        config = {
            "run_token": token,
            "model": IMAGE_MODEL,
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "seed": _seed(prompt),
        }
        self.worker.submit(
            slug=slug,
            title="FLUX2 Klein 4B Image",
            source=_image_source(config),
            machine_shape=self.machine_shape,
        )
        report = _wait_report(self.worker, slug, "manager_image_report.json", token, self.timeout)
        raw = self.worker.download(slug, "manager_image.png")
        if (
            not raw.startswith(b"\x89PNG\r\n\x1a\n")
            or hashlib.sha256(raw).hexdigest() != report["sha256"]
        ):
            raise RuntimeError("FLUX.2 Klein image integrity check failed")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"image-{token[:12]}.png"
        path.write_bytes(raw)
        return path

    def generate_video(
        self,
        prompt: str,
        output_dir: str | Path = "artifacts",
        seconds: float = 4.0,
    ) -> Path:
        token = uuid4().hex
        slug = f"wan13b-video-{token[:12]}"
        fps = 16
        seconds = max(2.0, min(float(seconds), 5.0))
        requested = int(round(seconds * fps))
        frames = max(33, ((requested - 1 + 3) // 4) * 4 + 1)
        config = {
            "run_token": token,
            "video_model": VIDEO_MODEL,
            "wan_commit": WAN_REPO_COMMIT,
            "realesrgan_commit": REALESRGAN_REPO_COMMIT,
            "prompt": prompt,
            "wan_size": "480*832",
            "num_frames": frames,
            "fps": fps,
            "sample_steps": 30,
            "upscale_factor": 1920 / 832,
            "seed": _seed(prompt + "|video"),
        }
        self.worker.submit(
            slug=slug,
            title="Wan2.1 1.3B + Real-ESRGAN x2 Video",
            source=_video_source(config),
            machine_shape=self.machine_shape,
        )
        report = _wait_report(self.worker, slug, "manager_video_report.json", token, self.timeout)
        raw = self.worker.download(slug, "manager_video.mp4")
        if (
            len(raw) < 20000
            or raw[4:8] != b"ftyp"
            or hashlib.sha256(raw).hexdigest() != report["sha256"]
            or int(report.get("width", 0)) != 1080
            or int(report.get("height", 0)) != 1920
        ):
            raise RuntimeError("Wan + Real-ESRGAN 1080p video integrity check failed")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"video-{token[:12]}.mp4"
        path.write_bytes(raw)
        return path
