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
VIDEO_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
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
import gc, json, os, subprocess, sys, time, traceback
from pathlib import Path
from urllib.request import urlretrieve

CONFIG = json.loads(__CONFIG__)
WORK = Path("/kaggle/working")
REPORT = WORK / "manager_video_report.json"
WAN_RAW = WORK / "manager_video_wan.mp4"
UPSCALED = WORK / "manager_video_x2.mp4"
OUTPUT = WORK / "manager_video.mp4"
FRAME_DIR = WORK / "wan_frames"
ESR_FRAME_DIR = WORK / "esr_frames"
ESR_WEIGHTS = WORK / "RealESRGAN_x2plus.pth"
ESR_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

def save(data):
    out = dict(data)
    out.setdefault("run_token", CONFIG["run_token"])
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed rc=%s\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (proc.returncode, proc.stdout[-12000:], proc.stderr[-12000:])
        )
    return proc

try:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    setup_started = time.perf_counter()
    # Keep Kaggle's binary stack intact. Do not upgrade NumPy/OpenCV/Pandas.
    run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "diffusers>=0.36,<0.37",
        "transformers>=4.57,<4.58",
        "accelerate>=1,<2",
        "safetensors", "sentencepiece", "ftfy", "imageio", "imageio-ffmpeg"
    ])

    # CPU preflight before model download/inference.
    preflight = run([
        sys.executable, "-c",
        (
            "import json, numpy, torch, transformers, diffusers; "
            "from PIL import Image; "
            "from diffusers import AutoencoderKLWan, WanPipeline; "
            "from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler; "
            "print(json.dumps({"
            "'numpy': numpy.__version__, 'torch': torch.__version__, "
            "'transformers': transformers.__version__, 'diffusers': diffusers.__version__"
            "}))"
        )
    ])
    env_versions = json.loads(preflight.stdout.strip().splitlines()[-1])

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    model_id = CONFIG["video_model"]
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.enable_model_cpu_offload()
    setup_seconds = time.perf_counter() - setup_started

    save({
        "status": "running",
        "stage": "wan_generate",
        "video_model": model_id,
        "upscaler": "RealESRGAN_x2plus_direct_pytorch",
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "setup_seconds": round(setup_seconds, 3),
        "environment": env_versions,
    })

    generator = torch.Generator(device="cpu").manual_seed(int(CONFIG["seed"]))
    wan_started = time.perf_counter()
    frames = pipe(
        prompt=CONFIG["prompt"],
        negative_prompt=(
            "blurry, jittery, distorted face, deformed anatomy, bad hands, "
            "low quality, text, subtitles, watermark"
        ),
        height=832,
        width=480,
        num_frames=int(CONFIG["num_frames"]),
        num_inference_steps=int(CONFIG["sample_steps"]),
        guidance_scale=5.0,
        generator=generator,
    ).frames[0]
    export_to_video(frames, str(WAN_RAW), fps=int(CONFIG["fps"]))
    wan_seconds = time.perf_counter() - wan_started

    if not WAN_RAW.exists() or WAN_RAW.stat().st_size < 20000:
        raise RuntimeError("Wan2.1 produced no valid MP4")

    del pipe, vae, frames
    gc.collect()
    torch.cuda.empty_cache()

    save({
        "status": "running",
        "stage": "realesrgan_x2_direct",
        "video_model": model_id,
        "upscaler": "RealESRGAN_x2plus_direct_pytorch",
        "gpu_name": torch.cuda.get_device_name(0),
        "setup_seconds": round(setup_seconds, 3),
        "wan_seconds": round(wan_seconds, 3),
        "wan_bytes": WAN_RAW.stat().st_size,
        "environment": env_versions,
    })

    # RealESRGAN_x2plus direct PyTorch path.
    # This intentionally avoids the legacy BasicSR/facexlib/GFPGAN pip stack.
    import numpy as np
    from PIL import Image
    from torch import nn
    from torch.nn import functional as F
    from torch.nn import init

    def default_init_weights(modules, scale=1.0):
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()

    def pixel_unshuffle(x, scale):
        b, ch, hh, ww = x.size()
        h, w = hh // scale, ww // scale
        x = x.view(b, ch, h, scale, w, scale)
        return x.permute(0, 1, 3, 5, 2, 4).reshape(b, ch * scale * scale, h, w)

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)
            default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            in_ch = num_in_ch * 4 if scale == 2 else num_in_ch
            self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = pixel_unshuffle(x, 2) if self.scale == 2 else x
            feat = self.conv_first(feat)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    if not ESR_WEIGHTS.exists():
        urlretrieve(ESR_URL, ESR_WEIGHTS)

    esr_model = RRDBNet()
    try:
        checkpoint = torch.load(ESR_WEIGHTS, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ESR_WEIGHTS, map_location="cpu")
    state = checkpoint["params_ema"] if "params_ema" in checkpoint else checkpoint["params"]
    esr_model.load_state_dict(state, strict=True)
    esr_model.eval().half().to("cuda")

    # Validate the exact x2 model before processing all frames.
    with torch.inference_mode():
        probe = torch.zeros((1, 3, 32, 32), device="cuda", dtype=torch.float16)
        probe_out = esr_model(probe)
        if tuple(probe_out.shape[-2:]) != (64, 64):
            raise RuntimeError(f"RealESRGAN x2 preflight failed: {tuple(probe_out.shape)}")
        del probe, probe_out
    torch.cuda.empty_cache()

    for d in (FRAME_DIR, ESR_FRAME_DIR):
        if d.exists():
            import shutil
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(WAN_RAW), "-vsync", "0", str(FRAME_DIR / "%06d.png")
    ])

    def upscale_frame(img, tile=256, tile_pad=10):
        h, w = img.shape[:2]
        out = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        for y0 in range(0, h, tile):
            y1 = min(y0 + tile, h)
            for x0 in range(0, w, tile):
                x1 = min(x0 + tile, w)
                py0, py1 = max(0, y0 - tile_pad), min(h, y1 + tile_pad)
                px0, px1 = max(0, x0 - tile_pad), min(w, x1 + tile_pad)
                crop = img[py0:py1, px0:px1]
                pad_h = crop.shape[0] % 2
                pad_w = crop.shape[1] % 2
                if pad_h or pad_w:
                    crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                inp = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).to("cuda")
                inp = inp.half().div_(255.0)
                with torch.inference_mode():
                    pred = esr_model(inp).clamp_(0, 1)
                pred = (pred[0].float().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
                pred = pred[: (py1 - py0) * 2, : (px1 - px0) * 2]
                sy0, sx0 = (y0 - py0) * 2, (x0 - px0) * 2
                sy1, sx1 = sy0 + (y1 - y0) * 2, sx0 + (x1 - x0) * 2
                out[y0 * 2:y1 * 2, x0 * 2:x1 * 2] = pred[sy0:sy1, sx0:sx1]
                del inp, pred
        return out

    upscale_started = time.perf_counter()
    frame_paths = sorted(FRAME_DIR.glob("*.png"))
    if not frame_paths:
        raise RuntimeError("ffmpeg extracted no Wan frames")
    for idx, frame_path in enumerate(frame_paths, 1):
        img = np.asarray(Image.open(frame_path).convert("RGB"))
        up = upscale_frame(img)
        Image.fromarray(up, "RGB").save(ESR_FRAME_DIR / f"{idx:06d}.png")
    upscale_seconds = time.perf_counter() - upscale_started

    del esr_model
    gc.collect()
    torch.cuda.empty_cache()

    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(CONFIG["fps"]), "-i", str(ESR_FRAME_DIR / "%06d.png"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(UPSCALED),
    ])

    finalize_started = time.perf_counter()
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(UPSCALED),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(OUTPUT),
    ])
    finalize_seconds = time.perf_counter() - finalize_started

    raw_bytes = OUTPUT.read_bytes()
    if len(raw_bytes) < 20000 or raw_bytes[4:8] != b"ftyp":
        raise RuntimeError("invalid final MP4 output")

    meta = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(OUTPUT)
    ], text=True))["streams"][0]
    if int(meta["width"]) != 1080 or int(meta["height"]) != 1920:
        raise RuntimeError(f"final video is not 1080x1920: {meta}")

    save({
        "status": "success",
        "stage": "wan_1_3b_plus_realesrgan_x2",
        "video_model": model_id,
        "upscaler": "RealESRGAN_x2plus_direct_pytorch",
        "gpu_name": torch.cuda.get_device_name(0),
        "setup_seconds": round(setup_seconds, 3),
        "wan_seconds": round(wan_seconds, 3),
        "upscale_seconds": round(upscale_seconds, 3),
        "finalize_seconds": round(finalize_seconds, 3),
        "total_seconds": round(setup_seconds + wan_seconds + upscale_seconds + finalize_seconds, 3),
        "frames": len(frame_paths),
        "fps": int(CONFIG["fps"]),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": len(raw_bytes),
        "environment": env_versions,
        "sha256": sha256(raw_bytes).hexdigest(),
    })
except Exception as exc:
    save({
        "status": "failed",
        "stage": "wan_1_3b_plus_realesrgan_x2",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-20000:],
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
