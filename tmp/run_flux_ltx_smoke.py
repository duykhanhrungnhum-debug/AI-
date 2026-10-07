from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

BASE = "https://www.kaggle.com/api/v1"
USERNAME = os.environ.get("KAGGLE_USERNAME", "duykhanhta").strip()
TOKEN = os.environ.get("KAGGLE_API_TOKEN", "").strip()
OUT = Path("smoke-results")
OUT.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    raise SystemExit("KAGGLE_API_TOKEN missing")

stamp = str(int(time.time()))
PROMPT_IMAGE = (
    "A beautiful young Vietnamese woman sitting naturally at an outdoor riverside cafe in soft morning light, "
    "realistic natural skin texture, elegant casual cream blouse and blue jeans, full head visible, balanced framing, "
    "candid lifestyle photography, detailed hands, lush green background, cinematic but photorealistic, no text, no watermark"
)
PROMPT_VIDEO = (
    "A beautiful young Vietnamese woman at an outdoor riverside cafe in soft morning light. "
    "She looks toward the camera, blinks naturally, lightly turns her head and smiles. "
    "Hair moves gently in the breeze. Stable face and body proportions, realistic motion, natural camera, no text, no watermark."
)

kernel = r'''from __future__ import annotations
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

WORK = Path("/kaggle/working")
REPORT = WORK / "flux_ltx_report.json"
IMAGE_OUT = WORK / "flux2_test.png"
VIDEO_OUT = WORK / "ltx2b_test.mp4"
IMAGE_PROMPT = __IMAGE_PROMPT__
VIDEO_PROMPT = __VIDEO_PROMPT__

def save(data):
    REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

report = {
    "status": "running",
    "image_model": "black-forest-labs/FLUX.2-klein-4B",
    "video_model": "Lightricks/LTX-Video ltxv-2b-0.9.8-distilled",
}
started_total = time.perf_counter()

try:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # IMAGE: exact direct FLUX.2 Klein path proposed for Money Agent.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "diffusers>=0.36", "transformers>=4.57,<5", "accelerate>=1,<2",
        "safetensors", "sentencepiece", "Pillow<13"
    ])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    report["gpu"] = torch.cuda.get_device_name(0)
    report["gpu_count"] = torch.cuda.device_count()

    from diffusers import Flux2KleinPipeline
    t0 = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    load_image_seconds = time.perf_counter() - t0

    generator = torch.Generator(device="cpu").manual_seed(20261007)
    t0 = time.perf_counter()
    image = pipe(
        prompt=IMAGE_PROMPT,
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=generator,
    ).images[0]
    image_seconds = time.perf_counter() - t0
    image.save(IMAGE_OUT, "PNG")

    report["image"] = {
        "status": "success",
        "load_seconds": round(load_image_seconds, 3),
        "inference_seconds": round(image_seconds, 3),
        "bytes": IMAGE_OUT.stat().st_size,
        "sha256": sha256(IMAGE_OUT.read_bytes()).hexdigest(),
    }
    save(report)

    del pipe, image
    gc.collect()
    torch.cuda.empty_cache()

    # VIDEO: official LTX-Video 2B 0.9.8 distilled inference, no ComfyUI.
    LTX = WORK / "LTX-Video"
    if LTX.exists():
        shutil.rmtree(LTX)
    subprocess.check_call([
        "git", "clone", "--quiet", "--filter=blob:none",
        "https://github.com/Lightricks/LTX-Video.git", str(LTX)
    ])
    subprocess.check_call([
        "git", "-C", str(LTX), "checkout", "--quiet",
        "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"
    ])
    t0 = time.perf_counter()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "-e", str(LTX) + "[inference]"
    ])
    install_ltx_seconds = time.perf_counter() - t0

    sys.path.insert(0, str(LTX))
    from ltx_video.inference import infer, InferenceConfig

    out_dir = WORK / "ltx-output"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    infer(
        InferenceConfig(
            pipeline_config=str(LTX / "configs" / "ltxv-2b-0.9.8-distilled.yaml"),
            prompt=VIDEO_PROMPT,
            height=512,
            width=768,
            num_frames=49,
            frame_rate=24,
            seed=20261007,
            output_path=str(out_dir),
            offload_to_cpu=True,
            negative_prompt=(
                "worst quality, inconsistent motion, blurry, jittery, distorted, "
                "deformed anatomy, cropped head, text, watermark"
            ),
        )
    )
    video_seconds = time.perf_counter() - t0

    candidates = list(out_dir.rglob("*.mp4"))
    if not candidates:
        raise FileNotFoundError("LTX-Video produced no MP4")
    generated = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copy2(generated, VIDEO_OUT)
    raw = VIDEO_OUT.read_bytes()
    if len(raw) < 20000 or raw[4:8] != b"ftyp":
        raise RuntimeError("invalid MP4 output")

    report["video"] = {
        "status": "success",
        "install_seconds": round(install_ltx_seconds, 3),
        "inference_seconds": round(video_seconds, 3),
        "frames": 49,
        "fps": 24,
        "approx_seconds": round(49 / 24, 3),
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
    }
    report["status"] = "success"
    report["total_seconds"] = round(time.perf_counter() - started_total, 3)
    save(report)

except Exception as exc:
    report["status"] = "failed"
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()[-16000:]
    report["total_seconds"] = round(time.perf_counter() - started_total, 3)
    save(report)
    raise
'''
kernel = kernel.replace("__IMAGE_PROMPT__", repr(PROMPT_IMAGE)).replace("__VIDEO_PROMPT__", repr(PROMPT_VIDEO))

def api(method: str, path: str, payload=None, timeout=180):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(
        BASE + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Money-Agent-FLUX-LTX-Smoke/1.0",
        },
    )
    with urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw)

submitted = None
for attempt in range(1, 11):
    slug = f"money-flux-ltx-{stamp}-{attempt}"
    payload = {
        "slug": f"{USERNAME}/{slug}",
        "newTitle": f"MA FLUX LTX {stamp[-8:]} {attempt}",
        "text": kernel,
        "language": "python",
        "kernelType": "script",
        "isPrivate": False,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "machineShape": "NvidiaTeslaT4",
    }
    try:
        print("SUBMIT", slug, flush=True)
        res = api("POST", "/kernels/push", payload)
        if res.get("error"):
            msg = str(res["error"])
            if "Maximum batch GPU session count" in msg and attempt < 10:
                print("GPU slots full; waiting 60s", flush=True)
                time.sleep(60)
                continue
            raise RuntimeError(msg)
        submitted = {"requested_slug": slug, "response": res}
        break
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if "Maximum batch GPU session count" in detail and attempt < 10:
            print("GPU slots full; waiting 60s", flush=True)
            time.sleep(60)
            continue
        raise RuntimeError(f"Kaggle HTTP {exc.code}: {detail}") from exc

if not submitted:
    raise RuntimeError("Could not submit Kaggle GPU smoke test")

actual_url = submitted["response"].get("url") or f"https://www.kaggle.com/code/{USERNAME}/{submitted['requested_slug']}"
parts = [p for p in urlparse(actual_url).path.split("/") if p]
owner, actual_slug = parts[-2], parts[-1]
submitted.update({"url": actual_url, "owner": owner, "slug": actual_slug})
(OUT / "submitted.json").write_text(json.dumps(submitted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("ACCEPTED", actual_url, flush=True)

headers = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "Money-Agent-FLUX-LTX-Smoke/1.0"}

def download_output(name: str, timeout=60):
    encoded = quote(name, safe="/")
    url = f"{BASE}/kernels/output/download/{owner}/{actual_slug}/{encoded}?version_number=1"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as r:
        return r.read()

deadline = time.time() + 5400
report = None
while time.time() < deadline:
    try:
        raw = download_output("flux_ltx_report.json", timeout=45)
        if raw:
            report = json.loads(raw.decode("utf-8"))
            print("REPORT", json.dumps(report, ensure_ascii=False), flush=True)
            if report.get("status") in {"success", "failed"}:
                (OUT / "flux_ltx_report.json").write_bytes(raw)
                break
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        pass
    time.sleep(30)

if report is None:
    raise TimeoutError("Timed out waiting for Kaggle test report")

for remote, local in [
    ("flux2_test.png", "flux2_test.png"),
    ("ltx2b_test.mp4", "ltx2b_test.mp4"),
]:
    try:
        data = download_output(remote, timeout=300)
        if data:
            (OUT / local).write_bytes(data)
            print("DOWNLOADED", local, len(data), flush=True)
    except Exception as exc:
        print("OUTPUT_NOT_AVAILABLE", remote, repr(exc), flush=True)

# Keep workflow alive long enough to upload report even on model failure.
print("FINAL_STATUS", report.get("status"), flush=True)
