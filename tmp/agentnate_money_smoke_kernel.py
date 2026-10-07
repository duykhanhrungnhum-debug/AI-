from __future__ import annotations

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
from urllib.request import Request, urlopen

WORK = Path("/kaggle/working")
AGENTNATE_DIR = WORK / "AgentNate"
COMFY = WORK / "ComfyUI"
REPORT = WORK / "agentnate_smoke_report.json"
PORT = 8188

PROMPT = (
    "A beautiful young Vietnamese woman sitting at an outdoor riverside cafe in soft morning light, "
    "natural realistic skin, casual elegant cream blouse and blue jeans, warm candid lifestyle photography, "
    "lush green background, subtle breeze, cinematic composition, realistic hands, no text, no watermark"
)

AGENTNATE_COMMIT = "dcef1a702ae661cd1c21367e213ae7835e5c0af6"
COMFYUI_COMMIT = "c9d8a6e69c4b5ab7fa0f789e7b988c172cd31fc9"

IMAGE_REPO = "stabilityai/sdxl-turbo"
IMAGE_FILE = "sd_xl_turbo_1.0_fp16.safetensors"
VIDEO_REPO = "stabilityai/stable-video-diffusion-img2vid-xt"
VIDEO_FILE = "svd_xt.safetensors"


def save(data):
    REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def request_json(method, path, body=None, timeout=60):
    raw = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=raw,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_server(timeout=240):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            request_json("GET", "/system_stats", timeout=5)
            return
        except Exception as exc:
            last = exc
            time.sleep(2)
    raise RuntimeError(f"ComfyUI did not start: {last}")


def submit_workflow(workflow):
    data = request_json("POST", "/prompt", {"prompt": workflow}, timeout=60)
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
    return prompt_id


def wait_workflow(prompt_id, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = request_json("GET", f"/history/{prompt_id}", timeout=30)
        entry = data.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI workflow failed: {status.get('messages', [])}")
            if entry.get("outputs"):
                return entry
        time.sleep(5)
    raise TimeoutError(f"ComfyUI workflow timeout: {prompt_id}")


def newest_output(prefix, extensions):
    candidates = []
    for ext in extensions:
        candidates.extend((COMFY / "output").rglob(f"{prefix}*{ext}"))
    if not candidates:
        raise FileNotFoundError(f"No output for prefix {prefix}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def checkout(url, path, commit):
    subprocess.check_call(["git", "clone", "--quiet", "--filter=blob:none", url, str(path)])
    subprocess.check_call(["git", "-C", str(path), "fetch", "--quiet", "--depth", "1", "origin", commit])
    subprocess.check_call(["git", "-C", str(path), "checkout", "--quiet", "FETCH_HEAD"])


process = None
log = None
try:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    checkout("https://github.com/aivrar/AgentNate.git", AGENTNATE_DIR, AGENTNATE_COMMIT)
    sys.path.insert(0, str(AGENTNATE_DIR))
    from backend.comfyui_workflow_templates import build_ready_made

    checkout("https://github.com/Comfy-Org/ComfyUI.git", COMFY, COMFYUI_COMMIT)

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "-r", str(COMFY / "requirements.txt"),
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "huggingface_hub", "hf_xet", "imageio-ffmpeg",
    ])

    from huggingface_hub import hf_hub_download

    image_path = hf_hub_download(
        repo_id=IMAGE_REPO,
        filename=IMAGE_FILE,
        local_dir=str(COMFY / "models" / "checkpoints"),
    )
    if not Path(image_path).is_file():
        raise RuntimeError("SDXL Turbo checkpoint download failed")

    log = (WORK / "comfyui.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable, str(COMFY / "main.py"),
            "--listen", "127.0.0.1",
            "--port", str(PORT),
            "--lowvram",
            "--disable-auto-launch",
        ],
        cwd=str(COMFY),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    wait_server()

    image_workflow, image_warnings = build_ready_made(
        "txt2img",
        {
            "checkpoint": IMAGE_FILE,
            "prompt": PROMPT,
            "negative_prompt": (
                "blurry, low quality, distorted anatomy, extra fingers, extra limbs, "
                "deformed face, text, watermark, plastic skin"
            ),
            "width": 768,
            "height": 768,
            "steps": 4,
            "cfg": 1.0,
            "seed": 20261007,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "filename_prefix": "AgentNate_image",
        },
    )
    image_id = submit_workflow(image_workflow)
    wait_workflow(image_id, timeout=1200)
    generated_image = newest_output("AgentNate_image", [".png", ".jpg", ".jpeg", ".webp"])
    final_image = WORK / "agentnate_image.png"
    from PIL import Image
    Image.open(generated_image).convert("RGB").save(final_image, "PNG")

    try:
        request_json("POST", "/free", {"unload_models": True, "free_memory": True}, timeout=30)
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    # Remove the image checkpoint before downloading SVD to conserve Kaggle disk.
    try:
        Path(image_path).unlink()
    except Exception:
        pass

    video_path = hf_hub_download(
        repo_id=VIDEO_REPO,
        filename=VIDEO_FILE,
        local_dir=str(COMFY / "models" / "checkpoints"),
    )
    if not Path(video_path).is_file():
        raise RuntimeError("SVD XT checkpoint download failed")

    prepared = COMFY / "input" / "agentnate_input.png"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_image, prepared)

    video_workflow, video_warnings = build_ready_made(
        "svd_img2video",
        {
            "checkpoint": VIDEO_FILE,
            "image_path": prepared.name,
            "width": 1024,
            "height": 576,
            "video_frames": 14,
            "motion_bucket_id": 95,
            "fps": 6,
            "output_fps": 6,
            "augmentation_level": 0.0,
            "min_cfg": 1.0,
            "steps": 15,
            "cfg": 2.5,
            "seed": 20261007,
            "sampler_name": "euler_ancestral",
            "scheduler": "karras",
            "filename_prefix": "AgentNate_svd",
        },
    )
    video_id = submit_workflow(video_workflow)
    wait_workflow(video_id, timeout=1800)
    animated = newest_output("AgentNate_svd", [".webp", ".gif", ".mp4", ".webm"])

    final_video = WORK / "agentnate_video.mp4"
    if animated.suffix.lower() == ".mp4":
        shutil.copy2(animated, final_video)
    else:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.check_call([
            ffmpeg, "-y", "-i", str(animated),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(final_video),
        ])

    image_bytes = final_image.read_bytes()
    video_bytes = final_video.read_bytes()
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("invalid PNG output")
    if len(video_bytes) < 20000 or video_bytes[4:8] != b"ftyp":
        raise RuntimeError("invalid MP4 output")

    save({
        "status": "success",
        "gpu": torch.cuda.get_device_name(0),
        "agentnate_commit": AGENTNATE_COMMIT,
        "comfyui_commit": COMFYUI_COMMIT,
        "image_workflow": "txt2img",
        "video_workflow": "svd_img2video",
        "image_warnings": image_warnings,
        "video_warnings": video_warnings,
        "image_sha256": sha256(image_bytes).hexdigest(),
        "video_sha256": sha256(video_bytes).hexdigest(),
        "image_bytes": len(image_bytes),
        "video_bytes": len(video_bytes),
    })
except Exception as exc:
    tail = ""
    try:
        if log:
            log.flush()
        log_path = WORK / "comfyui.log"
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
    except Exception:
        pass
    save({
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-12000:],
        "comfyui_log_tail": tail,
    })
    raise
finally:
    if process is not None:
        try:
            process.terminate()
            process.wait(timeout=15)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    if log is not None:
        try:
            log.close()
        except Exception:
            pass
