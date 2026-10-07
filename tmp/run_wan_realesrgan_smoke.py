from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

BASE = "https://www.kaggle.com/api/v1"
USER = os.environ.get("KAGGLE_USERNAME", "duykhanhta").strip()
TOKEN = os.environ.get("KAGGLE_API_TOKEN", "").strip()
if not TOKEN:
    raise SystemExit("KAGGLE_API_TOKEN missing")

money_file = Path("tmp/money_agent_kaggle_tools_snapshot.py")
spec = importlib.util.spec_from_file_location("money_agent_kaggle_tools", money_file)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import Money Agent kaggle_tools.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUT = Path("wan-realesrgan-results")
OUT.mkdir(parents=True, exist_ok=True)
stamp = str(int(time.time()))
run_token = f"wan-esr-smoke-{stamp}"

config = {
    "run_token": run_token,
    "video_model": mod.VIDEO_MODEL,
    "wan_commit": mod.WAN_REPO_COMMIT,
    "realesrgan_commit": mod.REALESRGAN_REPO_COMMIT,
    "prompt": (
        "Photorealistic vertical social media video of a beautiful young Vietnamese woman "
        "sitting naturally at an outdoor riverside cafe in soft morning light. She looks "
        "toward the camera, blinks naturally, lightly turns her head and gives a subtle warm smile. "
        "Her hair moves gently in the breeze. Natural skin texture, stable face, realistic hands "
        "and body proportions, smooth realistic motion, steady camera, no text, no watermark."
    ),
    "wan_size": "480*832",
    "num_frames": 33,
    "fps": 16,
    "sample_steps": 30,
    "upscale_factor": 1920 / 832,
    "seed": 20261007,
}
kernel = mod._video_source(config)

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
            "User-Agent": "Money-Agent-Wan-RealESRGAN-Smoke/1.0",
        },
    )
    with urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw)

slug = f"wan-realesrgan-{stamp}"
payload = {
    "slug": f"{USER}/{slug}",
    "newTitle": f"MA Wan RealESRGAN {stamp[-8:]}",
    "text": kernel,
    "language": "python",
    "kernelType": "script",
    "isPrivate": False,
    "enableGpu": True,
    "enableTpu": False,
    "enableInternet": True,
    "machineShape": "NvidiaTeslaT4",
}
res = api("POST", "/kernels/push", payload)
if res.get("error"):
    raise RuntimeError(str(res["error"]))

url = res.get("url") or f"https://www.kaggle.com/code/{USER}/{slug}"
parts = [p for p in urlparse(url).path.split("/") if p]
owner, actual_slug = parts[-2], parts[-1]
(OUT / "submitted.json").write_text(
    json.dumps({"url": url, "owner": owner, "slug": actual_slug, "config": config}, indent=2),
    encoding="utf-8",
)
print("SUBMITTED", url, flush=True)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "*/*",
    "User-Agent": "Money-Agent-Wan-RealESRGAN-Smoke/1.0",
}

def download(name: str, timeout=90):
    u = f"{BASE}/kernels/output/download/{owner}/{actual_slug}/{quote(name, safe='/')}?version_number=1"
    req = Request(u, headers=headers)
    with urlopen(req, timeout=timeout) as r:
        return r.read()

deadline = time.time() + 5400
report = None
while time.time() < deadline:
    try:
        raw = download("manager_video_report.json", 45)
        if raw:
            report = json.loads(raw.decode("utf-8"))
            print("REPORT", json.dumps(report, ensure_ascii=False), flush=True)
            if report.get("status") in {"success", "failed"}:
                (OUT / "manager_video_report.json").write_bytes(raw)
                break
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        pass
    time.sleep(30)

if report is None:
    raise TimeoutError("No final Wan + Real-ESRGAN report")

if report.get("status") == "success":
    video = download("manager_video.mp4", 300)
    (OUT / "manager_video.mp4").write_bytes(video)
    print("VIDEO_BYTES", len(video), flush=True)
    if report.get("width") != 1080 or report.get("height") != 1920:
        raise RuntimeError(f"wrong final resolution: {report.get('width')}x{report.get('height')}")
else:
    raise RuntimeError(report.get("error", "Wan + Real-ESRGAN smoke failed"))
