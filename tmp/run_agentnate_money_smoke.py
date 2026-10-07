from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://www.kaggle.com/api/v1"
USERNAME = os.environ.get("KAGGLE_USERNAME", "duykhanhta").strip()
TOKEN = os.environ.get("KAGGLE_API_TOKEN", "").strip()
SLUG = f"agentnate-money-smoke-{int(time.time())}"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tmp" / "agentnate_money_smoke_kernel.py").read_text(encoding="utf-8")
OUT = ROOT / "smoke-results"
OUT.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    raise SystemExit("KAGGLE_API_TOKEN missing")


def req(method: str, path: str, payload=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        BASE + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "AgentNate-Smoke/1.0",
        },
    )
    with urlopen(request, timeout=180) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw)


payload = {
    "slug": f"{USERNAME}/{SLUG}",
    "newTitle": "AgentNate Money Agent Real Smoke",
    "text": SOURCE,
    "language": "python",
    "kernelType": "script",
    "isPrivate": True,
    "enableGpu": True,
    "enableTpu": False,
    "enableInternet": True,
    "machineShape": "NvidiaTeslaT4",
}
print("Submitting", payload["slug"], flush=True)
res = req("POST", "/kernels/push", payload)
if res.get("error"):
    raise RuntimeError(res["error"])

deadline = time.time() + 5400
last = None
while time.time() < deadline:
    q = urlencode({"userName": USERNAME, "kernelSlug": SLUG})
    st = req("GET", f"/kernels/status?{q}")
    last = str(st.get("status") or "")
    print("STATUS", last, flush=True)
    if last.upper() in {"COMPLETE", "ERROR", "FAILED", "CANCELLED"}:
        if last.upper() != "COMPLETE":
            raise RuntimeError(f"Kaggle job failed: {st}")
        break
    time.sleep(20)
else:
    raise TimeoutError(f"Kaggle job timeout; last={last}")

q = urlencode({"userName": USERNAME, "kernelSlug": SLUG})
meta = req("GET", f"/kernels/output?{q}")
files = meta.get("files") or []
wanted = {
    "agentnate_image.png",
    "agentnate_video.mp4",
    "agentnate_smoke_report.json",
}
found = {}
for item in files:
    name = item.get("fileName", item.get("file_name"))
    url = item.get("url")
    if name in wanted and isinstance(url, str):
        found[name] = url

missing = wanted - set(found)
if missing:
    raise RuntimeError(f"Missing Kaggle outputs: {sorted(missing)}; metadata={meta}")

for name, url in found.items():
    print(f"DOWNLOAD_URL {name} {url}", flush=True)
    request = Request(url, headers={"User-Agent": "AgentNate-Smoke/1.0"})
    with urlopen(request, timeout=300) as response:
        data = response.read()
    (OUT / name).write_bytes(data)

report = json.loads((OUT / "agentnate_smoke_report.json").read_text(encoding="utf-8"))
if report.get("status") != "success":
    raise RuntimeError(f"Smoke report failed: {report}")
print("SMOKE_SUCCESS", json.dumps(report, ensure_ascii=False), flush=True)
