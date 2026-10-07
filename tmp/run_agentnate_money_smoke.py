from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.error import HTTPError
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
    try:
        with urlopen(request, timeout=180) as r:
            raw = r.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kaggle HTTP {exc.code}: {detail}") from exc
    return json.loads(raw)


payload = {
    "slug": f"{USERNAME}/{SLUG}",
    "newTitle": "AgentNate Money Agent Real Smoke",
    "text": SOURCE,
    "language": "python",
    "kernelType": "script",
    "isPrivate": False,
    "enableGpu": True,
    "enableTpu": False,
    "enableInternet": True,
    "machineShape": "NvidiaTeslaT4",
}
print("Submitting", payload["slug"], flush=True)
res = None
for attempt in range(1, 16):
    try:
        payload["newTitle"] = f"AgentNate Money Agent Real Smoke {SLUG} attempt {attempt}"
        res = req("POST", "/kernels/push", payload)
        if res.get("error"):
            message = str(res["error"])
            if "Maximum batch GPU session count" in message and attempt < 15:
                print(f"GPU slots full; waiting before submit attempt {attempt + 1}/15", flush=True)
                time.sleep(60)
                continue
            raise RuntimeError(message)
        break
    except RuntimeError as exc:
        message = str(exc)
        if "Maximum batch GPU session count" in message and attempt < 15:
            print(f"GPU slots full; waiting before submit attempt {attempt + 1}/15", flush=True)
            time.sleep(60)
            continue
        raise
if res is None:
    raise RuntimeError("Kaggle submit did not return a response")

record = {
    "status": "submitted",
    "slug": SLUG,
    "owner": USERNAME,
    "url": f"https://www.kaggle.com/code/{USERNAME}/{SLUG}",
    "response": res,
}
(OUT / "submitted.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print("SUBMITTED", json.dumps(record, ensure_ascii=False), flush=True)
