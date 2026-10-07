from __future__ import annotations

import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "smoke-results"
OUT.mkdir(parents=True, exist_ok=True)
submitted = json.loads((OUT / "submitted.json").read_text(encoding="utf-8"))
actual_url = submitted.get("response", {}).get("url") or submitted["url"]
slug = [p for p in urlparse(actual_url).path.split("/") if p][-1]
owner = submitted.get("owner", "duykhanhta")
BASE = "https://www.kaggle.com/api/v1"


def get_json(path: str):
    req = Request(BASE + path, headers={"Accept":"application/json","User-Agent":"AgentNate-Smoke-Fetch/1.0"})
    try:
        with urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


print("PUBLIC_KERNEL", owner, slug, actual_url, flush=True)
deadline = time.time() + 5400
last = ""
while time.time() < deadline:
    q = urlencode({"userName": owner, "kernelSlug": slug})
    st = get_json(f"/kernels/status?{q}")
    last = str(st.get("status") or "")
    print("STATUS", last, flush=True)
    if last.upper() in {"COMPLETE", "ERROR", "FAILED", "CANCELLED"}:
        if last.upper() != "COMPLETE":
            raise RuntimeError(f"Public Kaggle job failed: {st}")
        break
    time.sleep(20)
else:
    raise TimeoutError(f"Public Kaggle job timeout; last={last}")

q = urlencode({"userName": owner, "kernelSlug": slug})
meta = get_json(f"/kernels/output?{q}")
wanted = {"agentnate_image.png","agentnate_video.mp4","agentnate_smoke_report.json"}
found = {}
for item in meta.get("files") or []:
    name = item.get("fileName", item.get("file_name"))
    url = item.get("url")
    if name in wanted and isinstance(url, str):
        found[name] = url
missing = wanted - set(found)
if missing:
    raise RuntimeError(f"Missing outputs {sorted(missing)}; metadata={meta}")

for name,url in found.items():
    print("DOWNLOAD", name, url, flush=True)
    req = Request(url, headers={"User-Agent":"AgentNate-Smoke-Fetch/1.0"})
    with urlopen(req, timeout=300) as r:
        (OUT / name).write_bytes(r.read())

report = json.loads((OUT / "agentnate_smoke_report.json").read_text(encoding="utf-8"))
print("REPORT", json.dumps(report, ensure_ascii=False), flush=True)
if report.get("status") != "success":
    raise RuntimeError(f"Smoke report failed: {report}")
