from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "smoke-results"
submitted = json.loads((OUT / "submitted.json").read_text(encoding="utf-8"))
url = submitted.get("response", {}).get("url") or submitted["url"]

req = Request(url, headers={
    "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept":"text/html,application/xhtml+xml",
})
with urlopen(req, timeout=120) as r:
    html = r.read().decode("utf-8", errors="replace")

(OUT / "kaggle-public-page.html").write_text(html, encoding="utf-8")

patterns = [
    r'https?://[^"\s<>]+',
    r'"[^"]*(?:output|download|status|kernel|file)[^"]*"',
]
hits=[]
for pat in patterns:
    for m in re.findall(pat, html, flags=re.I):
        if isinstance(m, tuple):
            m=" ".join(m)
        s=str(m)
        if any(k in s.lower() for k in ("output","download","status","kernel","agentnate","137444801")):
            hits.append(s[:1000])
summary={
    "url":url,
    "html_bytes":len(html.encode("utf-8")),
    "contains_agentnate": "agentnate" in html.lower(),
    "contains_kernel_id": "137444801" in html,
    "contains_output": "output" in html.lower(),
    "contains_complete": "complete" in html.lower(),
    "hits":hits[:200],
}
(OUT / "page-inspect.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
