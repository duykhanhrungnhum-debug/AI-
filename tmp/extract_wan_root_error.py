from __future__ import annotations
import html, os, re
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote

OUT=Path("wan-log-results"); OUT.mkdir(exist_ok=True)
token=os.environ.get("KAGGLE_API_TOKEN","").strip()
if not token: raise SystemExit("missing token")
owner="duykhanhta"
slug="ma-wan-realesrgan-91365882"
base="https://www.kaggle.com/api/v1/kernels/output/download"
headers={"Authorization":f"Bearer {token}","Accept":"*/*"}

found=False
for name in ["__results__.html","__notebook__.ipynb","manager_video_report.json"]:
    url=f"{base}/{owner}/{slug}/{quote(name,safe='/')}?version_number=1"
    try:
        with urlopen(Request(url,headers=headers),timeout=90) as r:
            raw=r.read()
        p=OUT/name.replace("/","_")
        p.write_bytes(raw)
        txt=raw.decode("utf-8",errors="replace")
        plain=re.sub(r"<script[\s\S]*?</script>"," ",txt,flags=re.I)
        plain=re.sub(r"<style[\s\S]*?</style>"," ",plain,flags=re.I)
        plain=re.sub(r"<[^>]+>","\n",plain)
        plain=html.unescape(plain)
        lines=[ln.strip() for ln in plain.splitlines() if ln.strip()]
        hits=[]
        for i,ln in enumerate(lines):
            low=ln.lower()
            if any(k in low for k in ["traceback","error","exception","failed","no module","not found","runtimeerror","valueerror","typeerror","cuda","torch","attributeerror"]):
                lo=max(0,i-3); hi=min(len(lines),i+8)
                hits.extend(lines[lo:hi])
        text="\n".join(hits[-600:])
        (OUT/f"EXTRACT_{p.name}.txt").write_text(text,encoding="utf-8")
        print(f"DOWNLOADED {name} {len(raw)} bytes")
        print(text[-30000:])
        found=True
    except Exception as e:
        print("MISS",name,repr(e))
if not found:
    raise SystemExit("no diagnostic file downloaded")
