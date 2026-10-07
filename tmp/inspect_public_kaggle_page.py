from __future__ import annotations
import json,re
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"smoke-results"
submitted=json.loads((OUT/"submitted.json").read_text(encoding="utf-8"))
url=submitted.get("response",{}).get("url") or submitted["url"]
req=Request(url,headers={
 "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
 "Accept":"text/html,application/xhtml+xml",
})
with urlopen(req,timeout=120) as r:
    raw=r.read()
text=raw.decode("utf-8",errors="replace")
(OUT/"kaggle-public-page-latest.html").write_text(text,encoding="utf-8")

def first(pattern):
    m=re.search(pattern,text,re.I|re.S)
    return m.group(1) if m else None

rendered=first(r'"renderedOutputUrl"\s*:\s*"([^"]+)"')
if rendered:
    rendered=rendered.replace("\\/","/").replace("\u0026","&")
script_id=first(r'scriptVersionId(?:%3D|=)(\d+)')
patterns=["agentnate_image.png","agentnate_video.mp4","agentnate_smoke_report.json",
          "COMPLETE","RUNNING","ERROR","renderedOutputUrl","outputFiles","scriptVersionId"]
snips={}
low=text.lower()
for p in patterns:
    i=low.find(p.lower())
    if i>=0: snips[p]=text[max(0,i-300):min(len(text),i+1200)]
summary={
 "url":url,"bytes":len(raw),"renderedOutputUrl":rendered,"scriptVersionId":script_id,
 "contains_recap":"Checking your browser before accessing" in text,
 "contains_image":"agentnate_image.png" in text,
 "contains_video":"agentnate_video.mp4" in text,
 "snippets":snips,
}
(OUT/"latest-page-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
