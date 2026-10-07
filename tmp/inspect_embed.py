from __future__ import annotations
import json,re
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"smoke-results"
summary=json.loads((OUT/"oembed-summary.json").read_text(encoding="utf-8"))
html=summary["data"]["html"]
m=re.search(r'src="([^"]+)"',html)
if not m:
    raise RuntimeError("embed src not found")
url=m.group(1).replace("&amp;","&")
req=Request(url,headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36","Accept":"text/html,*/*"})
with urlopen(req,timeout=120) as r:
    raw=r.read()
text=raw.decode("utf-8",errors="replace")
(OUT/"embed-page.html").write_text(text,encoding="utf-8")

needles=["agentnate_image","agentnate_video","agentnate_smoke_report","status","complete","running","output","error","kernelSessionId","355978730"]
hits=[]
low=text.lower()
for needle in needles:
    pos=0
    while True:
        i=low.find(needle.lower(),pos)
        if i<0: break
        hits.append(text[max(0,i-300):min(len(text),i+900)])
        pos=i+len(needle)
        if len(hits)>=100: break
    if len(hits)>=100: break

data={
    "embed_url":url,
    "html_bytes":len(raw),
    "contains_image":"agentnate_image" in low,
    "contains_video":"agentnate_video" in low,
    "contains_report":"agentnate_smoke_report" in low,
    "contains_error":"error" in low,
    "contains_complete":"complete" in low,
    "hits":hits[:100],
}
(OUT/"embed-inspect.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(data,ensure_ascii=False,indent=2))
