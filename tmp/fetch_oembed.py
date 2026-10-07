from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"smoke-results"
submitted=json.loads((OUT/"submitted.json").read_text(encoding="utf-8"))
page_url=submitted.get("response",{}).get("url") or submitted["url"]
version_id="355978730"
target=page_url+"?scriptVersionId="+version_id
oembed="https://www.kaggle.com/oembed/kernel?url="+quote(target,safe="")
req=Request(oembed,headers={"User-Agent":"Mozilla/5.0","Accept":"application/json,text/html,*/*"})
with urlopen(req,timeout=120) as r:
    raw=r.read()
    ctype=r.headers.get("Content-Type","")
text=raw.decode("utf-8",errors="replace")
(OUT/"oembed-response.txt").write_text(text,encoding="utf-8")
try:
    data=json.loads(text)
except Exception:
    data={"raw":text}
summary={"oembed_url":oembed,"content_type":ctype,"data":data}
(OUT/"oembed-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
