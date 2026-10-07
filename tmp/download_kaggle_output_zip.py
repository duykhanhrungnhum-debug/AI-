from __future__ import annotations
import json,os
from pathlib import Path
from urllib.parse import quote,urlparse
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from http.client import IncompleteRead

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"smoke-results"
OUT.mkdir(parents=True,exist_ok=True)
submitted=json.loads((OUT/"submitted.json").read_text(encoding="utf-8"))
actual=submitted.get("response",{}).get("url") or submitted["url"]
parts=[p for p in urlparse(actual).path.split("/") if p]
owner=parts[-2]; slug=parts[-1]
token=os.environ.get("KAGGLE_API_TOKEN","").strip()
headers={"User-Agent":"AgentNate-Smoke-Output/1.0","Accept":"*/*"}
if token: headers["Authorization"]="Bearer "+token

wanted={
 "agentnate_image.png":"agentnate_image.png",
 "ComfyUI/output/AgentNate_svd_00001_.webp":"agentnate_video.webp",
 "agentnate_smoke_report.json":"agentnate_smoke_report.json",
}
report={}
for remote,local in wanted.items():
    encoded=quote(remote,safe="/")
    url=f"https://www.kaggle.com/api/v1/kernels/output/download/{owner}/{slug}/{encoded}?version_number=1"
    req=Request(url,headers=headers)
    try:
        with urlopen(req,timeout=600) as r:
            try: raw=r.read()
            except IncompleteRead as exc: raw=exc.partial
            report[local]={"status":r.status,"bytes":len(raw),"final_url":r.geturl(),"content_type":r.headers.get("Content-Type","")}
        if raw: (OUT/local).write_bytes(raw)
    except HTTPError as exc:
        body=exc.read().decode("utf-8",errors="replace")
        report[local]={"status":exc.code,"error":body[:4000]}

(OUT/"individual-download-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))

img=OUT/"agentnate_image.png"
webp=OUT/"agentnate_video.webp"
if img.is_file(): assert img.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
if webp.is_file():
    b=webp.read_bytes()
    assert len(b)>10000 and b[:4]==b"RIFF" and b[8:12]==b"WEBP", "invalid WEBP"
