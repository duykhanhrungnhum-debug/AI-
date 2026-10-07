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

wanted=["agentnate_image.png","agentnate_video.mp4","agentnate_smoke_report.json"]
report={}
for name in wanted:
    url=f"https://www.kaggle.com/api/v1/kernels/output/download/{owner}/{slug}/{quote(name)}?version_number=1"
    req=Request(url,headers=headers)
    try:
        with urlopen(req,timeout=600) as r:
            try:
                raw=r.read()
            except IncompleteRead as exc:
                raw=exc.partial
            status=r.status
            final_url=r.geturl()
            ctype=r.headers.get("Content-Type","")
        report[name]={"status":status,"bytes":len(raw),"final_url":final_url,"content_type":ctype}
        if raw:
            (OUT/name).write_bytes(raw)
    except HTTPError as exc:
        body=exc.read().decode("utf-8",errors="replace")
        report[name]={"status":exc.code,"error":body[:4000]}

(OUT/"individual-download-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))

img=OUT/"agentnate_image.png"
vid=OUT/"agentnate_video.mp4"
rep=OUT/"agentnate_smoke_report.json"
if img.is_file():
    assert img.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "invalid PNG"
if vid.is_file():
    b=vid.read_bytes()
    assert len(b)>20000 and b[4:8]==b"ftyp", "invalid MP4"
if rep.is_file():
    data=json.loads(rep.read_text(encoding="utf-8"))
    print("SMOKE_REPORT",json.dumps(data,ensure_ascii=False))
