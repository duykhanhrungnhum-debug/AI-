from __future__ import annotations
import io,json,os,zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request,urlopen
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"smoke-results"
submitted=json.loads((OUT/"submitted.json").read_text(encoding="utf-8"))
actual=submitted.get("response",{}).get("url") or submitted["url"]
parts=[p for p in urlparse(actual).path.split("/") if p]
owner=parts[-2]; slug=parts[-1]
token=os.environ.get("KAGGLE_API_TOKEN","").strip()
url=f"https://www.kaggle.com/api/v1/kernels/output/download/{owner}/{slug}?version_number=1"
headers={"User-Agent":"AgentNate-Smoke-Output/1.0","Accept":"application/zip,*/*"}
if token: headers["Authorization"]="Bearer "+token
req=Request(url,headers=headers)
try:
    with urlopen(req,timeout=600) as r:
        raw=r.read()
        final_url=r.geturl()
        status=r.status
        ctype=r.headers.get("Content-Type","")
except HTTPError as exc:
    body=exc.read().decode("utf-8",errors="replace")
    diag={"url":url,"status":exc.code,"error":body}
    (OUT/"output-download-diagnostic.json").write_text(json.dumps(diag,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(diag,ensure_ascii=False,indent=2))
    raise SystemExit(0)

diag={"url":url,"status":status,"final_url":final_url,"content_type":ctype,"bytes":len(raw),"zip_magic":raw[:4].hex()}
(OUT/"output-download-diagnostic.json").write_text(json.dumps(diag,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(diag,ensure_ascii=False,indent=2))
if raw[:2] != b"PK":
    (OUT/"output-download-body.bin").write_bytes(raw)
    raise SystemExit(0)

with zipfile.ZipFile(io.BytesIO(raw)) as z:
    names=z.namelist()
    print("ZIP FILES",names)
    for name in names:
        base=Path(name).name
        if base in {"agentnate_image.png","agentnate_video.mp4","agentnate_smoke_report.json"}:
            (OUT/base).write_bytes(z.read(name))
    (OUT/"output-zip-files.json").write_text(json.dumps(names,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
