from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

OUT=Path("diag-results")
OUT.mkdir(exist_ok=True)
user=os.environ.get("KAGGLE_USERNAME","duykhanhta")
token=os.environ.get("KAGGLE_API_TOKEN","").strip()
if not token:
    raise SystemExit("missing token")
os.environ["KAGGLE_API_TOKEN"] = token

subprocess.check_call([sys.executable,"-m","pip","install","--quiet","kaggle"])
cmd=["kaggle","kernels","list","--user",user,"--sort-by","dateRun","--page","1","--csv"]
proc=subprocess.run(cmd,capture_output=True,text=True)
(OUT/"kernels-list.stdout.txt").write_text(proc.stdout,encoding="utf-8")
(OUT/"kernels-list.stderr.txt").write_text(proc.stderr,encoding="utf-8")
print("LIST_RC",proc.returncode)
print(proc.stdout)
print(proc.stderr)

candidates=[]
for line in proc.stdout.splitlines():
    if "money-flux-ltx-" in line or "ma-flux-ltx-" in line:
        ref=line.split(",")[0].strip().strip('"')
        candidates.append(ref)

print("CANDIDATES",candidates)
headers={"Authorization":f"Bearer {token}","User-Agent":"Money-Agent-Diag/1.0","Accept":"*/*"}
results={}
for ref in candidates[:5]:
    owner,slug=ref.split("/",1)
    url=f"https://www.kaggle.com/api/v1/kernels/output/download/{owner}/{slug}/flux_ltx_report.json?version_number=1"
    req=Request(url,headers=headers)
    try:
        with urlopen(req,timeout=120) as r:
            raw=r.read()
        text=raw.decode("utf-8",errors="replace")
        results[ref]={"status":200,"text":text[:30000]}
        if text.strip().startswith("{"):
            try:
                data=json.loads(text)
                (OUT/f"{slug}-report.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            except Exception:
                pass
    except HTTPError as exc:
        body=exc.read().decode("utf-8",errors="replace")
        results[ref]={"status":exc.code,"text":body[:5000]}
    except Exception as exc:
        results[ref]={"status":"error","text":repr(exc)}

(OUT/"diagnostic.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(results,ensure_ascii=False,indent=2))
