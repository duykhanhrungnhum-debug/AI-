from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

OUT=Path("wan-diag-results")
OUT.mkdir(exist_ok=True)
user=os.environ.get("KAGGLE_USERNAME","duykhanhta").strip()
token=os.environ.get("KAGGLE_API_TOKEN","").strip()
if not token:
    raise SystemExit("missing token")
os.environ["KAGGLE_API_TOKEN"]=token

subprocess.check_call([sys.executable,"-m","pip","install","--quiet","kaggle"])
ref="duykhanhta/ma-wan-realesrgan-91365882"

results={}
for name,cmd in {
    "status":["kaggle","kernels","status",ref],
    "list":["kaggle","kernels","list","--user",user,"--sort-by","dateRun","--page","1","--csv"],
}.items():
    proc=subprocess.run(cmd,capture_output=True,text=True)
    results[name]={"rc":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr}
    (OUT/f"{name}.stdout.txt").write_text(proc.stdout,encoding="utf-8")
    (OUT/f"{name}.stderr.txt").write_text(proc.stderr,encoding="utf-8")

outdir=OUT/"kernel-output"
outdir.mkdir(exist_ok=True)
proc=subprocess.run(["kaggle","kernels","output",ref,"-p",str(outdir)],capture_output=True,text=True)
results["output"]={"rc":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr}
(OUT/"output.stdout.txt").write_text(proc.stdout,encoding="utf-8")
(OUT/"output.stderr.txt").write_text(proc.stderr,encoding="utf-8")

for p in outdir.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".html",".txt",".json",".log",".ipynb"}:
        try:
            txt=p.read_text(encoding="utf-8",errors="replace")
            (OUT/f"EXTRACT_{p.name}.txt").write_text(txt[-200000:],encoding="utf-8")
        except Exception:
            pass

(OUT/"diagnostic.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(results,ensure_ascii=False,indent=2))
