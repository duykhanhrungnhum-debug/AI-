from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

OUT=Path("diag-results")
OUT.mkdir(exist_ok=True)
user=os.environ.get("KAGGLE_USERNAME","duykhanhta").strip()
token=os.environ.get("KAGGLE_API_TOKEN","").strip()
if not token:
    raise SystemExit("missing token")
os.environ["KAGGLE_API_TOKEN"]=token

subprocess.check_call([sys.executable,"-m","pip","install","--quiet","kaggle"])
ref="duykhanhta/ma-flux-ltx-91361323-1"

commands={
    "status":["kaggle","kernels","status",ref],
    "list":["kaggle","kernels","list","--user",user,"--sort-by","dateRun","--page","1","--csv"],
}
results={}
for name,cmd in commands.items():
    proc=subprocess.run(cmd,capture_output=True,text=True)
    results[name]={"rc":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr}
    (OUT/f"{name}.stdout.txt").write_text(proc.stdout,encoding="utf-8")
    (OUT/f"{name}.stderr.txt").write_text(proc.stderr,encoding="utf-8")

# Try output command in a separate directory. It may reveal failure details even when no report exists.
outdir=OUT/"kernel-output"
outdir.mkdir(exist_ok=True)
proc=subprocess.run(["kaggle","kernels","output",ref,"-p",str(outdir)],capture_output=True,text=True)
results["output"]={"rc":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr}
(OUT/"output.stdout.txt").write_text(proc.stdout,encoding="utf-8")
(OUT/"output.stderr.txt").write_text(proc.stderr,encoding="utf-8")

(OUT/"diagnostic.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(results,ensure_ascii=False,indent=2))
