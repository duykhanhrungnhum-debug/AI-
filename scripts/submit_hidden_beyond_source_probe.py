#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from ai_agent.core.kaggle_worker import KaggleGpuWorker

def download(url:str,path:Path)->None:
    req=Request(url,headers={"User-Agent":"Hidden-Beyond-Source-Probe/1.0"})
    with urlopen(req,timeout=120) as src,path.open("wb") as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk:
                break
            dst.write(chunk)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    template=Path("hidden_beyond/source_probe_worker.py").read_text(encoding="utf-8")
    source=template.replace("# __SOURCE_PROBE_CONFIG_INJECT__","CONFIG = "+repr(config),1)
    worker=KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        timeout=120,
        submission_retry_attempts=3,
        submission_retry_delay_seconds=15,
    )
    slug="hidden-beyond-source-probe"
    s=worker.submit_script(
        slug=slug,
        title="Hidden Beyond Source Probe",
        source=source,
        enable_internet=True,
        enable_gpu=False,
        is_private=True,
    )
    print("SOURCE_PROBE_SUBMITTED",s.ref,s.version_number,flush=True)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    for attempt in range(1,61):
        try:
            meta=worker.output_metadata(slug)
            files=meta.get("files") if isinstance(meta,dict) else []
            for item in files or []:
                name=item.get("fileName",item.get("file_name")) if isinstance(item,dict) else None
                url=item.get("url") if isinstance(item,dict) else None
                if name=="source-probe.json" and isinstance(url,str):
                    download(url,out/"source-probe.json")
                    print((out/"source-probe.json").read_text(encoding="utf-8"),flush=True)
                    return
        except Exception as exc:
            print("SOURCE_PROBE_POLL",attempt,repr(exc),flush=True)
        if time.monotonic()-started>600:
            break
        time.sleep(10)
    try:
        print(worker.logs(slug)[-12000:],flush=True)
    except Exception as exc:
        print("SOURCE_PROBE_LOG_ERROR",repr(exc),flush=True)
    raise SystemExit("source probe output timeout")

if __name__=="__main__":
    main()
