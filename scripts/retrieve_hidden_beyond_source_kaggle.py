#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from ai_agent.core.kaggle_worker import KaggleGpuWorker

def stream_download(url:str,path:Path)->None:
    req=Request(url,headers={"User-Agent":"Hidden-Beyond-Source-Retriever/1.0"})
    with urlopen(req,timeout=180) as src,path.open("wb") as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk:
                break
            dst.write(chunk)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not config.get("source_url") or not config.get("source_video_id"):
        raise SystemExit("source_url and source_video_id are required")
    template=Path("hidden_beyond/source_download_worker.py").read_text(encoding="utf-8")
    marker="# __SOURCE_CONFIG_INJECT__"
    if marker not in template:
        raise SystemExit("source worker marker missing")
    source=template.replace(marker,"CONFIG = "+repr({
        "source_url":config["source_url"],
        "source_video_id":config["source_video_id"],
    }),1)

    worker=KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        timeout=120,
        submission_retry_attempts=4,
        submission_retry_delay_seconds=20,
    )
    safe="".join(ch for ch in str(config["source_video_id"]) if ch.isalnum() or ch in "-_")[:30]
    slug=f"hidden-beyond-source-{safe.lower()}"
    submission=worker.submit_script(
        slug=slug,
        title=f"HB Source {safe}",
        source=source,
        enable_internet=True,
        enable_gpu=False,
        is_private=True,
    )
    print("SOURCE_CPU_SUBMITTED",submission.ref,submission.version_number,flush=True)

    wanted={"source.mp4","source-metadata.json"}
    urls={}
    started=time.monotonic()
    permission_denied_streak=0
    for attempt in range(1,121):
        try:
            meta=worker.output_metadata(slug)
            permission_denied_streak=0
            files=meta.get("files") if isinstance(meta,dict) else []
            urls={}
            for item in files or []:
                if not isinstance(item,dict):
                    continue
                name=item.get("fileName",item.get("file_name"))
                url=item.get("url")
                if name in wanted and isinstance(url,str) and url.startswith(("http://","https://")):
                    urls[str(name)]=url
            print(f"SOURCE_CPU_WATCH attempt={attempt} files={sorted(urls)}",flush=True)
            if wanted.issubset(urls):
                break
        except Exception as exc:
            detail=repr(exc)
            print("SOURCE_CPU_WATCH_ERROR",attempt,detail,flush=True)
            if "403" in detail and ("kernels.get" in detail or "Permission" in detail):
                permission_denied_streak+=1
                if permission_denied_streak>=3:
                    raise SystemExit(
                        "Kaggle private output API denied kernels.get three times; "
                        "stop immediately instead of polling. Use direct storage handoff."
                    )
            else:
                permission_denied_streak=0
        if time.monotonic()-started>1200:
            break
        time.sleep(10)
    if not wanted.issubset(urls):
        try:
            print(worker.logs(slug)[-12000:],flush=True)
        except Exception as exc:
            print("SOURCE_CPU_LOG_ERROR",repr(exc),flush=True)
        raise SystemExit("Kaggle CPU source download timeout or missing output")

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    stream_download(urls["source.mp4"],out)
    stream_download(urls["source-metadata.json"],out.parent/"source-metadata.json")
    report=json.loads((out.parent/"source-metadata.json").read_text(encoding="utf-8"))
    if report.get("ok") is not True:
        raise SystemExit(f"source metadata not OK: {report}")
    if int(report.get("size_bytes") or 0)!=out.stat().st_size:
        raise SystemExit("source size mismatch")
    print("SOURCE_CPU_READY",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
