#!/usr/bin/env python3
from __future__ import annotations

# __SOURCE_CONFIG_INJECT__

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

WORK=Path("/kaggle/working")
OUT=WORK/"source.mp4"
META=WORK/"source-metadata.json"

def post(path:str,payload:dict)->dict:
    data=json.dumps(payload,ensure_ascii=False).encode()
    req=Request(
        CONFIG["callback_base"].rstrip("/")+path,
        data=data,
        headers={"content-type":"application/json","x-handoff-token":CONFIG["handoff_token"]},
        method="POST",
    )
    with urlopen(req,timeout=120) as r:
        return json.loads(r.read().decode())

def upload_file(url:str,path:Path)->None:
    data=path.read_bytes()
    req=Request(url,data=data,headers={"content-type":"video/mp4","x-upsert":"true"},method="PUT")
    with urlopen(req,timeout=900) as r:
        if r.status not in (200,201):
            raise RuntimeError(f"source upload failed {r.status}")

def main():
    subprocess.run(
        [sys.executable,"-m","pip","install","--quiet","yt-dlp[default]>=2026.1"],
        check=True,
    )
    cmd=[
        sys.executable,"-m","yt_dlp",
        "--no-playlist","--retries","5","--fragment-retries","5",
        "--remote-components","ejs:github",
    ]
    if shutil.which("node"):
        cmd += ["--js-runtimes","node"]
    cmd += [
        "-f","bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format","mp4",
        "-o",str(OUT),
        CONFIG["source_url"],
    ]
    subprocess.run(cmd,check=True)
    if not OUT.exists() or OUT.stat().st_size<1_000_000:
        raise RuntimeError("source download missing or unexpectedly small")
    duration=float(subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=nw=1:nk=1",str(OUT)
    ],text=True).strip())
    video=subprocess.run([
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=codec_type","-of","csv=p=0",str(OUT)
    ],text=True,capture_output=True,check=False)
    audio=subprocess.run([
        "ffprobe","-v","error","-select_streams","a:0",
        "-show_entries","stream=codec_type","-of","csv=p=0",str(OUT)
    ],text=True,capture_output=True,check=False)
    if "video" not in video.stdout or "audio" not in audio.stdout:
        raise RuntimeError("downloaded source missing video/audio stream")
    sha=hashlib.sha256()
    with OUT.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            sha.update(chunk)
    report={
        "ok":True,
        "source_video_id":CONFIG.get("source_video_id"),
        "source_url":CONFIG.get("source_url"),
        "duration_seconds":round(duration,3),
        "size_bytes":OUT.stat().st_size,
        "sha256":sha.hexdigest(),
        "retriever":"kaggle-cpu-yt-dlp-ejs",
    }
    META.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    post("/heartbeat",{"handoff_id":CONFIG["handoff_id"],"message":"source_downloaded_uploading"})
    upload_file(CONFIG["upload_url"],OUT)
    result=post("/complete",{"handoff_id":CONFIG["handoff_id"],"metadata":report})
    print("SOURCE_DOWNLOAD_OK",json.dumps(report,ensure_ascii=False),flush=True)
    print("SOURCE_HANDOFF_COMPLETE",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        try:
            post("/fail",{"handoff_id":CONFIG.get("handoff_id"),"error":repr(exc)})
        except Exception as fail_exc:
            print("SOURCE_HANDOFF_FAIL_REPORT_ERROR",repr(fail_exc),flush=True)
        raise
