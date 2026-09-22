#!/usr/bin/env python3
from __future__ import annotations

# __SOURCE_CONFIG_INJECT__

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

WORK=Path("/kaggle/working")
OUT=WORK/"source.mp4"
META=WORK/"source-metadata.json"

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
    print("SOURCE_DOWNLOAD_OK",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
