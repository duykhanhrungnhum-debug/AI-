#!/usr/bin/env python3
from __future__ import annotations

# __SOURCE_PROBE_CONFIG_INJECT__

import json
import shutil
import subprocess
import sys
from pathlib import Path

OUT=Path("/kaggle/working/source-probe.json")

def main():
    subprocess.run(
        [sys.executable,"-m","pip","install","--quiet","yt-dlp[default]>=2026.1"],
        check=True,
    )
    node=shutil.which("node")
    node_version=""
    if node:
        p=subprocess.run([node,"--version"],text=True,capture_output=True)
        node_version=(p.stdout or p.stderr).strip()
    cmd=[
        sys.executable,"-m","yt_dlp",
        "--no-playlist","--skip-download","--dump-single-json",
        "--remote-components","ejs:github",
    ]
    if node:
        cmd += ["--js-runtimes","node"]
    cmd += [CONFIG["source_url"]]
    p=subprocess.run(cmd,text=True,capture_output=True)
    result={
        "ok":p.returncode==0,
        "returncode":p.returncode,
        "source_url":CONFIG["source_url"],
        "node":node or None,
        "node_version":node_version or None,
        "stderr_tail":p.stderr[-4000:],
    }
    if p.returncode==0:
        try:
            meta=json.loads(p.stdout)
            result.update({
                "id":meta.get("id"),
                "title":meta.get("title"),
                "duration":meta.get("duration"),
                "format_count":len(meta.get("formats") or []),
                "extractor":meta.get("extractor"),
            })
        except Exception as exc:
            result["parse_error"]=repr(exc)
            result["stdout_tail"]=p.stdout[-4000:]
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SOURCE_PROBE_RESULT",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
