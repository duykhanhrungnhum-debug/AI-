#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ai_agent.core.kaggle_worker import KaggleGpuWorker

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--handoff",required=True)
    args=ap.parse_args()

    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    handoff=json.loads(Path(args.handoff).read_text(encoding="utf-8"))
    required_cfg=("source_url","source_video_id")
    required_handoff=("handoff_id","handoff_token","callback_base")
    missing=[k for k in required_cfg if not config.get(k)]
    missing += [k for k in required_handoff if not handoff.get(k)]
    if missing:
        raise SystemExit(f"missing source handoff fields: {missing}")

    template=Path("hidden_beyond/source_download_worker.py").read_text(encoding="utf-8")
    marker="# __SOURCE_CONFIG_INJECT__"
    if marker not in template:
        raise SystemExit("source worker marker missing")
    payload={
        "source_url":config["source_url"],
        "source_video_id":config["source_video_id"],
        "handoff_id":handoff["handoff_id"],
        "handoff_token":handoff["handoff_token"],
        "callback_base":handoff["callback_base"],
    }
    source=template.replace(marker,"CONFIG = "+repr(payload),1)

    worker=KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        timeout=120,
        submission_retry_attempts=4,
        submission_retry_delay_seconds=20,
    )
    safe="".join(ch for ch in str(config["source_video_id"]) if ch.isalnum() or ch in "-_")[:20]
    suffix=str(handoff["handoff_id"]).replace("-","")[:10]
    slug=f"hidden-beyond-source-{safe.lower()}-{suffix}"
    submission=worker.submit_script(
        slug=slug,
        title=f"HB Source {safe} {suffix}",
        source=source,
        enable_internet=True,
        enable_gpu=False,
        is_private=True,
    )
    print("SOURCE_CPU_SUBMITTED",submission.ref,submission.version_number,flush=True)
    Path("source-submission.json").write_text(
        json.dumps({"ok":True,"ref":submission.ref,"version":submission.version_number,"handoff_id":handoff["handoff_id"]},indent=2)+"\n",
        encoding="utf-8",
    )

if __name__=="__main__":
    main()
