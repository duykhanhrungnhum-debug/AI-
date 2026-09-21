#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from ai_agent.core.kaggle_worker import KaggleGpuWorker


def post_fail(callback_base: str, job_id: str, token: str, message: str) -> None:
    try:
        payload=json.dumps({"job_id":job_id,"error":message}).encode()
        req=Request(
            callback_base.rstrip("/")+"/fail",
            data=payload,
            headers={"content-type":"application/json","x-job-token":token},
            method="POST",
        )
        with urlopen(req,timeout=60) as r:
            print("FAIL_REPORTED",r.status,r.read().decode(),flush=True)
    except Exception as exc:
        print("FAIL_REPORT_ERROR",repr(exc),flush=True)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",required=True)
    args=ap.parse_args()

    job=json.loads(Path(args.start).read_text(encoding="utf-8"))
    required=("job_id","job_token","callback_base")
    missing=[k for k in required if not job.get(k)]
    if missing:
        raise SystemExit(f"Missing job fields: {missing}")

    template=Path("hidden_beyond/longform_audio_worker.py").read_text(encoding="utf-8")
    marker="# __JOB_CONFIG_INJECT__"
    if marker not in template:
        raise SystemExit("Worker config marker missing")
    source=template.replace(marker,"JOB = "+repr({
        "job_id":job["job_id"],
        "job_token":job["job_token"],
        "callback_base":job["callback_base"],
    }),1)

    slug=f"hidden-beyond-ai-{str(job['job_id']).replace('-','')[:12]}"
    worker=KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        timeout=120,
        submission_retry_attempts=3,
        submission_retry_delay_seconds=20,
    )
    mode="gpu"
    s=None
    last_exc=None
    for attempt in range(1,5):
        try:
            s=worker.submit_script(
                slug=slug,
                title=f"HB AI {str(job['job_id'])[:8]}",
                source=source,
                enable_internet=True,
                enable_gpu=True,
                is_private=True,
            )
            print("HB_GPU_SUBMITTED",s.ref,s.version_number,flush=True)
            break
        except Exception as gpu_exc:
            last_exc=gpu_exc
            detail=str(gpu_exc).casefold()
            capacity_limited=(
                "maximum batch gpu session count" in detail
                or ("gpu session" in detail and "reached" in detail)
                or ("capacity" in detail and "gpu" in detail)
            )
            if not capacity_limited:
                message=f"GPU submit failed: {gpu_exc!r}"
                post_fail(job["callback_base"],job["job_id"],job["job_token"],message)
                raise
            if attempt<4:
                wait=90*attempt
                print("HB_GPU_CAPACITY_RETRY",attempt,f"wait={wait}s",repr(gpu_exc),flush=True)
                import time
                time.sleep(wait)
    if s is None:
        message=f"GPU required for quality translation/voice; capacity unavailable after retries: {last_exc!r}"
        post_fail(job["callback_base"],job["job_id"],job["job_token"],message)
        raise RuntimeError(message)

    Path("gpu-submission.json").write_text(
        json.dumps({"ok":True,"mode":mode,"slug":slug,"ref":s.ref,"version":s.version_number},indent=2)+"\n",
        encoding="utf-8",
    )


if __name__=="__main__":
    main()
