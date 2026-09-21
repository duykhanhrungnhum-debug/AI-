#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
from ai_agent.core.kaggle_worker import KaggleGpuWorker

def main():
    start=json.loads(Path("bench-start.json").read_text(encoding="utf-8"))
    source=Path("hidden_beyond/professional_translation_benchmark.py").read_text(encoding="utf-8")
    marker="# __BENCH_CONFIG_INJECT__"
    if marker not in source:
        raise SystemExit("benchmark marker missing")
    source=source.replace(marker,"BENCH = "+repr({
        "run_id":start["run_id"],
        "run_token":start["run_token"],
        "callback_base":start["callback_base"],
    }),1)
    suffix=str(start["run_id"]).replace("-","")[:10]
    worker=KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        timeout=120,
        submission_retry_attempts=3,
        submission_retry_delay_seconds=20,
    )
    sub=worker.submit_script(
        slug=f"hb-pro-translation-{suffix}",
        title=f"HB Pro Translation {suffix}",
        source=source,
        enable_internet=True,
        enable_gpu=True,
        is_private=True,
    )
    Path("professional-translation-submission.json").write_text(
        json.dumps({"ref":sub.ref,"version":sub.version_number},indent=2)+"\n",
        encoding="utf-8",
    )
    print("PROFESSIONAL_TRANSLATION_SUBMITTED",sub.ref,sub.version_number,flush=True)

if __name__=="__main__":
    main()
