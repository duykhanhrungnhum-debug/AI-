#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from ai_agent.core.kaggle_worker import KaggleGpuWorker


def main()->None:
    start=json.loads(Path("bench-start.json").read_text(encoding="utf-8"))
    source=Path("hidden_beyond/translation_skill_benchmark.py").read_text(encoding="utf-8")
    marker="# __BENCH_CONFIG_INJECT__"
    if marker not in source:
        raise SystemExit("benchmark marker missing")
    source=source.replace(marker,"BENCH = "+repr({
        "run_id":start["run_id"],
        "run_token":start["run_token"],
        "callback_base":start["callback_base"],
    }),1)

    worker=KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        timeout=120,
        submission_retry_attempts=3,
        submission_retry_delay_seconds=20,
    )
    slug="hidden-beyond-translation-skill-v2"
    sub=worker.submit_script(
        slug=slug,
        title="HB Translation Skill V2",
        source=source,
        enable_internet=True,
        enable_gpu=True,
        is_private=True,
    )
    Path("translation-submission.json").write_text(
        json.dumps({"ref":sub.ref,"version":sub.version_number,"slug":slug},indent=2)+"\n",
        encoding="utf-8",
    )
    print("TRANSLATION_SKILL_SUBMITTED",sub.ref,sub.version_number,flush=True)


if __name__=="__main__":
    main()
