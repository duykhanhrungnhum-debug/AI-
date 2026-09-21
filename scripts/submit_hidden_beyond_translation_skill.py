#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path

from ai_agent.core.kaggle_worker import KaggleGpuWorker


def main()->None:
    source=Path("hidden_beyond/translation_skill_benchmark.py").read_text(encoding="utf-8")
    worker=KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        timeout=120,
        submission_retry_attempts=3,
        submission_retry_delay_seconds=20,
    )
    slug="hidden-beyond-translation-skill"
    sub=worker.submit_script(
        slug=slug,
        title="HB Translation Skill",
        source=source,
        enable_internet=True,
        enable_gpu=True,
        is_private=True,
    )
    print("TRANSLATION_SKILL_SUBMITTED",sub.ref,sub.version_number,flush=True)

    deadline=time.time()+20*60
    while time.time()<deadline:
        status=worker.status(slug)
        print("TRANSLATION_SKILL_STATUS",status.status,status.failure_message,flush=True)
        if status.terminal:
            if not status.successful:
                try:
                    print(worker.logs(slug),flush=True)
                except Exception as exc:
                    print("TRANSLATION_SKILL_LOG_ERROR",repr(exc),flush=True)
                raise RuntimeError(f"translation skill benchmark failed: {status.status} {status.failure_message}")
            try:
                logs=worker.logs(slug)
                print(logs,flush=True)
                if "TRANSLATION_SKILL_VERIFIED" not in logs:
                    print("TRANSLATION_SKILL_LOG_MARKER_MISSING",flush=True)
            except Exception as exc:
                print("TRANSLATION_SKILL_LOG_READ_SKIPPED",repr(exc),flush=True)
            print("TRANSLATION_SKILL_BENCHMARK_SUCCESS",flush=True)
            return
        time.sleep(20)
    raise TimeoutError("translation skill benchmark exceeded 20 minutes")


if __name__=="__main__":
    main()
