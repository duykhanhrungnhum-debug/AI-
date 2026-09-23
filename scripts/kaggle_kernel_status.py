#!/usr/bin/env python3
"""Inspect one Kaggle kernel without starting another GPU session."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ai_agent.core.kaggle_worker import KaggleGpuWorker


def main() -> int:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    slug = os.environ.get("KAGGLE_KERNEL_SLUG", "").strip()
    if not token or not username or not slug:
        raise RuntimeError("KAGGLE_API_TOKEN, KAGGLE_USERNAME, and KAGGLE_KERNEL_SLUG are required")

    worker = KaggleGpuWorker(api_token=token, username=username, timeout=60)
    status = worker.status(slug)
    report = {
        "kernel": f"{username}/{slug}",
        "status": status.status,
        "terminal": status.terminal,
        "successful": status.successful,
        "failure_message": status.failure_message,
    }

    try:
        metadata = worker.output_metadata(slug)
        files = metadata.get("files") if isinstance(metadata, dict) else None
        report["output_files"] = [
            str(item.get("fileName", item.get("file_name", "")))
            for item in (files or [])
            if isinstance(item, dict)
        ]
    except Exception as exc:
        report["output_metadata_error"] = str(exc)

    if status.terminal:
        try:
            logs = worker.logs(slug)
            report["log_tail"] = logs[-12000:]
        except Exception as exc:
            report["logs_error"] = str(exc)
    else:
        report["log_tail"] = ""
        report["logs_note"] = "log stream skipped while kernel is non-terminal"

    output_dir = Path(os.environ.get("OUTPUT_DIR", "kaggle-kernel-status-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("AI_AGENT_KAGGLE_KERNEL_STATUS")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
