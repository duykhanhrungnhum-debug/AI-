"""Minimal Kaggle GPU worker smoke test.

This script intentionally does not call any external AI service. It only proves
that the Kaggle worker received a real NVIDIA GPU and writes auditable evidence
to /kaggle/working for GitHub Actions to download.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


OUTPUT = Path("/kaggle/working/gpu_report.json")


def query_gpu() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    report: dict[str, object] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "gpu_available": completed.returncode == 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_returncode": completed.returncode,
    }

    if completed.returncode != 0:
        report["error"] = (completed.stderr or completed.stdout or "nvidia-smi failed").strip()
        return report

    line = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        report["gpu_available"] = False
        report["error"] = f"unexpected nvidia-smi output: {line}"
        return report

    report["gpu_name"] = parts[0]
    try:
        report["memory_total_mib"] = int(parts[1])
    except ValueError:
        report["memory_total_mib"] = parts[1]
    report["driver_version"] = parts[2]
    return report


def main() -> int:
    report = query_gpu()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("gpu_available") is True else 1


if __name__ == "__main__":
    sys.exit(main())
