#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from ai_agent.executors.video import VideoExecutor


def main() -> int:
    if not os.environ.get("KAGGLE_API_TOKEN", "").strip():
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    if not os.environ.get("KAGGLE_USERNAME", "").strip():
        raise RuntimeError("KAGGLE_USERNAME is required")

    output_dir = Path(os.environ.get("OUTPUT_DIR", "kaggle-video-smoke-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Four denoising steps produced an MP4 that was still raw noise. Wan's
    # official pipeline defaults to 50; 20 is the smallest production-oriented
    # quality gate we use here while keeping this tiny smoke inexpensive.
    executor = VideoExecutor(
        kernel_slug="ai-agent-video-smoke",
        inference_steps=20,
        poll_interval=5,
        max_poll_attempts=180,
    )
    prompt = (
        "A cinematic realistic Vietnamese roadside food stall at dusk. "
        "A paper lantern sways gently in the breeze while steam rises from a fryer."
    )
    artifact = executor.execute(
        prompt,
        width=480,
        height=272,
        num_frames=9,
        fps=8,
    )

    video_path = output_dir / "smoke.mp4"
    video_path.write_bytes(artifact.data)
    report = {
        "verified": True,
        "provider": artifact.provider,
        "model": artifact.model,
        "duration_seconds": artifact.duration_seconds,
        "width": artifact.width,
        "height": artifact.height,
        "fps": artifact.fps,
        "generation_seconds": artifact.elapsed_seconds,
        "evidence": artifact.evidence,
        "video_path": str(video_path),
        "video_size_bytes": video_path.stat().st_size,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("AIKA_VIDEO_TOOL_SMOKE_OK")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
