#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from ai_agent.executors.video import VideoExecutor


MIN_SMOKE_DURATION_SECONDS = 4.0


def main() -> int:
    if not os.environ.get("KAGGLE_API_TOKEN", "").strip():
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    if not os.environ.get("KAGGLE_USERNAME", "").strip():
        raise RuntimeError("KAGGLE_USERNAME is required")

    output_dir = Path(os.environ.get("OUTPUT_DIR", "kaggle-video-smoke-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    executor = VideoExecutor(
        kernel_slug="ai-agent-video-smoke",
        inference_steps=40,
        poll_interval=5,
        max_poll_attempts=180,
    )
    prompt = (
        "A realistic Vietnamese roadside food stall at dusk. Keep the camera locked and stable. "
        "A paper lantern under the awning visibly swings left and right in the breeze. "
        "A metal fryer is clearly visible below it and a continuous plume of steam rises upward. "
        "Keep the food stall, lantern, fryer, and steam visible for the whole shot."
    )
    artifact = executor.execute(
        prompt,
        width=480,
        height=272,
        num_frames=33,
        fps=8,
    )

    if artifact.duration_seconds < MIN_SMOKE_DURATION_SECONDS:
        raise RuntimeError(
            f"video smoke is too short: {artifact.duration_seconds:.3f}s < "
            f"{MIN_SMOKE_DURATION_SECONDS:.1f}s"
        )

    video_path = output_dir / "smoke.mp4"
    video_path.write_bytes(artifact.data)
    report = {
        "verified": False,
        "technical_verified": True,
        "quality_verified": False,
        "qa_required": "visual_semantic_motion",
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
    print("AIKA_VIDEO_TOOL_SMOKE_TECHNICAL_OK_PENDING_VISUAL_QA")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
