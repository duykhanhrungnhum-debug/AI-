#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from ai_agent.core.kaggle_video import KaggleBatchVideoProvider, SceneVideoRequest
from ai_agent.core.kaggle_worker import KaggleGpuWorker


def main() -> int:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    if not username:
        raise RuntimeError("KAGGLE_USERNAME is required")

    output_dir = Path(os.environ.get("OUTPUT_DIR", "kaggle-video-smoke-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    worker = KaggleGpuWorker(
        api_token=token,
        username=username,
        timeout=120,
        submission_retry_attempts=5,
        submission_retry_delay_seconds=30,
    )
    provider = KaggleBatchVideoProvider(
        worker=worker,
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        kernel_slug="ai-agent-video-smoke",
        poll_interval=20,
        max_poll_attempts=180,
        inference_steps=4,
        guidance_scale=5.0,
        min_video_bytes=5_000,
    )
    request = SceneVideoRequest(
        scene_id="smoke-001",
        prompt=(
            "A cinematic realistic Vietnamese roadside food stall at dusk. "
            "A paper lantern sways gently in the breeze while steam rises from a fryer. "
            "Natural motion, stable composition, subtle handheld camera movement, no text, no logo."
        ),
        negative_prompt=(
            "subtitles, text, watermark, logo, static image, blurry, distorted objects, "
            "deformed anatomy, duplicate objects, low quality"
        ),
        width=480,
        height=272,
        num_frames=9,
        fps=8,
        seed=20260923,
    )
    result = provider.generate_with_retries([request], max_rounds=1)
    if not result.verified:
        raise RuntimeError(f"video smoke verification failed: {result.failed_scene_ids}")

    scene = result.scenes[0]
    video_path = output_dir / "smoke.mp4"
    video_path.write_bytes(scene.artifact.data)
    report = {
        "verified": scene.verified,
        "issues": scene.issues,
        "provider": scene.artifact.provider,
        "model": scene.artifact.model,
        "duration_seconds": scene.artifact.duration_seconds,
        "width": scene.artifact.width,
        "height": scene.artifact.height,
        "fps": scene.artifact.fps,
        "evidence": scene.artifact.evidence,
        "rounds": result.rounds,
        "video_path": str(video_path),
        "video_size_bytes": video_path.stat().st_size,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("AI_AGENT_GENERATIVE_VIDEO_SMOKE_OK")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
