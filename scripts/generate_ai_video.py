#!/usr/bin/env python3
"""Generate a complete narrated motion video with AI-'s own on-demand Kaggle stack."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ai_agent.core.generative_video_pipeline import GenerativeVideoPipeline
from ai_agent.core.kaggle_model import KaggleModelProvider
from ai_agent.core.kaggle_video import KaggleBatchVideoProvider
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.scene_planner import ScenePlanner
from ai_agent.core.tts_model import PiperTTSProvider
from ai_agent.core.video_builder import FFmpegVideoBuilder


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def read_script() -> str:
    direct = os.environ.get("VIDEO_SCRIPT", "").strip()
    path = os.environ.get("VIDEO_SCRIPT_FILE", "").strip()
    if direct and path:
        raise ValueError("set only one of VIDEO_SCRIPT or VIDEO_SCRIPT_FILE")
    if path:
        direct = Path(path).read_text(encoding="utf-8").strip()
    if not direct:
        raise ValueError("VIDEO_SCRIPT or VIDEO_SCRIPT_FILE is required")
    return direct


def main() -> int:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    if not username:
        raise RuntimeError("KAGGLE_USERNAME is required")

    script = read_script()
    output_dir = Path(os.environ.get("VIDEO_OUTPUT_DIR", "ai-video-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_scenes = env_int("VIDEO_MAX_SCENES", 7)
    clip_width = env_int("VIDEO_CLIP_WIDTH", 480)
    clip_height = env_int("VIDEO_CLIP_HEIGHT", 832)
    clip_frames = env_int("VIDEO_CLIP_FRAMES", 33)
    clip_fps = env_int("VIDEO_CLIP_FPS", 16)
    inference_steps = env_int("VIDEO_INFERENCE_STEPS", 10)
    final_width = env_int("VIDEO_FINAL_WIDTH", 720)
    final_height = env_int("VIDEO_FINAL_HEIGHT", 1280)
    composition = os.environ.get("VIDEO_COMPOSITION", "9:16 vertical").strip()
    visual_style = os.environ.get(
        "VIDEO_VISUAL_STYLE",
        "photorealistic cinematic Vietnamese short film, natural acting, consistent characters",
    ).strip()
    if not composition or not visual_style:
        raise ValueError("VIDEO_COMPOSITION and VIDEO_VISUAL_STYLE must not be empty")

    worker = KaggleGpuWorker(
        api_token=token,
        username=username,
        timeout=120,
        submission_retry_attempts=5,
        submission_retry_delay_seconds=30,
    )

    planner_model = KaggleModelProvider(
        worker=worker,
        model=os.environ.get("VIDEO_PLANNER_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        kernel_slug="ai-agent-video-scene-planner",
        poll_interval=15,
        max_poll_attempts=120,
        max_new_tokens=1800,
        temperature=0.0,
    )
    scene_planner = ScenePlanner(planner_model, max_scenes=max_scenes)

    video_provider = KaggleBatchVideoProvider(
        worker=worker,
        model=os.environ.get(
            "VIDEO_GENERATION_MODEL",
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ),
        kernel_slug="ai-agent-video-generator",
        poll_interval=20,
        max_poll_attempts=180,
        inference_steps=inference_steps,
        guidance_scale=5.0,
        min_video_bytes=20_000,
    )

    voice_name = os.environ.get("PIPER_VOICE", "vi_VN-vais1000-medium").strip()
    voice_dir = Path(os.environ.get("PIPER_VOICE_DIR", "piper-voices"))
    voice_path = voice_dir / f"{voice_name}.onnx"
    if not voice_path.exists():
        raise FileNotFoundError(f"Piper voice model is missing: {voice_path}")
    speech_provider = PiperTTSProvider(
        model_path=str(voice_path),
        binary=os.environ.get("PIPER_BINARY", "piper"),
        timeout=600,
        min_duration_seconds=0.2,
    )

    video_builder = FFmpegVideoBuilder(
        width=final_width,
        height=final_height,
        fps=30,
        timeout=1800,
    )
    pipeline = GenerativeVideoPipeline(
        scene_planner=scene_planner,
        video_provider=video_provider,
        speech_provider=speech_provider,
        video_builder=video_builder,
        width=clip_width,
        height=clip_height,
        num_frames=clip_frames,
        fps=clip_fps,
        composition=composition,
        max_video_rounds=2,
    )

    result = pipeline.produce(
        script,
        output_dir,
        visual_style=visual_style,
    )
    if not result.media_ready:
        raise RuntimeError("AI generative video pipeline finished without media_ready")

    manifest = {
        "media_ready": result.media_ready,
        "script": script,
        "composition": composition,
        "visual_style": visual_style,
        "scene_count": len(result.scene_plan.scenes),
        "video_model": result.clips.scenes[0].artifact.model if result.clips.scenes else None,
        "video_rounds": result.clips.rounds,
        "audio": {
            "provider": result.audio.provider,
            "model": result.audio.model,
            "duration_seconds": result.audio.duration_seconds,
            "quality_passed": result.audio_quality.passed,
        },
        "final_video": {
            "path": result.video.path,
            "duration_seconds": result.video.duration_seconds,
            "width": result.video.width,
            "height": result.video.height,
            "has_video": result.video.has_video,
            "has_audio": result.video.has_audio,
        },
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "narration": scene.narration,
                "prompt": scene.image_prompt,
            }
            for scene in result.scene_plan.scenes
        ],
        "evidence": list(result.evidence),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("AI_AGENT_COMPLETE_VIDEO_OK")
    print(json.dumps(manifest["final_video"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
