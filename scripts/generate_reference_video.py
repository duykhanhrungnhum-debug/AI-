#!/usr/bin/env python3
"""Generate a narrated story video from a script plus one character reference image."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from ai_agent.core.episode_pipeline import MediaProductionPipeline
from ai_agent.core.kaggle_image_batch import KaggleBatchImageProvider
from ai_agent.core.kaggle_model import KaggleModelProvider
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.production_learning import HttpProductionLearningStore
from ai_agent.core.scene_planner import ScenePlanner
from ai_agent.core.tts_model import PiperTTSProvider
from ai_agent.core.video_builder import FFmpegVideoBuilder


def positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def read_script() -> str:
    direct = os.environ.get("VIDEO_SCRIPT", "").strip()
    source = os.environ.get("VIDEO_SCRIPT_FILE", "").strip()
    if direct and source:
        raise ValueError("set only one of VIDEO_SCRIPT or VIDEO_SCRIPT_FILE")
    if source:
        direct = Path(source).read_text(encoding="utf-8").strip()
    if not direct:
        raise ValueError("VIDEO_SCRIPT or VIDEO_SCRIPT_FILE is required")
    return direct


def github_oidc_token(audience: str) -> str | None:
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "").strip()
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "").strip()
    if not request_url or not request_token:
        return None
    separator = "&" if "?" in request_url else "?"
    request = Request(
        request_url + separator + "audience=" + quote(audience),
        headers={"Authorization": f"bearer {request_token}"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    value = str(payload.get("value") or "").strip()
    if not value:
        raise RuntimeError("GitHub OIDC response did not contain a token")
    return value


def learning_store():
    base_url = os.environ.get("PRODUCTION_LEARNING_API", "").strip()
    if not base_url:
        return None
    token = os.environ.get("PRODUCTION_LEARNING_BEARER_TOKEN", "").strip()
    if not token:
        token = github_oidc_token("hidden-beyond-story-processor") or ""
    if not token:
        return None
    return HttpProductionLearningStore(base_url=base_url, bearer_token=token)


def main() -> int:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    reference_path = Path(os.environ.get("VIDEO_REFERENCE_IMAGE", "").strip())
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    if not username:
        raise RuntimeError("KAGGLE_USERNAME is required")
    if not str(reference_path) or not reference_path.is_file():
        raise FileNotFoundError("VIDEO_REFERENCE_IMAGE must point to a readable image file")

    reference = reference_path.read_bytes()
    if not reference:
        raise ValueError("reference image is empty")

    script = read_script()
    output_dir = Path(os.environ.get("VIDEO_OUTPUT_DIR", "reference-video-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    worker = KaggleGpuWorker(
        api_token=token,
        username=username,
        timeout=120,
        submission_retry_attempts=5,
        submission_retry_delay_seconds=30,
    )
    planner = ScenePlanner(
        KaggleModelProvider(
            worker=worker,
            model=os.environ.get("VIDEO_PLANNER_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
            kernel_slug="ai-agent-reference-video-planner",
            poll_interval=15,
            max_poll_attempts=120,
            max_new_tokens=1800,
            temperature=0.0,
        ),
        max_scenes=positive_int("VIDEO_MAX_SCENES", 7),
    )
    image_provider = KaggleBatchImageProvider(
        worker=worker,
        model=os.environ.get(
            "VIDEO_IMAGE_MODEL",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ),
        kernel_slug="ai-agent-reference-video-images",
        poll_interval=15,
        max_poll_attempts=120,
        inference_steps=positive_int("VIDEO_IMAGE_STEPS", 16),
        guidance_scale=float(os.environ.get("VIDEO_IMAGE_GUIDANCE", "7.0")),
        identity_threshold=float(os.environ.get("VIDEO_IDENTITY_THRESHOLD", "0.40")),
    )

    voice_name = os.environ.get("PIPER_VOICE", "vi_VN-vais1000-medium").strip()
    voice_dir = Path(os.environ.get("PIPER_VOICE_DIR", "piper-voices"))
    voice_path = voice_dir / f"{voice_name}.onnx"
    if not voice_path.exists():
        raise FileNotFoundError(f"Piper voice model is missing: {voice_path}")
    speech = PiperTTSProvider(
        model_path=str(voice_path),
        binary=os.environ.get("PIPER_BINARY", "piper"),
        timeout=600,
        min_duration_seconds=0.2,
    )

    pipeline = MediaProductionPipeline(
        scene_planner=planner,
        image_provider=image_provider,
        speech_provider=speech,
        video_builder=FFmpegVideoBuilder(
            width=positive_int("VIDEO_FINAL_WIDTH", 1280),
            height=positive_int("VIDEO_FINAL_HEIGHT", 720),
            fps=positive_int("VIDEO_FINAL_FPS", 30),
            timeout=1800,
        ),
        image_width=positive_int("VIDEO_IMAGE_WIDTH", 512),
        image_height=positive_int("VIDEO_IMAGE_HEIGHT", 288),
        max_image_rounds=positive_int("VIDEO_IMAGE_ROUNDS", 2),
        learning_store=learning_store(),
    )
    result = pipeline.produce(
        script,
        output_dir,
        visual_style=os.environ.get(
            "VIDEO_VISUAL_STYLE",
            "cinematic story illustration, coherent recurring character, detailed environment",
        ).strip(),
        reference_image=reference,
    )
    if not result.media_ready:
        raise RuntimeError("reference video pipeline finished without media_ready")

    manifest = {
        "media_ready": result.media_ready,
        "scene_count": len(result.scene_plan.scenes),
        "reference_image": str(reference_path),
        "final_video": {
            "path": result.video.path,
            "duration_seconds": result.video.duration_seconds,
            "width": result.video.width,
            "height": result.video.height,
            "has_video": result.video.has_video,
            "has_audio": result.video.has_audio,
        },
        "evidence": list(result.evidence),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("AI_AGENT_REFERENCE_VIDEO_OK")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
