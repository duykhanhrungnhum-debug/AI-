#!/usr/bin/env python3
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from ai_agent.core.kaggle_image_batch import KaggleBatchImageProvider, SceneImageRequest
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.video_builder import FFmpegVideoBuilder


API = os.environ.get(
    "STORY_PROCESSOR_API",
    "https://rlqqcuuphjmwksanbfml.supabase.co/functions/v1/story-processor-api",
).rstrip("/")
SOURCE_KEY = os.environ.get("STORY_SOURCE_KEY", "").strip()
AUDIENCE = "hidden-beyond-story-processor"
REPAIR_SIGNATURE = "kaggle-sd15-ai-agent-v1"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "hidden-beyond-story-visual-output"))


def http_json(url: str, *, method: str = "GET", headers: dict[str, str] | None = None,
              payload: dict | None = None, timeout: float = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:4000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return parsed


def http_bytes(url: str, *, timeout: float = 300) -> bytes:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"download failed HTTP {exc.code}: {detail}") from exc


def upload_signed(url: str, path: Path, content_type: str) -> None:
    body = path.read_bytes()
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Content-Type": content_type,
            "Cache-Control": "max-age=3600",
            "x-upsert": "true",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"signed upload failed HTTP {exc.code}: {detail}") from exc


def get_oidc() -> str:
    request_url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    separator = "&" if "?" in request_url else "?"
    url = request_url + separator + "audience=" + urllib.parse.quote(AUDIENCE)
    token = os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
    data = http_json(
        url,
        headers={"Authorization": f"bearer {token}"},
        timeout=30,
    )
    value = str(data.get("value") or "").strip()
    if not value:
        raise RuntimeError("GitHub OIDC response did not contain a token")
    return value


def api_post(route: str, payload: dict) -> dict:
    token = get_oidc()
    return http_json(
        f"{API}/{route}",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=120,
    )


def stable_seed(episode_id: int, scene_no: int) -> int:
    digest = sha256(f"{episode_id}:{scene_no}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def media_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "ffprobe failed").strip())
    data = json.loads(completed.stdout)
    try:
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe did not return media duration") from exc


def record_failure(episode_id: int, error: BaseException) -> None:
    try:
        response = api_post(
            "visual-fail",
            {
                "episode_id": episode_id,
                "error": str(error)[:2000],
                "repair_signature": REPAIR_SIGNATURE,
            },
        )
        print("VISUAL_FAILURE_RECORDED", json.dumps(response, ensure_ascii=False))
    except Exception as report_error:
        print(f"Could not record visual failure: {report_error}", file=sys.stderr)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("KAGGLE_API_TOKEN", "").strip():
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    if not os.environ.get("KAGGLE_USERNAME", "").strip():
        raise RuntimeError("KAGGLE_USERNAME is required")

    claim_payload = {"repair_signature": REPAIR_SIGNATURE}
    if SOURCE_KEY:
        claim_payload["source_key"] = SOURCE_KEY
    claim = api_post("visual-claim", claim_payload)
    (OUTPUT_DIR / "claim.json").write_text(
        json.dumps(claim, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if claim.get("stage") == "idle":
        print("HIDDEN_BEYOND_STORY_VISUAL_IDLE")
        print(json.dumps(claim, ensure_ascii=False))
        return 0
    if claim.get("stage") != "visual_claimed":
        raise RuntimeError(f"Unexpected visual claim response: {claim}")

    job = claim["job"]
    episode_id = int(job["episode_id"])
    assets = list(job.get("assets") or [])
    if not assets:
        error = RuntimeError("visual claim returned no assets")
        record_failure(episode_id, error)
        raise error

    try:
        worker = KaggleGpuWorker(
            api_token=os.environ["KAGGLE_API_TOKEN"],
            username=os.environ["KAGGLE_USERNAME"],
            timeout=120,
        )
        image_provider = KaggleBatchImageProvider(
            worker=worker,
            model="stable-diffusion-v1-5/stable-diffusion-v1-5",
            kernel_slug=f"hidden-beyond-story-visual-{episode_id}",
            poll_interval=15,
            max_poll_attempts=120,
            inference_steps=12,
            guidance_scale=7.0,
        )

        requests: list[SceneImageRequest] = []
        for asset in assets:
            scene_no = int(asset["scene_no"])
            prompt_vi = str(asset.get("prompt_vi") or "").strip()
            excerpt = str(asset.get("narration_excerpt_vi") or "").strip()
            if not prompt_vi:
                raise RuntimeError(f"missing visual prompt for scene {scene_no}")
            prompt = (
                "Cinematic historical Chinese story illustration, coherent recurring characters, "
                "dramatic realistic lighting, detailed ancient environment, 16:9 widescreen composition, "
                "no text, no logo, no watermark. "
                f"Scene {scene_no}: {prompt_vi}. Narrative context: {excerpt}"
            )
            requests.append(
                SceneImageRequest(
                    scene_id=f"scene-{scene_no:03d}",
                    prompt=prompt,
                    negative_prompt=(
                        "text, subtitles, logo, watermark, modern objects, blurry, low quality, "
                        "deformed anatomy, duplicate people, gore"
                    ),
                    width=512,
                    height=288,
                    seed=stable_seed(episode_id, scene_no),
                )
            )

        batch = image_provider.generate_with_retries(requests, max_rounds=2)
        if not batch.verified:
            raise RuntimeError(
                "Kaggle image verification failed for scenes: "
                + ", ".join(batch.failed_scene_ids)
            )

        work = Path(tempfile.mkdtemp(prefix=f"hidden-beyond-{episode_id}-"))
        image_paths: list[Path] = []
        result_by_scene = {scene.scene_id: scene for scene in batch.scenes}
        completed_assets: list[dict] = []
        image_evidence: dict[str, list[str]] = {}

        for asset in assets:
            scene_no = int(asset["scene_no"])
            scene_id = f"scene-{scene_no:03d}"
            scene = result_by_scene[scene_id]
            path = work / f"{scene_id}.png"
            path.write_bytes(scene.artifact.data)
            if path.stat().st_size < 20_000:
                raise RuntimeError(f"generated image too small for scene {scene_no}")
            image_paths.append(path)
            completed_assets.append({"scene_no": scene_no, "path": asset["path"]})
            image_evidence[scene_id] = list(scene.artifact.evidence)

        narration = work / "narration.wav"
        narration.write_bytes(http_bytes(job["narration"]["signedUrl"]))
        if narration.stat().st_size < 10_000:
            raise RuntimeError("downloaded narration is too small")
        audio_duration = media_duration(narration)
        if audio_duration < 20:
            raise RuntimeError(f"narration is unexpectedly short: {audio_duration:.3f}s")

        final_video = work / "final.mp4"
        video = FFmpegVideoBuilder(
            width=1280,
            height=720,
            fps=30,
            timeout=900,
        ).build(
            [str(path) for path in image_paths],
            str(narration),
            str(final_video),
        )
        if final_video.stat().st_size < 1_000_000:
            raise RuntimeError("final video is unexpectedly small")
        duration_match = abs(video.duration_seconds - audio_duration) <= 3.0
        if not duration_match:
            raise RuntimeError(
                "final duration mismatch: "
                f"video={video.duration_seconds:.3f}s audio={audio_duration:.3f}s"
            )

        for asset, path in zip(assets, image_paths, strict=True):
            upload_signed(asset["signedUrl"], path, "image/png")
        upload_signed(job["final_video"]["signedUrl"], final_video, "video/mp4")

        verification = {
            "images_valid": batch.verified,
            "audio_valid": True,
            "video_valid": video.has_video and video.has_audio,
            "duration_match": duration_match,
            "resolution_ok": video.width == 1280 and video.height == 720,
            "audio_duration_seconds": round(audio_duration, 3),
            "video_duration_seconds": round(video.duration_seconds, 3),
            "final_size_bytes": final_video.stat().st_size,
        }
        completed = api_post(
            "visual-complete",
            {
                "episode_id": episode_id,
                "repair_signature": REPAIR_SIGNATURE,
                "assets": completed_assets,
                "final_video_path": job["final_video"]["path"],
                "verification": verification,
            },
        )
        if completed.get("stage") not in {"technical_video_ready", "verified_publish_ready"}:
            raise RuntimeError(f"visual-complete did not accept episode: {completed}")

        evidence = {
            "stage": completed.get("stage"),
            "episode_id": episode_id,
            "series_title": job.get("series_title"),
            "episode_no": job.get("episode_no"),
            "repair_signature": REPAIR_SIGNATURE,
            "image_rounds": batch.rounds,
            "image_count": len(image_paths),
            "image_evidence": image_evidence,
            "verification": verification,
            "video_evidence": list(video.evidence),
            "complete_response": completed,
        }
        (OUTPUT_DIR / "result.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("HIDDEN_BEYOND_STORY_VISUAL_OK")
        print(json.dumps(evidence, ensure_ascii=False))
        return 0
    except BaseException as exc:
        record_failure(episode_id, exc)
        failure = {
            "stage": "failed",
            "episode_id": episode_id,
            "repair_signature": REPAIR_SIGNATURE,
            "error": str(exc),
        }
        (OUTPUT_DIR / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
