"""Minimal asynchronous media-tool broker for AIKA.

AIKA decides which tool to call. This broker never classifies user intent and
never knows which GPU, model, provider, or orchestration strategy a tool uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from uuid import uuid4

from .executors.image import ImageExecutor
from .executors.video import VideoExecutor


@dataclass
class MediaJob:
    job_id: str
    kind: str
    command: str
    status: str = "pending"
    text: str = ""
    provider: str = ""
    model: str = ""
    error: str = ""
    created_at: float = 0.0
    artifacts: list[bytes] = field(default_factory=list)
    mimes: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    per_artifact_seconds: list[float] = field(default_factory=list)


class MediaToolBroker:
    """One broker, two stable tool calls: image.generate() and video.generate()."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, MediaJob] = {}
        self._image = ImageExecutor()
        self._video = VideoExecutor()

    def create_job(self, kind: str, command: str, *, job_id: str | None = None) -> MediaJob:
        kind = kind.strip().casefold()
        command = command.strip()
        if kind not in {"image", "video"}:
            raise ValueError("media kind must be image or video")
        if not command:
            raise ValueError("media command is required")
        resolved_id = (job_id or uuid4().hex).strip()
        if not resolved_id:
            raise ValueError("job_id must not be empty")
        job = MediaJob(
            job_id=resolved_id,
            kind=kind,
            command=command,
            status="processing",
            created_at=time.time(),
        )
        with self._lock:
            if resolved_id in self._jobs:
                raise ValueError(f"media job already exists: {resolved_id}")
            self._jobs[resolved_id] = job
            if len(self._jobs) > 40:
                oldest = sorted(self._jobs.values(), key=lambda item: item.created_at)[:-30]
                for item in oldest:
                    self._jobs.pop(item.job_id, None)
        target = self._run_image if kind == "image" else self._run_video
        threading.Thread(
            target=target,
            args=(job.job_id,),
            name=f"aika-{kind}-tool-{job.job_id[:8]}",
            daemon=True,
        ).start()
        return job

    def _command(self, job_id: str) -> str:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.command

    def _fail(self, job_id: str, exc: Exception) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"[:2000]

    def _run_image(self, job_id: str) -> None:
        started = time.perf_counter()
        try:
            command = self._command(job_id)
            results = self._image.generate(
                command,
                width=1024,
                height=1024,
                max_images=6,
            )
            if not results:
                raise RuntimeError("image tool returned no artifacts")
            with self._lock:
                job = self._jobs[job_id]
                job.status = "done"
                job.text = (
                    "AIKA đã tạo ảnh xong."
                    if len(results) == 1
                    else f"AIKA đã tạo xong {len(results)} ảnh riêng."
                )
                job.provider = "image-tool"
                job.model = results[0].model
                job.artifacts = [result.data for result in results]
                job.mimes = ["image/png" for _ in results]
                job.prompts = [result.prompt for result in results]
                job.per_artifact_seconds = [result.elapsed_seconds for result in results]
                job.elapsed_seconds = time.perf_counter() - started
                job.error = ""
        except Exception as exc:
            self._fail(job_id, exc)

    def _run_video(self, job_id: str) -> None:
        started = time.perf_counter()
        try:
            command = self._command(job_id)
            artifact = self._video.generate(command)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "done"
                job.text = "AIKA đã tạo video xong."
                job.provider = "video-tool"
                job.model = artifact.model
                job.artifacts = [artifact.data]
                job.mimes = [artifact.mime_type]
                job.prompts = [command]
                job.elapsed_seconds = time.perf_counter() - started
                job.per_artifact_seconds = [job.elapsed_seconds]
                job.error = ""
        except Exception as exc:
            self._fail(job_id, exc)

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            ready = job.status == "done"
            image_urls: list[str] = []
            video_url = ""
            if ready and job.kind == "image":
                image_urls = [
                    f"/v1/chat/image?job_id={job.job_id}&index={index}"
                    for index in range(len(job.artifacts))
                ]
            elif ready and job.kind == "video" and job.artifacts:
                video_url = f"/v1/chat/video?job_id={job.job_id}"
            artifact_urls = image_urls or ([video_url] if video_url else [])
            return {
                "job_id": job.job_id,
                "kind": job.kind,
                "status": job.status,
                "text": job.text,
                "provider": job.provider,
                "model": job.model,
                "error": job.error,
                "worker_state": job.status,
                "worker_error": job.error,
                "has_image": bool(image_urls),
                "image_count": len(image_urls),
                "image_urls": image_urls,
                "image_url": image_urls[0] if image_urls else "",
                "has_video": bool(video_url),
                "video_url": video_url,
                "artifact_urls": artifact_urls,
                "generation_prompt": job.prompts[0] if job.prompts else "",
                "generation_prompts": list(job.prompts),
                "elapsed_seconds": job.elapsed_seconds,
                "per_artifact_seconds": list(job.per_artifact_seconds),
            }

    def get_image(self, job_id: str, *, index: int = 0) -> tuple[bytes, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.kind != "image":
                raise ValueError("job is not an image job")
            if job.status != "done":
                raise ValueError("image is not ready")
            if index < 0 or index >= len(job.artifacts):
                raise IndexError(index)
            return job.artifacts[index], job.mimes[index]

    def get_video(self, job_id: str) -> tuple[bytes, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.kind != "video":
                raise ValueError("job is not a video job")
            if job.status != "done" or not job.artifacts:
                raise ValueError("video is not ready")
            return job.artifacts[0], job.mimes[0]


MEDIA_BROKER = MediaToolBroker()