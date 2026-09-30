"""Image jobs launched directly from the AIKA web chat.

This module is intentionally thin: chat routing does not reimplement image
creation. It hands the user's natural-language command to the existing V2
pipeline and stores the resulting PNG until the browser retrieves it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
import threading
import time
import unicodedata
from uuid import uuid4

from .core.kaggle_worker import KaggleGpuWorker
from .core.media_v2 import ImageRequestV2, KaggleImageV2Provider


_IMAGE_COMMANDS = (
    "tao anh",
    "tao mot anh",
    "tao hinh",
    "ve anh",
    "ve mot anh",
    "ve ",
    "lam anh",
    "sinh anh",
    "generate image",
    "generate an image",
    "create image",
    "create an image",
    "draw image",
    "draw an image",
    "make an image",
)


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).strip()


def is_image_request(message: str) -> bool:
    """Return True only for explicit image-creation commands.

    This keeps questions *about* images in normal chat while routing commands
    such as "AIKA tạo ảnh ..." or "vẽ ..." to the image skill.
    """
    folded = " ".join(_fold(message).split())
    if not folded:
        return False
    if folded.startswith("aika "):
        folded = folded[5:].lstrip()
    return any(folded.startswith(prefix) or f" {prefix}" in folded for prefix in _IMAGE_COMMANDS)


@dataclass
class ImageChatJob:
    job_id: str
    command: str
    status: str = "pending"
    text: str = ""
    provider: str = ""
    model: str = ""
    prompt: str = ""
    error: str = ""
    image: bytes = b""
    mime_type: str = "image/png"
    elapsed_seconds: float = 0.0
    created_at: float = 0.0


class ImageChatBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._jobs: dict[str, ImageChatJob] = {}

    def create_job(self, command: str) -> ImageChatJob:
        command = command.strip()
        if not command:
            raise ValueError("message is required")
        job = ImageChatJob(job_id=uuid4().hex, command=command, created_at=time.time())
        with self._lock:
            self._jobs[job.job_id] = job
            if len(self._jobs) > 30:
                oldest = sorted(self._jobs.values(), key=lambda item: item.created_at)[:-20]
                for item in oldest:
                    self._jobs.pop(item.job_id, None)
        threading.Thread(
            target=self._run_job,
            args=(job.job_id,),
            name=f"aika-image-{job.job_id[:8]}",
            daemon=True,
        ).start()
        return job

    def _run_job(self, job_id: str) -> None:
        try:
            with self._generation_lock:
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = "processing"
                token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
                username = os.environ.get("KAGGLE_USERNAME", "").strip()
                if not token:
                    raise RuntimeError("KAGGLE_API_TOKEN is required")
                if not username:
                    raise RuntimeError("KAGGLE_USERNAME is required")

                with self._lock:
                    command = self._jobs[job_id].command
                seed = int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")
                provider = KaggleImageV2Provider(
                    worker=KaggleGpuWorker(
                        api_token=token,
                        username=username,
                        timeout=120,
                        submission_retry_attempts=5,
                        submission_retry_delay_seconds=30,
                    ),
                    kernel_slug=os.environ.get("MEDIA_V2_KERNEL_SLUG", "ai-agent-image-v2").strip(),
                    poll_interval=3,
                    max_poll_attempts=int(os.environ.get("MEDIA_V2_MAX_POLL_ATTEMPTS", "600")),
                )
                result = provider.generate_many((ImageRequestV2(
                    item_id="chat-image",
                    command=command,
                    seed=seed,
                    width=1024,
                    height=1024,
                ),))[0]
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = "done"
                    job.text = "AIKA đã tạo ảnh xong."
                    job.provider = "kaggle-image-v2"
                    job.model = result.model
                    job.prompt = result.prompt
                    job.image = result.data
                    job.elapsed_seconds = result.elapsed_seconds
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = "error"
                    job.error = f"{type(exc).__name__}: {exc}"[:2000]

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                "job_id": job.job_id,
                "kind": "image",
                "status": job.status,
                "text": job.text,
                "provider": job.provider,
                "model": job.model,
                "prompt": job.prompt,
                "error": job.error,
                "elapsed_seconds": job.elapsed_seconds,
                "worker_state": "running" if job.status == "processing" else job.status,
                "image_url": f"/v1/chat/image?job_id={job.job_id}" if job.status == "done" else "",
            }

    def get_image(self, job_id: str) -> tuple[bytes, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != "done" or not job.image:
                raise ValueError("image is not ready")
            return job.image, job.mime_type


IMAGE_CHAT_BROKER = ImageChatBroker()
