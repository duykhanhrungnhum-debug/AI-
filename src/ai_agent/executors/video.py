"""Simple production video tool for AIKA.

AIKA owns intent and prompt semantics. This executor only turns one prompt into
one short MP4 artifact. It deliberately does not plan scenes, add narration,
or retry failed generations; those are separate higher-level workflows.
"""
from __future__ import annotations

from hashlib import sha256
import os

from ai_agent.core.kaggle_video import (
    KaggleBatchVideoProvider,
    SceneVideoArtifact,
    SceneVideoRequest,
)
from ai_agent.core.kaggle_worker import KaggleGpuWorker


PRODUCTION_VIDEO_KERNEL = "ai-agent-video-batch"


class VideoExecutor:
    """One bounded tool call: original prompt -> one short video artifact."""

    def __init__(self, *, kernel_slug: str = PRODUCTION_VIDEO_KERNEL) -> None:
        self.kernel_slug = kernel_slug

    @staticmethod
    def seed_for(command: str) -> int:
        return int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")

    def _provider(self) -> KaggleBatchVideoProvider:
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        if not token:
            raise RuntimeError("KAGGLE_API_TOKEN is required")
        if not username:
            raise RuntimeError("KAGGLE_USERNAME is required")
        return KaggleBatchVideoProvider(
            worker=KaggleGpuWorker(
                api_token=token,
                username=username,
                timeout=45,
                submission_retry_attempts=2,
                submission_retry_delay_seconds=10,
            ),
            kernel_slug=self.kernel_slug,
            poll_interval=10,
            max_poll_attempts=90,
            inference_steps=16,
            guidance_scale=5.0,
        )

    def execute(
        self,
        command: str,
        *,
        width: int = 832,
        height: int = 480,
        num_frames: int = 17,
        fps: int = 16,
    ) -> SceneVideoArtifact:
        prompt = command.strip()
        if not prompt:
            raise ValueError("video prompt is required")
        request = SceneVideoRequest(
            scene_id="video-01",
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            seed=self.seed_for(prompt),
        )
        batch = self._provider().generate_batch((request,))
        if len(batch.scenes) != 1:
            raise RuntimeError("video tool did not return exactly one clip")
        scene = batch.scenes[0]
        if not scene.verified:
            detail = "; ".join(scene.issues) or "unknown verification failure"
            raise RuntimeError("video verification failed: " + detail)
        return scene.artifact
