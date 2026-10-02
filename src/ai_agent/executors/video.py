"""Simple production video tool for AIKA.

AIKA chooses the tool. This executor performs exactly one bounded video
creation call and returns one MP4 artifact.
"""
from __future__ import annotations

from hashlib import sha256
import os

from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.simple_video_tool import SimpleKaggleVideoTool, SimpleVideoArtifact


PRODUCTION_VIDEO_KERNEL = "ai-agent-video-tool"
GENERIC_VIDEO_CONSTRAINTS = (
    "Keep every named subject visually distinct and anatomically coherent for the whole shot. "
    "Preserve exact subject identity and count. Do not merge, hybridize, morph, substitute, duplicate, "
    "or omit subjects. Use one continuous coherent shot with natural motion and stable composition. "
    "No text, subtitles, logos, or watermarks unless requested."
)


def _tool_prompt(command: str) -> str:
    command = command.strip()
    if not command:
        raise ValueError("video prompt is required")
    return f"{command}. {GENERIC_VIDEO_CONSTRAINTS}"


class VideoExecutor:
    """One bounded tool call: original request + generic guard -> one MP4."""

    def __init__(
        self,
        *,
        kernel_slug: str = PRODUCTION_VIDEO_KERNEL,
        inference_steps: int = 20,
        poll_interval: float = 5.0,
        max_poll_attempts: int = 180,
    ) -> None:
        self.kernel_slug = kernel_slug
        self.inference_steps = inference_steps
        self.poll_interval = poll_interval
        self.max_wait_seconds = max(1.0, poll_interval * max_poll_attempts)

    @staticmethod
    def seed_for(command: str) -> int:
        return int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")

    def _provider(self) -> SimpleKaggleVideoTool:
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        if not token:
            raise RuntimeError("KAGGLE_API_TOKEN is required")
        if not username:
            raise RuntimeError("KAGGLE_USERNAME is required")
        return SimpleKaggleVideoTool(
            worker=KaggleGpuWorker(
                api_token=token,
                username=username,
                timeout=45,
                submission_retry_attempts=2,
                submission_retry_delay_seconds=10,
            ),
            kernel_slug=self.kernel_slug,
            poll_interval=self.poll_interval,
            max_wait_seconds=min(self.max_wait_seconds, 900.0),
            inference_steps=self.inference_steps,
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
    ) -> SimpleVideoArtifact:
        original = command.strip()
        prompt = _tool_prompt(original)
        return self._provider().generate(
            prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            seed=self.seed_for(original),
        )
