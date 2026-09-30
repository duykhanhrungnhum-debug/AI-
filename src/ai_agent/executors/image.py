"""Image skill executor with the existing verified cold provider as fallback."""
from __future__ import annotations

from hashlib import sha256
import os

from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_v2 import ImageRequestV2, ImageResultV2, KaggleImageV2Provider


class ImageExecutor:
    """Execute image work without exposing Kaggle details to the router/API layer."""

    def __init__(self, *, kernel_slug: str = "ai-agent-image-v2") -> None:
        self.kernel_slug = kernel_slug

    @staticmethod
    def seed_for(command: str) -> int:
        return int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")

    def _provider(self) -> KaggleImageV2Provider:
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        if not token:
            raise RuntimeError("KAGGLE_API_TOKEN is required")
        if not username:
            raise RuntimeError("KAGGLE_USERNAME is required")
        return KaggleImageV2Provider(
            worker=KaggleGpuWorker(
                api_token=token,
                username=username,
                timeout=120,
                submission_retry_attempts=5,
                submission_retry_delay_seconds=30,
            ),
            kernel_slug=self.kernel_slug,
            poll_interval=3,
            max_poll_attempts=600,
        )

    def execute_cold(
        self,
        command: str,
        *,
        item_id: str,
        width: int = 1024,
        height: int = 1024,
    ) -> ImageResultV2:
        """Backward-compatible one-image cold execution."""
        return self._provider().generate_many((ImageRequestV2(
            item_id=item_id,
            command=command,
            seed=self.seed_for(command),
            width=width,
            height=height,
        ),))[0]

    def execute_command_cold(
        self,
        command: str,
        *,
        width: int = 1024,
        height: int = 1024,
        max_images: int = 6,
    ) -> tuple[ImageResultV2, ...]:
        """Execute all separate images requested by one command in one session."""
        return self._provider().generate_command(
            command,
            seed=self.seed_for(command),
            width=width,
            height=height,
            max_images=max_images,
        )
