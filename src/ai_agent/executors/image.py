"""Production image executor using the stable single-shot Image V2 path."""
from __future__ import annotations

from hashlib import sha256
import os
import re

from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_v2 import ImageRequestV2, ImageResultV2
from ai_agent.core.media_v2_stable import StableKaggleImageV2Provider


# Production recaption is intentionally compact: it only translates/normalizes
# image requests. Subject identity is enforced separately by the stable semantic
# gate, so a larger language model only increases cold-start without improving
# the image model itself.
PRODUCTION_RECAPTION_MODEL = "Qwen/Qwen3-0.6B"
PRODUCTION_IMAGE_MAX_WAIT_SECONDS = 900.0


class ArtifactPollingKaggleImageWorker(KaggleGpuWorker):
    """Kaggle image worker whose control plane never blocks on live log streaming.

    The Kaggle logs endpoint can hold a request for 60-120 seconds while a GPU
    kernel is queued/running. Production completion is already proven by the
    run-tokened output artifact, so live logs must not sit in the polling loop.
    Logs remain available to separate diagnostic/probe tooling when needed.
    """

    def logs(self, slug: str) -> str:
        return ""

    def submit_script(self, **kwargs):
        # Request the supported T4 accelerator explicitly instead of leaving
        # accelerator selection implicit in Kaggle's scheduler.
        kwargs.setdefault("machine_shape", "NvidiaTeslaT4")
        return super().submit_script(**kwargs)


_VI_ORDINALS = {
    "nhất": 1,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "tư": 4,
    "bốn": 4,
    "năm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
}
_EN_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
}


def _split_explicit_image_items(command: str, *, max_images: int) -> tuple[str, ...] | None:
    """Split only explicitly enumerated separate images; never classify subjects/styles."""
    patterns = (
        re.compile(
            r"(?i)\b(?:ảnh|hình)\s+(?:thứ\s+)?"
            r"(nhất|một|hai|ba|tư|bốn|năm|sáu|bảy|tám)\b"
        ),
        re.compile(r"(?i)\b(?:ảnh|hình)\s+(?:số\s*)?([1-8])\b"),
        re.compile(
            r"(?i)\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\s+"
            r"(?:image|picture)\b"
        ),
        re.compile(r"(?i)\b(?:image|picture)\s*#?\s*([1-8])\b"),
    )
    found: list[tuple[int, int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(command):
            token = match.group(1).casefold()
            if token.isdigit():
                index = int(token)
            else:
                index = _VI_ORDINALS.get(token) or _EN_ORDINALS.get(token) or 0
            if index:
                found.append((match.start(), match.end(), index))

    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    markers: list[tuple[int, int, int]] = []
    last_end = -1
    for marker in found:
        if marker[0] < last_end:
            continue
        markers.append(marker)
        last_end = marker[1]

    if len(markers) < 2 or len(markers) > max_images:
        return None
    if [marker[2] for marker in markers] != list(range(1, len(markers) + 1)):
        return None

    items: list[str] = []
    for position, (_, end, _) in enumerate(markers):
        next_start = markers[position + 1][0] if position + 1 < len(markers) else len(command)
        body = command[end:next_start].strip(" \t\r\n,;:.-")
        body = re.sub(r"(?i)^(?:là|is)\s+", "", body).strip()
        if not body:
            return None
        items.append(body)
    return tuple(items)


class ImageExecutor:
    """Execute image work through one simple, bounded cold Image V2 run."""

    def __init__(self, *, kernel_slug: str = "ai-agent-image-v2") -> None:
        self.kernel_slug = kernel_slug

    @staticmethod
    def seed_for(command: str) -> int:
        return int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")

    def _provider(self) -> StableKaggleImageV2Provider:
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        if not token:
            raise RuntimeError("KAGGLE_API_TOKEN is required")
        if not username:
            raise RuntimeError("KAGGLE_USERNAME is required")
        provider = StableKaggleImageV2Provider(
            worker=ArtifactPollingKaggleImageWorker(
                api_token=token,
                username=username,
                timeout=120,
                submission_retry_attempts=2,
                submission_retry_delay_seconds=10,
            ),
            kernel_slug=self.kernel_slug,
            poll_interval=3,
            max_poll_attempts=300,
            recaption_model=PRODUCTION_RECAPTION_MODEL,
        )
        provider.max_wait_seconds = PRODUCTION_IMAGE_MAX_WAIT_SECONDS
        return provider

    def execute_cold(
        self,
        command: str,
        *,
        item_id: str,
        width: int = 1024,
        height: int = 1024,
    ) -> ImageResultV2:
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
        """Generate requested images in one bounded GPU session.

        Explicitly enumerated batches are split deterministically before Qwen so a
        language model cannot accidentally collapse ``ảnh thứ nhất/hai/ba`` into
        one picture. Each item is still semantically recaptioned by the same Qwen
        model loaded once inside the same Kaggle run; FLUX is also loaded once.
        """
        provider = self._provider()
        explicit_items = _split_explicit_image_items(command, max_images=max_images)
        if explicit_items:
            base_seed = self.seed_for(command)
            requests = tuple(
                ImageRequestV2(
                    item_id=f"image-{index:02d}",
                    command=item_command,
                    seed=(base_seed + index - 1) % (2 ** 32),
                    width=width,
                    height=height,
                )
                for index, item_command in enumerate(explicit_items, 1)
            )
            return provider.generate_many(requests)

        return provider.generate_command(
            command,
            seed=self.seed_for(command),
            width=width,
            height=height,
            max_images=max_images,
        )
