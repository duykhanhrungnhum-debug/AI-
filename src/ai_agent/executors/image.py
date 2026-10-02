"""Production image executor: preserve user semantics and run FLUX once."""
from __future__ import annotations

from hashlib import sha256
import os
import re

from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_v2 import ImageRequestV2, ImageResultV2
from ai_agent.core.media_v2_direct import DirectStableKaggleImageV2Provider


# Compatibility/evidence label: there is no external recaption model in the
# production image path. FLUX.2 Klein uses its own bundled text encoder.
PRODUCTION_RECAPTION_MODEL = "direct-original-language"
PRODUCTION_IMAGE_MAX_WAIT_SECONDS = 720.0
PRODUCTION_IMAGE_KERNEL = "ai-agent-image-v2-direct"


class ArtifactPollingKaggleImageWorker(KaggleGpuWorker):
    """Kaggle worker with nonblocking artifact-only completion polling."""

    def logs(self, slug: str) -> str:
        return ""

    def submit_script(self, **kwargs):
        kwargs.setdefault("machine_shape", "NvidiaTeslaT4")
        return super().submit_script(**kwargs)


_VI_ORDINALS = {
    "nhất": 1, "một": 1, "hai": 2, "ba": 3, "tư": 4, "bốn": 4,
    "năm": 5, "sáu": 6, "bảy": 7, "tám": 8,
}
_EN_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
}


def _split_explicit_image_items(command: str, *, max_images: int) -> tuple[str, ...] | None:
    """Split explicit ordered images without interpreting subjects or styles."""
    patterns = (
        re.compile(r"(?i)\b(?:ảnh|hình)\s+(?:thứ\s+)?(nhất|một|hai|ba|tư|bốn|năm|sáu|bảy|tám)\b"),
        re.compile(r"(?i)\b(?:ảnh|hình)\s+(?:số\s*)?([1-8])\b"),
        re.compile(r"(?i)\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\s+(?:image|picture)\b"),
        re.compile(r"(?i)\b(?:image|picture)\s*#?\s*([1-8])\b"),
    )
    found: list[tuple[int, int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(command):
            token = match.group(1).casefold()
            index = int(token) if token.isdigit() else (_VI_ORDINALS.get(token) or _EN_ORDINALS.get(token) or 0)
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
    """One bounded production path: original prompt -> FLUX.2 Klein."""

    def __init__(self, *, kernel_slug: str = PRODUCTION_IMAGE_KERNEL) -> None:
        # Migrate legacy callers transparently to the clean direct kernel slug.
        self.kernel_slug = PRODUCTION_IMAGE_KERNEL if kernel_slug == "ai-agent-image-v2" else kernel_slug

    @staticmethod
    def seed_for(command: str) -> int:
        return int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")

    def _provider(self) -> DirectStableKaggleImageV2Provider:
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        if not token:
            raise RuntimeError("KAGGLE_API_TOKEN is required")
        if not username:
            raise RuntimeError("KAGGLE_USERNAME is required")
        provider = DirectStableKaggleImageV2Provider(
            worker=ArtifactPollingKaggleImageWorker(
                api_token=token,
                username=username,
                timeout=45,
                submission_retry_attempts=2,
                submission_retry_delay_seconds=10,
            ),
            kernel_slug=self.kernel_slug,
            poll_interval=3,
            max_poll_attempts=240,
            recaption_model=PRODUCTION_RECAPTION_MODEL,
        )
        provider.max_wait_seconds = PRODUCTION_IMAGE_MAX_WAIT_SECONDS
        return provider

    def execute_cold(self, command: str, *, item_id: str, width: int = 1024, height: int = 1024) -> ImageResultV2:
        return self._provider().generate_many((ImageRequestV2(
            item_id=item_id,
            command=command.strip(),
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
        """Preserve every item prompt verbatim; split only explicit file boundaries."""
        provider = self._provider()
        explicit_items = _split_explicit_image_items(command, max_images=max_images)
        item_commands = explicit_items or (command.strip(),)
        base_seed = self.seed_for(command)
        requests = tuple(
            ImageRequestV2(
                item_id=f"image-{index:02d}",
                command=item_command,
                seed=(base_seed + index - 1) % (2 ** 32),
                width=width,
                height=height,
            )
            for index, item_command in enumerate(item_commands, 1)
        )
        return provider.generate_many(requests)
