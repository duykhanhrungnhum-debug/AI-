"""Structural planner for explicit multi-image commands.

The planner owns file count and item boundaries when the user's command makes
those boundaries explicit. It never classifies subject/species/style; those
remain semantic work for Qwen. This prevents model precision/device changes
from silently turning N requested files into a different number of files.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PlannedImageItem:
    index: int
    command: str


_VI_ORDINALS = {
    "nhất": 1,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "tư": 4,
    "bốn": 4,
    "năm": 5,
    "sáu": 6,
}
_EN_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
}

_VI_NUMBER = {
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "sáu": 6,
}
_EN_NUMBER = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def requested_image_count(command: str, *, max_images: int = 6) -> int | None:
    """Return an explicit requested file count, never infer one from subjects."""
    text = " ".join(command.casefold().split())
    digit = re.search(
        r"\b([1-8])\s*(?:ảnh|hình|image|images|picture|pictures|file|files)\b",
        text,
    )
    if digit:
        value = int(digit.group(1))
        return value if 1 <= value <= max_images else None

    for word, value in {**_VI_NUMBER, **_EN_NUMBER}.items():
        if re.search(
            rf"\b{re.escape(word)}\s*(?:ảnh|hình|image|images|picture|pictures|file|files)\b",
            text,
        ):
            return value if value <= max_images else None
    return None


def _markers(command: str) -> list[tuple[int, int, int]]:
    """Return (start,end,index) for explicit per-image ordinal markers."""
    found: list[tuple[int, int, int]] = []
    patterns = [
        # Vietnamese: ảnh thứ nhất / hình thứ hai / ảnh 1 / hình số 2
        re.compile(
            r"(?i)\b(?:ảnh|hình)\s+(?:thứ\s+)?(nhất|một|hai|ba|tư|bốn|năm|sáu)\b"
        ),
        re.compile(r"(?i)\b(?:ảnh|hình)\s+(?:số\s*)?([1-6])\b"),
        # English: first image / second picture / image 1 / picture #2
        re.compile(
            r"(?i)\b(first|second|third|fourth|fifth|sixth)\s+(?:image|picture|file)\b"
        ),
        re.compile(r"(?i)\b(?:image|picture|file)\s*#?\s*([1-6])\b"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(command):
            token = match.group(1).casefold()
            if token.isdigit():
                index = int(token)
            else:
                index = _VI_ORDINALS.get(token) or _EN_ORDINALS.get(token) or 0
            if index:
                found.append((match.start(), match.end(), index))
    # Avoid overlapping alternative regexes; prefer the earliest/longest marker.
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    deduped: list[tuple[int, int, int]] = []
    last_end = -1
    for item in found:
        if item[0] < last_end:
            continue
        deduped.append(item)
        last_end = item[1]
    return deduped


def plan_explicit_images(
    command: str,
    *,
    max_images: int = 6,
) -> tuple[PlannedImageItem, ...] | None:
    """Split explicitly enumerated image requests; otherwise return None.

    Returning None deliberately leaves ambiguous natural-language batching to
    the semantic recaption model. A structural plan is emitted only when item
    boundaries are explicit and contiguous from 1..N.
    """
    command = command.strip()
    if not command:
        return None
    markers = _markers(command)
    if len(markers) < 2:
        return None

    indexes = [item[2] for item in markers]
    expected = list(range(1, len(markers) + 1))
    if indexes != expected or len(markers) > max_images:
        return None

    explicit_count = requested_image_count(command, max_images=max_images)
    if explicit_count is not None and explicit_count != len(markers):
        return None

    items: list[PlannedImageItem] = []
    for pos, (start, end, index) in enumerate(markers):
        next_start = markers[pos + 1][0] if pos + 1 < len(markers) else len(command)
        body = command[end:next_start].strip(" \t\r\n,;:.-")
        if not body:
            return None
        items.append(PlannedImageItem(index=index, command=body))
    return tuple(items)
