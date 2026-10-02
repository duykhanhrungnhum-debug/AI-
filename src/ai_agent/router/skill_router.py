"""Central skill router for AIKA commands.

The router classifies only which tool AIKA should call. It deliberately does not
interpret species, garments, characters, visual styles, or scene semantics.
"""
from __future__ import annotations

import re
import unicodedata


def _fold_text(value: str) -> str:
    text = unicodedata.normalize("NFD", value.casefold())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def route_skill(command: str, *, skill_hint: str = "") -> str:
    """Return one stable AIKA tool name: chat, image, or video."""
    hint = skill_hint.strip().casefold()
    if hint in {"chat", "image", "video"}:
        return hint

    text = _fold_text(command.strip())
    if not text:
        return "chat"
    routed = re.sub(r"^aika[\s,:;\-]+", "", text).strip()
    question_prefixes = (
        "giai thich", "tai sao", "vi sao", "cho toi biet", "kiem tra", "bao cao",
        "phan tich", "danh gia", "lam sao", "nhu the nao", "co the",
    )
    if routed.endswith("?") or any(routed.startswith(prefix) for prefix in question_prefixes):
        return "chat"

    create_verbs = (
        "tao", "ve", "lam", "generate", "create", "draw", "render", "thiet ke",
        "animate", "hoat hoa",
    )
    has_action = any(re.search(rf"\b{re.escape(verb)}\b", routed) for verb in create_verbs)
    if not has_action:
        return "chat"

    # Video wins before image so requests such as "tạo video từ ảnh" correctly
    # call the video tool rather than being captured by the word "ảnh".
    video_nouns = (
        "video", "clip", "doan phim", "phim ngan", "motion video", "animation video",
    )
    has_video = any(re.search(rf"\b{re.escape(noun)}\b", routed) for noun in video_nouns)
    if has_video:
        return "video"

    image_nouns = (
        "anh", "hinh anh", "hinh", "image", "photo", "picture", "portrait",
        "poster", "minh hoa", "illustration",
    )
    has_image = any(re.search(rf"\b{re.escape(noun)}\b", routed) for noun in image_nouns)
    return "image" if has_image else "chat"
