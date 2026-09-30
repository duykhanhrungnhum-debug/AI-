"""Central skill router for AIKA commands.

The router classifies at the skill level only. It intentionally does not contain
species, garment, visual-style, or other subject-specific tables.
"""
from __future__ import annotations

import re
import unicodedata


def _fold_text(value: str) -> str:
    text = unicodedata.normalize("NFD", value.casefold())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def route_skill(command: str, *, skill_hint: str = "") -> str:
    """Return the stable skill name for one natural-language command.

    Only skills with verified runtime executors are routed explicitly. Unknown or
    conversational input stays on the existing chat path so this router cannot
    silently break another AIKA skill during rollout.
    """
    hint = skill_hint.strip().casefold()
    if hint in {"chat", "image"}:
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

    image_nouns = (
        "anh", "hinh anh", "hinh", "image", "photo", "picture", "portrait",
        "poster", "minh hoa", "illustration",
    )
    create_verbs = (
        "tao", "ve", "lam", "generate", "create", "draw", "render", "thiet ke",
    )
    has_image = any(re.search(rf"\b{re.escape(noun)}\b", routed) for noun in image_nouns)
    has_action = any(re.search(rf"\b{re.escape(verb)}\b", routed) for verb in create_verbs)
    return "image" if has_image and has_action else "chat"
