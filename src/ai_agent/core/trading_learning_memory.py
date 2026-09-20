"""Bounded failure/lesson memory for unattended Trading Skill training."""
from __future__ import annotations

from urllib.parse import urlparse

COOLDOWN_THRESHOLD = 2
DEFAULT_COOLDOWN_CYCLES = 6
MAX_ERROR_SIGNATURES = 100
MAX_LESSONS = 100


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def normalize_error(error: str) -> tuple[str, str | None, str]:
    """Return (signature, host, kind) without retaining volatile URL details."""
    text = error.strip()
    if text.startswith("fetch:"):
        parts = text.split(":", 2)
        if len(parts) == 3:
            url, detail = parts[1], parts[2].strip()
            host = _host(url)
            if "HTTP Error 403" in detail:
                kind = "http_403"
            elif "HTTP Error 429" in detail:
                kind = "http_429"
            elif "timed out" in detail.lower():
                kind = "timeout"
            else:
                kind = detail[:80].lower().replace(" ", "_")
            return f"fetch:{host}:{kind}", host or None, kind
    if text.startswith("news-empty:"):
        return "news-empty:" + text.split(":", 2)[1], None, "news_empty"
    if text.startswith("news:"):
        category = text.split(":", 2)[1] if ":" in text else "unknown"
        return f"news:{category}:failure", None, "news_failure"
    return text[:120], None, "other"


def source_on_cooldown(state: dict, url: str, cycle: int) -> bool:
    host = _host(url)
    if not host:
        return False
    cooldown = state.get("source_cooldowns", {}).get(host)
    if not isinstance(cooldown, dict):
        return False
    return int(cooldown.get("until_cycle", 0)) >= cycle


def record_learning_memory(state: dict, errors: list[str], *, cycle: int) -> list[dict]:
    """Persist recurring failures as lessons and temporary source cooldowns."""
    memory = dict(state.get("error_memory", {}))
    cooldowns = dict(state.get("source_cooldowns", {}))
    lessons = list(state.get("lessons", []))
    lesson_signatures = {item.get("signature") for item in lessons if isinstance(item, dict)}
    new_lessons: list[dict] = []

    for error in errors:
        signature, host, kind = normalize_error(error)
        entry = dict(memory.get(signature, {}))
        count = int(entry.get("count", 0)) + 1
        entry.update({
            "count": count,
            "last_seen_cycle": cycle,
            "sample": error[:300],
            "kind": kind,
        })
        if host:
            entry["host"] = host
        memory[signature] = entry

        if host and count >= COOLDOWN_THRESHOLD and kind in {"http_403", "http_429", "timeout"}:
            until_cycle = cycle + DEFAULT_COOLDOWN_CYCLES
            previous = cooldowns.get(host, {})
            cooldowns[host] = {
                "reason": signature,
                "until_cycle": max(int(previous.get("until_cycle", 0)), until_cycle),
                "updated_cycle": cycle,
            }
            if signature not in lesson_signatures:
                lesson = {
                    "signature": signature,
                    "created_cycle": cycle,
                    "lesson": (
                        f"Repeated {kind} from {host}; stop retrying that direct source every cycle. "
                        "Use other verified primary sources/current-news signals and retry after cooldown."
                    ),
                }
                lessons.append(lesson)
                new_lessons.append(lesson)
                lesson_signatures.add(signature)

    # Keep the most recently seen signatures and a bounded lesson history.
    ranked = sorted(
        memory.items(),
        key=lambda item: int(item[1].get("last_seen_cycle", 0)),
        reverse=True,
    )[:MAX_ERROR_SIGNATURES]
    state["error_memory"] = dict(ranked)
    state["source_cooldowns"] = cooldowns
    state["lessons"] = lessons[-MAX_LESSONS:]
    return new_lessons
