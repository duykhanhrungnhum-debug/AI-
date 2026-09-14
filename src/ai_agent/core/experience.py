"""Persistent experience records for learning from successes and failures."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .invariants import assert_core_invariants


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Experience:
    goal: str
    outcome: str
    lesson: str
    success: bool
    id: str = field(default_factory=lambda: str(uuid4()))
    evidence: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal is required")
        if not self.outcome.strip():
            raise ValueError("outcome is required")
        if not self.lesson.strip():
            raise ValueError("lesson is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "outcome": self.outcome,
            "lesson": self.lesson,
            "success": self.success,
            "evidence": list(self.evidence),
            "tags": list(self.tags),
            "created_at": self.created_at,
        }


class ExperienceStore:
    """Bounded in-memory experience store with invariant protection."""

    def __init__(self, *, max_items: int = 1000) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self._items: list[Experience] = []

    def record(self, experience: Experience) -> Experience:
        assert_core_invariants()
        self._items.append(experience)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]
        return experience

    def all(self) -> list[Experience]:
        assert_core_invariants()
        return list(self._items)

    def relevant(self, goal: str, *, limit: int = 5) -> list[Experience]:
        assert_core_invariants()
        if not goal.strip():
            raise ValueError("goal is required")
        if limit <= 0:
            raise ValueError("limit must be positive")
        tokens = {token.lower() for token in goal.split() if len(token) >= 4}
        ranked = []
        for item in self._items:
            haystack = f"{item.goal} {item.lesson} {item.outcome}".lower()
            score = sum(token in haystack for token in tokens)
            if score:
                ranked.append((score, item.created_at, item))
        ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return [item for _, _, item in ranked[:limit]]
