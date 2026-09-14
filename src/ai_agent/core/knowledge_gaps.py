"""Detect missing knowledge and turn it into bounded learning work."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from .invariants import assert_core_invariants
from .knowledge import KnowledgeItem, KnowledgeStatus


class GapStatus(str, Enum):
    OPEN = "open"
    PLANNED = "planned"
    RESOLVED = "resolved"
    BLOCKED = "blocked"


@dataclass
class KnowledgeGap:
    """A concrete piece of knowledge the agent currently lacks or cannot trust."""

    topic: str
    reason: str
    priority: int = 50
    id: str = field(default_factory=lambda: str(uuid4()))
    status: GapStatus = GapStatus.OPEN
    related_knowledge_ids: list[str] = field(default_factory=list)
    attempts: int = 0

    def __post_init__(self) -> None:
        self.topic = self.topic.strip()
        self.reason = self.reason.strip()
        if not self.topic:
            raise ValueError("gap topic must not be empty")
        if not self.reason:
            raise ValueError("gap reason must not be empty")
        if not 0 <= self.priority <= 100:
            raise ValueError("gap priority must be between 0 and 100")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "reason": self.reason,
            "priority": self.priority,
            "status": self.status.value,
            "related_knowledge_ids": list(self.related_knowledge_ids),
            "attempts": self.attempts,
        }


class KnowledgeGapDetector:
    """Find actionable gaps without pretending absence of evidence is falsehood."""

    def __init__(self, *, max_gaps: int = 5):
        if max_gaps <= 0:
            raise ValueError("max_gaps must be positive")
        self.max_gaps = max_gaps

    def detect(self, task: str, knowledge: list[KnowledgeItem]) -> list[KnowledgeGap]:
        assert_core_invariants()
        task = task.strip()
        if not task:
            raise ValueError("task must not be empty")

        normalized = task.lower()
        matches = [item for item in knowledge if any(token in item.statement.lower() for token in normalized.split() if len(token) > 3)]
        gaps: list[KnowledgeGap] = []

        if not matches:
            gaps.append(KnowledgeGap(task, "No related knowledge was found.", priority=100))
        else:
            unresolved = [item for item in matches if item.status is not KnowledgeStatus.VERIFIED]
            if unresolved:
                gaps.append(
                    KnowledgeGap(
                        task,
                        "Related knowledge exists but is not fully verified.",
                        priority=90,
                        related_knowledge_ids=[item.id for item in unresolved],
                    )
                )

        return gaps[: self.max_gaps]
