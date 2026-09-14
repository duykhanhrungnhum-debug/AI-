"""Core data models for task execution and verification."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Evidence:
    """Evidence supporting a claim about an action or result."""

    kind: str
    description: str
    source: str | None = None
    verified: bool = False


@dataclass
class TaskResult:
    """Execution result that cannot claim success without verification."""

    status: TaskStatus
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        return any(item.verified for item in self.evidence)
