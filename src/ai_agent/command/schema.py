"""Canonical command envelope shared by AIKA entry points."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from uuid import uuid4


@dataclass(frozen=True)
class CommandEnvelope:
    """One normalized AIKA command regardless of caller transport."""

    command: str
    source: str = "chat"
    command_id: str = field(default_factory=lambda: uuid4().hex)
    skill_hint: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("command is required")
        if not self.source.strip():
            raise ValueError("source is required")
        if not self.command_id.strip():
            raise ValueError("command_id is required")
