"""Structured long-term knowledge with provenance and verification state."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class KnowledgeStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    CONFLICTED = "conflicted"
    REJECTED = "rejected"


class KnowledgeKind(str, Enum):
    FACT = "fact"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    RULE = "rule"
    HYPOTHESIS = "hypothesis"
    LESSON = "lesson"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class KnowledgeSource:
    uri: str
    title: str = ""
    source_type: str = "web"
    retrieved_at: str = field(default_factory=utc_now)
    content_hash: str = ""


@dataclass
class KnowledgeItem:
    statement: str
    kind: KnowledgeKind = KnowledgeKind.FACT
    id: str = field(default_factory=lambda: str(uuid4()))
    status: KnowledgeStatus = KnowledgeStatus.PROPOSED
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    sources: list[KnowledgeSource] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        # Keep the public dataclass tolerant of serialized/string enum values
        # while maintaining enum-backed state internally.
        if isinstance(self.kind, str):
            self.kind = KnowledgeKind(self.kind)
        if isinstance(self.status, str):
            self.status = KnowledgeStatus(self.status)

    def verify(self, evidence: list[str] | None = None) -> None:
        supplied = evidence if evidence is not None else self.evidence
        if not supplied:
            raise ValueError("knowledge cannot be verified without evidence")
        if not self.sources:
            raise ValueError("knowledge cannot be verified without provenance")
        self.evidence = list(supplied)
        self.status = KnowledgeStatus.VERIFIED
        self.confidence = max(self.confidence, 1.0)
        self.updated_at = utc_now()

    def mark_conflicted(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("conflict reason is required")
        self.status = KnowledgeStatus.CONFLICTED
        self.evidence.append(reason)
        self.updated_at = utc_now()

    def reject(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("rejection reason is required")
        self.status = KnowledgeStatus.REJECTED
        self.evidence.append(reason)
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "kind": self.kind.value,
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "sources": [source.__dict__ for source in self.sources],
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
