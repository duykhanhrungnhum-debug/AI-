"""Learning pipeline that keeps observation separate from verified knowledge."""
from __future__ import annotations

from dataclasses import dataclass, field

from .invariants import assert_core_invariants
from .knowledge import KnowledgeItem, KnowledgeSource, KnowledgeStatus
from .researcher import ResearchDocument


@dataclass
class LearningEngine:
    """Turn researched material into proposals; verification is explicit."""

    knowledge: list[KnowledgeItem] = field(default_factory=list)

    def observe(self, statement: str, document: ResearchDocument, *, kind="fact", tags=None) -> KnowledgeItem:
        """Create an unverified candidate with durable provenance."""
        assert_core_invariants()
        item = KnowledgeItem(
            statement=statement.strip(),
            kind=kind,
            sources=[KnowledgeSource(uri=document.uri, retrieved_at=document.retrieved_at, content_hash=document.content_hash)],
            evidence=[f"retrieved document hash: {document.content_hash}"],
            tags=list(tags or []),
        )
        if not item.statement:
            raise ValueError("knowledge statement must not be empty")
        self.knowledge.append(item)
        return item

    def verify(self, item: KnowledgeItem, evidence: list[str]) -> KnowledgeItem:
        """Promote a proposal only when provenance and evidence exist."""
        assert_core_invariants()
        item.verify(evidence)
        return item

    def conflict(self, item: KnowledgeItem, reason: str) -> KnowledgeItem:
        """Retain contradictory information instead of silently overwriting it."""
        assert_core_invariants()
        item.mark_conflicted(reason)
        return item

    def verified(self) -> list[KnowledgeItem]:
        return [item for item in self.knowledge if item.status is KnowledgeStatus.VERIFIED]
