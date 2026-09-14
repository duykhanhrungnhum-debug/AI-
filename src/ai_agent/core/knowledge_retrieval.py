"""Bounded retrieval of learned knowledge and prior execution experience."""
from __future__ import annotations

from dataclasses import dataclass

from .experience import Experience, ExperienceStore
from .invariants import assert_core_invariants
from .knowledge import KnowledgeItem, KnowledgeStatus


@dataclass(frozen=True)
class RetrievalContext:
    """Relevant prior knowledge and experience, with verification state preserved."""

    knowledge: tuple[KnowledgeItem, ...] = ()
    experiences: tuple[Experience, ...] = ()

    def format_for_prompt(self, *, max_chars: int = 6000) -> str:
        """Format bounded context without presenting unverified knowledge as fact."""
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        sections: list[str] = []
        if self.knowledge:
            lines = ["Relevant learned knowledge (status is authoritative):"]
            for item in self.knowledge:
                sources = ", ".join(source.uri for source in item.sources[:3]) or "none"
                lines.append(
                    f"- [{item.status.value}] {item.statement} "
                    f"(confidence={item.confidence:.2f}; sources={sources})"
                )
            sections.append("\n".join(lines))
        if self.experiences:
            lines = ["Relevant prior experiences (lessons, not guarantees):"]
            for item in self.experiences:
                outcome = "success" if item.success else "failure"
                lines.append(f"- [{outcome}] {item.lesson} Outcome: {item.outcome}")
            sections.append("\n".join(lines))
        if not sections:
            return "No relevant prior knowledge or experience was retrieved."
        return "\n\n".join(sections)[:max_chars]


class KnowledgeRetriever:
    """Lexical baseline retriever over knowledge plus bounded execution experience."""

    def __init__(
        self,
        knowledge_items: list[KnowledgeItem] | tuple[KnowledgeItem, ...] = (),
        *,
        experience_store: ExperienceStore | None = None,
        max_knowledge: int = 5,
        max_experiences: int = 5,
    ) -> None:
        if max_knowledge <= 0 or max_experiences <= 0:
            raise ValueError("retrieval limits must be positive")
        self._knowledge = list(knowledge_items)
        self.experience_store = experience_store
        self.max_knowledge = max_knowledge
        self.max_experiences = max_experiences

    def replace_knowledge(self, items: list[KnowledgeItem]) -> None:
        assert_core_invariants()
        self._knowledge = list(items)

    @staticmethod
    def _tokens(query: str) -> set[str]:
        return {token.lower() for token in query.split() if len(token) >= 4}

    def retrieve(self, query: str) -> RetrievalContext:
        assert_core_invariants()
        if not query.strip():
            raise ValueError("query is required")
        tokens = self._tokens(query)
        ranked: list[tuple[int, str, KnowledgeItem]] = []
        for item in self._knowledge:
            haystack = f"{item.statement} {' '.join(item.tags)}".lower()
            score = sum(token in haystack for token in tokens)
            if score:
                # Verified knowledge is preferred, but status is never rewritten.
                verified_bonus = 1 if item.status is KnowledgeStatus.VERIFIED else 0
                ranked.append((score + verified_bonus, item.updated_at, item))
        ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
        knowledge = tuple(item for _, _, item in ranked[: self.max_knowledge])
        experiences = tuple(
            self.experience_store.relevant(query, limit=self.max_experiences)
            if self.experience_store is not None
            else ()
        )
        return RetrievalContext(knowledge=knowledge, experiences=experiences)
