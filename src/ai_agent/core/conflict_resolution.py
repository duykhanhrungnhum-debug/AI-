"""Conservative conflict detection for multi-source research.

Conflict detection here is intentionally deterministic: sources that provide
opposite lexical signals for the same statement are marked as conflicting.
The component never silently chooses a winner.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .invariants import assert_core_invariants
from .researcher import ResearchDocument


@dataclass(frozen=True)
class ConflictAnalysis:
    source_count: int
    supporting_sources: list[str]
    opposing_sources: list[str]
    conflicted: bool
    reason: str


class ConflictResolver:
    """Detect explicit negation conflicts without inventing a resolution."""

    _NEGATIONS = ("not", "no", "never", "false", "incorrect", "wrong")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def analyze(self, statement: str, documents: list[ResearchDocument]) -> ConflictAnalysis:
        assert_core_invariants()
        normalized = statement.strip().lower()
        if not normalized:
            raise ValueError("statement must not be empty")

        statement_tokens = self._tokens(normalized)
        keywords = {
            token for token in statement_tokens
            if len(token) >= 4 and token not in self._NEGATIONS
        }
        supporting: list[str] = []
        opposing: list[str] = []

        for document in documents:
            content_tokens = self._tokens(document.content)
            if not keywords or not keywords.issubset(content_tokens):
                continue
            if self._tokens(document.content) & set(self._NEGATIONS):
                opposing.append(document.uri)
            else:
                supporting.append(document.uri)

        conflicted = bool(supporting and opposing)
        reason = (
            "Sources contain opposing lexical signals; no source was selected as the winner."
            if conflicted
            else "No explicit lexical conflict detected."
        )
        return ConflictAnalysis(
            source_count=len(documents),
            supporting_sources=supporting,
            opposing_sources=opposing,
            conflicted=conflicted,
            reason=reason,
        )
