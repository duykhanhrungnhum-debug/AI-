"""Deterministic multi-source comparison for research evidence.

This module deliberately performs lexical corroboration only. It does not
claim semantic understanding or truth merely because sources agree.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .invariants import assert_core_invariants
from .researcher import ResearchDocument


@dataclass(frozen=True)
class SourceComparison:
    source_count: int
    distinct_content_count: int
    corroborated: bool
    evidence: list[str]
    reason: str


class SourceComparator:
    """Compare retrieved documents using conservative lexical corroboration."""

    def __init__(self, *, min_sources: int = 2, min_keyword_length: int = 4):
        if min_sources < 2:
            raise ValueError("min_sources must be at least 2")
        if min_keyword_length <= 0:
            raise ValueError("min_keyword_length must be positive")
        self.min_sources = min_sources
        self.min_keyword_length = min_keyword_length

    def compare(self, statement: str, documents: list[ResearchDocument]) -> SourceComparison:
        assert_core_invariants()
        statement = statement.strip().lower()
        if not statement:
            raise ValueError("statement must not be empty")
        if not documents:
            return SourceComparison(0, 0, False, [], "No documents were retrieved.")

        unique: dict[str, ResearchDocument] = {doc.content_hash or doc.uri: doc for doc in documents}
        keywords = [
            token for token in re.findall(r"[a-z0-9]+", statement)
            if len(token) >= self.min_keyword_length
        ]
        if not keywords:
            return SourceComparison(
                len(documents), len(unique), False, [],
                "Statement has no sufficiently distinctive keywords for lexical corroboration.",
            )

        supporting = []
        for doc in unique.values():
            content = doc.content.lower()
            if all(keyword in content for keyword in keywords):
                supporting.append(doc)

        evidence = [
            f"lexical corroboration: {doc.uri} contains all statement keywords; hash={doc.content_hash}"
            for doc in supporting
        ]
        corroborated = len(supporting) >= self.min_sources
        reason = (
            f"{len(supporting)} distinct sources lexically corroborate the statement."
            if corroborated
            else "Insufficient independent lexical corroboration; keep knowledge proposed."
        )
        return SourceComparison(len(documents), len(unique), corroborated, evidence, reason)
