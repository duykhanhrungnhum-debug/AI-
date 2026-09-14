"""Orchestrate research, evidence capture, verification, and durable learning."""
from __future__ import annotations

from dataclasses import dataclass

from .knowledge import KnowledgeItem
from .learning import LearningEngine
from .researcher import InternetResearcher


@dataclass(frozen=True)
class LearningCycleResult:
    item: KnowledgeItem
    verified: bool
    reason: str


class AutonomousLearningLoop:
    """Run one bounded learning cycle without confusing retrieval with truth."""

    def __init__(self, researcher: InternetResearcher | None = None, learner: LearningEngine | None = None):
        self.researcher = researcher or InternetResearcher()
        self.learner = learner or LearningEngine()

    def research_and_propose(self, statement: str, uri: str, *, kind="fact", tags=None) -> LearningCycleResult:
        document = self.researcher.fetch(uri)
        item = self.learner.observe(statement, document, kind=kind, tags=tags)
        return LearningCycleResult(item, False, "Retrieved and proposed; independent verification is still required.")

    def verify_proposal(self, item: KnowledgeItem, evidence: list[str]) -> LearningCycleResult:
        verified = self.learner.verify(item, evidence)
        return LearningCycleResult(verified, True, "Knowledge promoted to VERIFIED with provenance and evidence.")
