"""Bounded autonomous learning orchestration.

This module connects gap detection, learning planning, research, and explicit
verification without claiming that retrieval alone is knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .invariants import assert_core_invariants
from .knowledge import KnowledgeItem
from .learning import LearningEngine
from .learning_planner import LearningPlanner, LearningTask
from .researcher import InternetResearcher, ResearchDocument


@dataclass
class LearningAttempt:
    task_id: str
    objective: str
    status: str
    reason: str
    knowledge_id: str | None = None
    source_uri: str | None = None


@dataclass
class AutonomousLearningSession:
    """State for one bounded learning session."""

    goal: str
    attempts: list[LearningAttempt] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return any(attempt.status == "verified" for attempt in self.attempts)


class AutonomousLearningCoordinator:
    """Connect planning to research and verification while remaining bounded."""

    def __init__(
        self,
        *,
        planner: LearningPlanner | None = None,
        researcher: InternetResearcher | None = None,
        learner: LearningEngine | None = None,
        max_attempts_per_task: int = 3,
    ) -> None:
        if max_attempts_per_task <= 0:
            raise ValueError("max_attempts_per_task must be positive")
        self.planner = planner or LearningPlanner()
        self.researcher = researcher or InternetResearcher()
        self.learner = learner or LearningEngine()
        self.max_attempts_per_task = max_attempts_per_task

    def plan(self, goal: str) -> list[LearningTask]:
        assert_core_invariants()
        return self.planner.plan(goal, self.learner.knowledge)

    def research(self, learning_task: LearningTask, uri: str) -> tuple[ResearchDocument, KnowledgeItem]:
        assert_core_invariants()
        document = self.researcher.fetch(uri)
        item = self.learner.observe(
            learning_task.objective,
            document,
            kind="lesson",
            tags=["autonomous-learning", learning_task.gap_id],
        )
        return document, item

    def verify(self, learning_task: LearningTask, item: KnowledgeItem, evidence: list[str]) -> LearningAttempt:
        assert_core_invariants()
        self.learner.verify(item, evidence)
        return LearningAttempt(
            task_id=learning_task.gap_id,
            objective=learning_task.objective,
            status="verified",
            reason="Evidence and provenance accepted.",
            knowledge_id=item.id,
            source_uri=item.sources[0].uri if item.sources else None,
        )

    def run_once(self, goal: str, uri: str, evidence: list[str]) -> AutonomousLearningSession:
        """Run exactly one bounded research/verification cycle.

        Repeated cycles must be explicitly requested by the caller, preventing
        accidental infinite autonomous loops.
        """
        assert_core_invariants()
        session = AutonomousLearningSession(goal=goal)
        tasks = self.plan(goal)
        if not tasks:
            session.attempts.append(
                LearningAttempt("", goal, "no_gap", "No learning gap was detected.")
            )
            return session

        task = tasks[0]
        if self.max_attempts_per_task < 1:
            session.attempts.append(
                LearningAttempt(task.gap_id, task.objective, "blocked", "Attempt limit reached.")
            )
            return session

        try:
            _, item = self.research(task, uri)
            session.attempts.append(self.verify(task, item, evidence))
        except Exception as exc:
            session.attempts.append(
                LearningAttempt(
                    task_id=task.gap_id,
                    objective=task.objective,
                    status="failed",
                    reason=str(exc),
                )
            )
        return session
