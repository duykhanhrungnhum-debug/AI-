"""Bounded autonomous learning orchestration.

This module connects gap detection, learning planning, source discovery,
research, and explicit verification without claiming that retrieval alone is
knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .invariants import assert_core_invariants
from .knowledge import KnowledgeItem
from .learning import LearningEngine
from .learning_planner import LearningPlanner, LearningTask
from .research_plan import ResearchPlan, ResearchPlanner
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
    """Connect planning to source discovery, research, and verification."""

    def __init__(
        self,
        *,
        planner: LearningPlanner | None = None,
        research_planner: ResearchPlanner | None = None,
        researcher: InternetResearcher | None = None,
        learner: LearningEngine | None = None,
        max_attempts_per_task: int = 3,
    ) -> None:
        if max_attempts_per_task <= 0:
            raise ValueError("max_attempts_per_task must be positive")
        self.planner = planner or LearningPlanner()
        self.research_planner = research_planner
        self.researcher = researcher or InternetResearcher()
        self.learner = learner or LearningEngine()
        self.max_attempts_per_task = max_attempts_per_task

    def plan(self, goal: str) -> list[LearningTask]:
        assert_core_invariants()
        return self.planner.plan(goal, self.learner.knowledge)

    def discover(self, learning_task: LearningTask) -> ResearchPlan:
        """Search for bounded candidate sources; discovery never verifies truth."""
        assert_core_invariants()
        if self.research_planner is None:
            raise RuntimeError("research_planner is required for autonomous source discovery")
        return self.research_planner.plan(learning_task.objective)

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

    def discover_and_research(self, goal: str) -> AutonomousLearningSession:
        """Detect one gap, search multiple sources, and retain each result as proposed knowledge."""
        assert_core_invariants()
        session = AutonomousLearningSession(goal=goal)
        tasks = self.plan(goal)
        if not tasks:
            session.attempts.append(LearningAttempt("", goal, "no_gap", "No learning gap was detected."))
            return session

        task = tasks[0]
        try:
            research_plan = self.discover(task)
            if not research_plan.sources:
                session.attempts.append(
                    LearningAttempt(task.gap_id, task.objective, "failed", "Search returned no candidate sources.")
                )
                return session
            for source in research_plan.sources:
                try:
                    _, item = self.research(task, source.url)
                    session.attempts.append(
                        LearningAttempt(
                            task_id=task.gap_id,
                            objective=task.objective,
                            status="proposed",
                            reason="Source retrieved; verification still required.",
                            knowledge_id=item.id,
                            source_uri=source.url,
                        )
                    )
                except Exception as exc:
                    session.attempts.append(
                        LearningAttempt(task.gap_id, task.objective, "failed", str(exc), source_uri=source.url)
                    )
        except Exception as exc:
            session.attempts.append(LearningAttempt(task.gap_id, task.objective, "failed", str(exc)))
        return session

    def run_once(
        self,
        goal: str,
        uri: str | None = None,
        evidence: list[str] | None = None,
    ) -> AutonomousLearningSession:
        """Run one bounded cycle.

        With ``uri`` supplied, this preserves the explicit single-source flow.
        Without it, a configured ResearchPlanner searches multiple sources and
        leaves the resulting knowledge proposed until evidence is supplied.
        """
        assert_core_invariants()
        if uri is None:
            return self.discover_and_research(goal)

        session = AutonomousLearningSession(goal=goal)
        tasks = self.plan(goal)
        if not tasks:
            session.attempts.append(LearningAttempt("", goal, "no_gap", "No learning gap was detected."))
            return session

        task = tasks[0]
        if self.max_attempts_per_task < 1:
            session.attempts.append(LearningAttempt(task.gap_id, task.objective, "blocked", "Attempt limit reached."))
            return session

        try:
            _, item = self.research(task, uri)
            if evidence is None:
                session.attempts.append(
                    LearningAttempt(task.gap_id, task.objective, "proposed", "Source retrieved; verification still required.", item.id, uri)
                )
            else:
                session.attempts.append(self.verify(task, item, evidence))
        except Exception as exc:
            session.attempts.append(LearningAttempt(task.gap_id, task.objective, "failed", str(exc)))
        return session
