"""Turn detected knowledge gaps into explicit, bounded learning tasks."""
from __future__ import annotations

from dataclasses import dataclass

from .experience import ExperienceStore
from .invariants import assert_core_invariants
from .knowledge_gaps import GapStatus, KnowledgeGapDetector


@dataclass(frozen=True)
class LearningTask:
    gap_id: str
    objective: str
    max_attempts: int = 3


class LearningPlanner:
    """Create bounded learning work and use prior experience to avoid repeated mistakes."""

    def __init__(self, detector: KnowledgeGapDetector | None = None, *, max_tasks: int = 5,
                 experience_store: ExperienceStore | None = None):
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        self.detector = detector or KnowledgeGapDetector(max_gaps=max_tasks)
        self.max_tasks = max_tasks
        self.experience_store = experience_store

    def plan(self, task: str, knowledge) -> list[LearningTask]:
        assert_core_invariants()
        gaps = self.detector.detect(task, knowledge)
        planned: list[LearningTask] = []
        experiences = self.experience_store.relevant(task, limit=3) if self.experience_store else []
        failures = [item for item in experiences if not item.success]
        for gap in sorted(gaps, key=lambda item: item.priority, reverse=True)[: self.max_tasks]:
            gap.status = GapStatus.PLANNED
            objective = f"Research and verify: {gap.topic}. Reason: {gap.reason}"
            if failures:
                lessons = "; ".join(item.lesson for item in failures)
                objective += f" Prior failure lessons to avoid repeating: {lessons}"
            planned.append(LearningTask(gap_id=gap.id, objective=objective))
        return planned
