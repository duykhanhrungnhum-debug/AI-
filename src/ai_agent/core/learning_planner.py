"""Turn detected knowledge gaps into explicit, bounded learning tasks."""
from __future__ import annotations

from dataclasses import dataclass

from .invariants import assert_core_invariants
from .knowledge_gaps import GapStatus, KnowledgeGap, KnowledgeGapDetector


@dataclass(frozen=True)
class LearningTask:
    gap_id: str
    objective: str
    max_attempts: int = 3


class LearningPlanner:
    """Create research objectives from gaps while preventing unbounded self-loops."""

    def __init__(self, detector: KnowledgeGapDetector | None = None, *, max_tasks: int = 5):
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        self.detector = detector or KnowledgeGapDetector(max_gaps=max_tasks)
        self.max_tasks = max_tasks

    def plan(self, task: str, knowledge) -> list[LearningTask]:
        assert_core_invariants()
        gaps = self.detector.detect(task, knowledge)
        planned: list[LearningTask] = []
        for gap in sorted(gaps, key=lambda item: item.priority, reverse=True)[: self.max_tasks]:
            gap.status = GapStatus.PLANNED
            planned.append(
                LearningTask(
                    gap_id=gap.id,
                    objective=f"Research and verify: {gap.topic}. Reason: {gap.reason}",
                )
            )
        return planned
