"""Record verified runtime outcomes as reusable experience."""

from __future__ import annotations

from .experience import Experience, ExperienceStore
from .invariants import assert_core_invariants
from .model_runtime import ActionExecution


class RuntimeExperienceRecorder:
    """Turn observable execution outcomes into bounded learning records."""

    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def record(self, goal: str, action: str, result: ActionExecution) -> Experience:
        assert_core_invariants()
        success = bool(result.success)
        lesson = (
            "Verified execution outcome can be reused."
            if success
            else "Execution failed; retain the failure as a recovery lesson."
        )
        experience = Experience(
            goal=goal,
            outcome=result.outcome,
            lesson=lesson,
            success=success,
            evidence=list(result.evidence),
            tags=["runtime", "success" if success else "failure"],
        )
        return self.store.record(experience)
