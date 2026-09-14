"""Recovery rules for resuming long-running work without looping."""

from dataclasses import dataclass, field

from .project_state import StepStatus, TaskState


@dataclass
class Attempt:
    step_id: str
    action: str
    outcome: str
    success: bool


@dataclass
class RecoveryGuard:
    """Track attempts and prevent blind repetition of failed actions."""

    attempts: list[Attempt] = field(default_factory=list)
    max_identical_failures: int = 2

    def record(self, step_id: str, action: str, outcome: str, success: bool) -> None:
        self.attempts.append(Attempt(step_id, action, outcome, success))

    def may_retry(self, step_id: str, action: str) -> bool:
        failures = sum(
            1
            for item in self.attempts
            if item.step_id == step_id and item.action == action and not item.success
        )
        return failures < self.max_identical_failures


def resume_task(task: TaskState, guard: RecoveryGuard) -> str:
    """Return a deterministic resume instruction based on persisted state."""
    current = task.current_step()
    if current is None:
        current = task.advance_to_next_pending()
    if current is None:
        task.status = StepStatus.VERIFIED
        return "Task has no unfinished step; verify the final task result."
    if current.status == StepStatus.BLOCKED:
        return "Task is blocked; resolve the blocker before executing more work."
    return f"Resume step {current.id}: {current.title}"
