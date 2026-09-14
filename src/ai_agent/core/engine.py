"""State-aware orchestration for long-running agent tasks."""

from dataclasses import dataclass

from .project_state import ProjectState, StateStore, StepStatus
from .recovery import RecoveryGuard, resume_task


@dataclass
class ExecutionEngine:
    """Resume, execute, and checkpoint work without losing project position."""

    state_store: StateStore
    recovery: RecoveryGuard

    def resume(self, state: ProjectState) -> str:
        """Return the exact next action implied by persisted state."""
        task = state.current_task()
        if task is None:
            return "No current task is selected."
        return resume_task(task, self.recovery)

    def begin_current_step(self, state: ProjectState) -> str:
        """Select the next unfinished step and persist the checkpoint."""
        task = state.current_task()
        if task is None:
            raise ValueError("Cannot begin work without a current task.")
        step = task.current_step() or task.advance_to_next_pending()
        if step is None:
            task.status = StepStatus.VERIFIED
            self.state_store.save(state)
            return "No unfinished step remains; verify the task result."
        self.state_store.save(state)
        return f"Started step {step.id}: {step.title}"

    def record_step_result(
        self,
        state: ProjectState,
        action: str,
        outcome: str,
        success: bool,
        evidence: str | None = None,
    ) -> None:
        """Record an attempt and checkpoint before the engine proceeds."""
        task = state.current_task()
        if task is None or task.current_step() is None:
            raise ValueError("Cannot record a result without a current step.")
        step = task.current_step()
        step.attempts += 1
        step.result = outcome
        if evidence:
            step.evidence.append(evidence)
        self.recovery.record(step.id, action, outcome, success)
        step.status = StepStatus.VERIFIED if success else StepStatus.FAILED
        if not success and not self.recovery.may_retry(step.id, action):
            step.status = StepStatus.BLOCKED
            task.blocked_reason = f"Repeated failure for action: {action}"
            task.status = StepStatus.BLOCKED
        self.state_store.save(state)
