"""Deterministic planning for turning a goal into ordered executable steps."""

from dataclasses import dataclass

from .project_state import ProjectPhase, ProjectState, Step, StepStatus, TaskState


@dataclass(frozen=True)
class PlanStepSpec:
    """A validated step produced by a planner."""

    id: str
    title: str


class Planner:
    """Create and apply ordered plans without disturbing verified progress."""

    def build_steps(self, specs: list[PlanStepSpec]) -> list[Step]:
        if not specs:
            raise ValueError("A plan must contain at least one step.")
        ids = [spec.id.strip() for spec in specs]
        if any(not item for item in ids):
            raise ValueError("Every planned step must have a non-empty id.")
        if len(ids) != len(set(ids)):
            raise ValueError("Planned step ids must be unique.")
        if any(not spec.title.strip() for spec in specs):
            raise ValueError("Every planned step must have a non-empty title.")
        return [Step(id=spec.id, title=spec.title) for spec in specs]

    def apply(self, state: ProjectState, task_id: str, title: str, specs: list[PlanStepSpec]) -> TaskState:
        """Attach a new task while preserving any existing task progress."""
        if any(task.id == task_id for task in state.tasks):
            raise ValueError(f"Task already exists: {task_id}")
        task = TaskState(id=task_id, title=title, steps=self.build_steps(specs))
        state.tasks.append(task)
        if state.current_task_id is None:
            state.current_task_id = task.id
            state.phase = ProjectPhase.ACTIVE
        return task

    def next_step_id(self, task: TaskState) -> str | None:
        """Return the first pending step without mutating task state."""
        for step in task.steps:
            if step.status == StepStatus.PENDING:
                return step.id
        return None
