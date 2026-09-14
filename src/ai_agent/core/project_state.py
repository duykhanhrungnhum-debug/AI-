"""Durable project/task state for long-running agent work."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path


class ProjectPhase(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class Step:
    id: str
    title: str
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    result: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class TaskState:
    id: str
    title: str
    steps: list[Step] = field(default_factory=list)
    current_step_id: str | None = None
    status: StepStatus = StepStatus.PENDING
    blocked_reason: str | None = None

    def current_step(self) -> Step | None:
        if self.current_step_id is None:
            return None
        return next((step for step in self.steps if step.id == self.current_step_id), None)

    def advance_to_next_pending(self) -> Step | None:
        """Move only to the next pending step; never silently jump backward."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                self.current_step_id = step.id
                step.status = StepStatus.RUNNING
                self.status = StepStatus.RUNNING
                return step
        return None


@dataclass
class ProjectState:
    id: str
    name: str
    phase: ProjectPhase = ProjectPhase.PLANNED
    tasks: list[TaskState] = field(default_factory=list)
    current_task_id: str | None = None
    version: int = 1
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def current_task(self) -> TaskState | None:
        if self.current_task_id is None:
            return None
        return next((task for task in self.tasks if task.id == self.current_task_id), None)

    def checkpoint(self) -> None:
        """Update the durable checkpoint timestamp/version."""
        self.version += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()


class StateStore:
    """Small JSON-backed store; replaceable later by a database."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, state: ProjectState) -> None:
        state.checkpoint()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")

    def load(self) -> ProjectState:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        tasks = []
        for task_data in data["tasks"]:
            steps = [Step(**step) for step in task_data["steps"]]
            tasks.append(TaskState(**{**task_data, "steps": steps}))
        return ProjectState(**{**data, "tasks": tasks})
