"""Model-backed runtime that separates reasoning, execution, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .engine import ExecutionEngine
from .invariants import assert_core_invariants
from .model_agent import ModelAgent, ModelDecision
from .project_state import ProjectState


@dataclass(frozen=True)
class ActionExecution:
    """Result returned by an executor; evidence is independently checkable."""

    success: bool
    outcome: str
    evidence: tuple[str, ...] = ()


class ActionExecutor(Protocol):
    """Execute a model proposal using real tools or a controlled adapter."""

    def execute(self, decision: ModelDecision) -> ActionExecution:
        """Run the proposed action and return observable outcome/evidence."""


class ModelRuntime:
    """Connect model reasoning to execution while keeping verification external."""

    def __init__(self, model_agent: ModelAgent, executor: ActionExecutor, engine: ExecutionEngine):
        self.model_agent = model_agent
        self.executor = executor
        self.engine = engine

    def run_current_step(self, state: ProjectState) -> ModelDecision:
        """Reason, execute, checkpoint, and leave verification to the engine."""
        assert_core_invariants()
        task = state.current_task()
        if task is None:
            raise ValueError("Cannot run without a current task.")
        step = task.current_step()
        if step is None:
            self.engine.begin_current_step(state)
            step = task.current_step()
        if step is None:
            raise ValueError("No unfinished step remains.")

        decision = self.model_agent.decide(task.title, step.id, step.title)
        result = self.executor.execute(decision)
        self.engine.record_step_result(
            state,
            action=decision.response.text,
            outcome=result.outcome,
            success=result.success,
            evidence="\n".join(result.evidence) if result.evidence else None,
        )
        return decision
