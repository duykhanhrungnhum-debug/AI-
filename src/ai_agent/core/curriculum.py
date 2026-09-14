"""Capability curriculum for training and evaluating the autonomous agent."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingExercise:
    id: str
    domain: str
    objective: str
    difficulty: int = 1
    prerequisite_ids: tuple[str, ...] = ()


DEFAULT_CURRICULUM: tuple[TrainingExercise, ...] = (
    TrainingExercise("instruction-001", "instruction_following", "Follow a valid multi-constraint user instruction exactly."),
    TrainingExercise("truth-001", "truthfulness", "Separate known, unknown, inferred, and verified claims."),
    TrainingExercise("reasoning-001", "reasoning", "Solve a multi-step logic problem and expose assumptions."),
    TrainingExercise("math-001", "mathematics", "Solve and independently check a quantitative problem."),
    TrainingExercise("code-001", "programming", "Implement a small feature with tests and evidence."),
    TrainingExercise("debug-001", "debugging", "Diagnose a failing program without inventing execution results."),
    TrainingExercise("research-001", "internet_research", "Research a topic from multiple Internet sources and preserve provenance."),
    TrainingExercise("verify-001", "verification", "Detect a claim that lacks sufficient evidence."),
    TrainingExercise("conflict-001", "conflict_resolution", "Retain and investigate contradictory sources instead of overwriting one."),
    TrainingExercise("plan-001", "planning", "Decompose a long task into ordered resumable steps."),
    TrainingExercise("recovery-001", "recovery", "Recover from a failed step without repeating an identical failed action forever."),
    TrainingExercise("state-001", "long_horizon", "Resume a task from its durable checkpoint after interruption."),
    TrainingExercise("tools-001", "tool_use", "Use a tool, record its output, and distinguish output from verified conclusions."),
    TrainingExercise("self-eval-001", "self_evaluation", "Review an outcome and identify unsupported claims or missing evidence."),
    TrainingExercise("learning-001", "learning_from_failure", "Extract a reusable lesson from a failed attempt without weakening invariants."),
)


def get_curriculum() -> tuple[TrainingExercise, ...]:
    """Return the stable ordered baseline curriculum."""
    return DEFAULT_CURRICULUM
