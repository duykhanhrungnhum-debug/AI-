"""Executable curriculum runner with scoring, progression, retries, and failure memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .curriculum import TrainingExercise, get_curriculum
from .experience import Experience, ExperienceStore
from .invariants import assert_core_invariants


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    passed: bool
    feedback: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if not self.feedback.strip():
            raise ValueError("feedback is required")
        if self.passed and self.score < 0.8:
            raise ValueError("passed evaluations require score >= 0.8")


@dataclass
class TrainingRecord:
    exercise_id: str
    attempt: int
    result: EvaluationResult


@dataclass
class TrainingState:
    records: list[TrainingRecord] = field(default_factory=list)

    def best_score(self, exercise_id: str) -> float:
        scores = [r.result.score for r in self.records if r.exercise_id == exercise_id]
        return max(scores, default=0.0)

    def passed(self, exercise_id: str) -> bool:
        return any(r.exercise_id == exercise_id and r.result.passed for r in self.records)


Evaluator = Callable[[TrainingExercise, str], EvaluationResult]


class CurriculumRunner:
    """Run curriculum exercises without allowing training to alter core invariants."""

    def __init__(self, *, curriculum: tuple[TrainingExercise, ...] | None = None,
                 evaluator: Evaluator | None = None,
                 experience_store: ExperienceStore | None = None,
                 state: TrainingState | None = None,
                 max_attempts: int = 3) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.curriculum = curriculum or get_curriculum()
        self.evaluator = evaluator
        self.experience_store = experience_store or ExperienceStore()
        self.state = state or TrainingState()
        self.max_attempts = max_attempts

    def eligible(self) -> list[TrainingExercise]:
        assert_core_invariants()
        return [
            exercise for exercise in self.curriculum
            if not self.state.passed(exercise.id)
            and all(self.state.passed(prerequisite) for prerequisite in exercise.prerequisite_ids)
        ]

    def run(self, exercise: TrainingExercise, response: str) -> TrainingRecord:
        assert_core_invariants()
        if self.evaluator is None:
            raise RuntimeError("an evaluator is required to execute curriculum training")
        if exercise not in self.curriculum:
            raise ValueError("exercise is not part of this curriculum")
        if any(not self.state.passed(prerequisite) for prerequisite in exercise.prerequisite_ids):
            raise ValueError("exercise prerequisites are not complete")
        previous_attempts = sum(1 for r in self.state.records if r.exercise_id == exercise.id)
        if previous_attempts >= self.max_attempts:
            raise RuntimeError("maximum attempts reached for exercise")

        result = self.evaluator(exercise, response)
        record = TrainingRecord(exercise.id, previous_attempts + 1, result)
        self.state.records.append(record)
        if result.passed:
            self.experience_store.record(Experience(
                goal=exercise.objective, outcome="passed", lesson=result.feedback, success=True,
                evidence=list(result.evidence), tags=["training", exercise.domain],
            ))
        else:
            self.experience_store.record(Experience(
                goal=exercise.objective, outcome="failed", lesson=result.feedback, success=False,
                evidence=list(result.evidence), tags=["training", exercise.domain],
            ))
        return record

    def next_exercise(self) -> TrainingExercise | None:
        eligible = self.eligible()
        return eligible[0] if eligible else None

    def completion_ratio(self) -> float:
        assert_core_invariants()
        if not self.curriculum:
            return 1.0
        return sum(self.state.passed(exercise.id) for exercise in self.curriculum) / len(self.curriculum)
