import pytest

from ai_agent.core.curriculum import TrainingExercise
from ai_agent.core.experience import ExperienceStore
from ai_agent.core.training import CurriculumRunner, EvaluationResult


def test_training_scores_retries_and_advances():
    exercises = (
        TrainingExercise("a", "demo", "pass a"),
        TrainingExercise("b", "demo", "pass b", prerequisite_ids=("a",)),
    )
    store = ExperienceStore()
    calls = iter([
        EvaluationResult(0.4, False, "insufficient evidence"),
        EvaluationResult(0.9, True, "evidence is sufficient", ("test output",)),
        EvaluationResult(0.95, True, "passed", ("verified output",)),
    ])
    runner = CurriculumRunner(curriculum=exercises, evaluator=lambda exercise, response: next(calls), experience_store=store)

    assert runner.next_exercise().id == "a"
    first = runner.run(exercises[0], "first")
    assert not first.result.passed
    assert runner.next_exercise().id == "a"
    second = runner.run(exercises[0], "second")
    assert second.result.passed
    assert runner.next_exercise().id == "b"
    runner.run(exercises[1], "third")
    assert runner.completion_ratio() == 1.0
    assert len(store.all()) == 3


def test_prerequisites_and_attempt_limit_are_enforced():
    exercise = TrainingExercise("b", "demo", "pass b", prerequisite_ids=("a",))
    runner = CurriculumRunner(curriculum=(exercise,), evaluator=lambda e, r: EvaluationResult(0.1, False, "fail"), max_attempts=1)
    with pytest.raises(ValueError):
        runner.run(exercise, "answer")
