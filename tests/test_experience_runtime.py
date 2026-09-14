from ai_agent.core.experience import ExperienceStore
from ai_agent.core.experience_runtime import RuntimeExperienceRecorder
from ai_agent.core.model_runtime import ActionExecution


def test_runtime_success_is_recorded_with_evidence() -> None:
    store = ExperienceStore()
    recorder = RuntimeExperienceRecorder(store)
    item = recorder.record("learn tool use", "echo", ActionExecution(True, "done", ("output",)))
    assert item.success
    assert item.evidence == ["output"]
    assert store.all() == [item]


def test_runtime_failure_becomes_recovery_lesson() -> None:
    store = ExperienceStore()
    item = RuntimeExperienceRecorder(store).record(
        "recover tool failure", "broken", ActionExecution(False, "timeout")
    )
    assert not item.success
    assert "failure" in item.tags
    assert "recovery lesson" in item.lesson
