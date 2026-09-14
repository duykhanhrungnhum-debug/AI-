from ai_agent.core.engine import ExecutionEngine
from ai_agent.core.project_state import ProjectState, StateStore, Step, StepStatus, TaskState
from ai_agent.core.recovery import RecoveryGuard


def make_state():
    return ProjectState(
        id="p1",
        name="demo",
        current_task_id="t1",
        tasks=[TaskState(id="t1", title="Build", steps=[Step("s1", "Implement")])],
    )


def test_failed_attempts_survive_restart(tmp_path):
    store = StateStore(tmp_path / "state.json")
    state = make_state()
    engine = ExecutionEngine(store, RecoveryGuard())

    engine.begin_current_step(state)
    engine.record_step_result(state, "run-tool", "error", False)
    loaded = store.load()

    restarted = ExecutionEngine(store, RecoveryGuard())
    restarted.resume(loaded)
    restarted.record_step_result(loaded, "run-tool", "error", False)

    assert loaded.current_task().current_step().status is StepStatus.BLOCKED
    assert loaded.current_task().status is StepStatus.BLOCKED
    assert len(loaded.current_task().current_step().history) == 2
