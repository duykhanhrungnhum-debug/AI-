from ai_agent.core.engine import ExecutionEngine
from ai_agent.core.project_state import ProjectState, StateStore, Step, TaskState, StepStatus
from ai_agent.core.recovery import RecoveryGuard


def make_state():
    task = TaskState(
        id="task-1",
        title="Example long task",
        steps=[Step("step-1", "First"), Step("step-2", "Second")],
        current_step_id="step-1",
    )
    return ProjectState(id="project-1", name="Test", tasks=[task], current_task_id="task-1")


def test_engine_records_verified_step(tmp_path):
    store = StateStore(tmp_path / "state.json")
    engine = ExecutionEngine(store, RecoveryGuard())
    state = make_state()

    engine.begin_current_step(state)
    engine.record_step_result(state, "run", "done", True, "test evidence")

    assert state.current_task().current_step().status is StepStatus.VERIFIED
    assert state.current_task().current_step().evidence == ["test evidence"]
    assert store.load().version > 1


def test_engine_blocks_repeated_identical_failures(tmp_path):
    store = StateStore(tmp_path / "state.json")
    engine = ExecutionEngine(store, RecoveryGuard(max_identical_failures=2))
    state = make_state()

    engine.begin_current_step(state)
    engine.record_step_result(state, "run", "failed", False)
    state.current_task().current_step().status = StepStatus.RUNNING
    engine.record_step_result(state, "run", "failed again", False)

    assert state.current_task().current_step().status is StepStatus.BLOCKED
    assert state.current_task().status is StepStatus.BLOCKED
