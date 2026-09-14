from ai_agent.core.project_state import (
    ProjectState,
    StateStore,
    Step,
    StepStatus,
    TaskState,
)


def test_resume_keeps_exact_current_step(tmp_path):
    state = ProjectState(
        id="p1",
        name="Long project",
        tasks=[
            TaskState(
                id="t1",
                title="Task",
                steps=[Step("s1", "First", StepStatus.VERIFIED), Step("s2", "Second", StepStatus.RUNNING)],
                current_step_id="s2",
            )
        ],
        current_task_id="t1",
    )
    store = StateStore(tmp_path / "state.json")
    store.save(state)

    loaded = store.load()
    assert loaded.current_task().current_step().id == "s2"
    assert loaded.current_task().current_step().status == StepStatus.RUNNING


def test_checkpoint_increments_version(tmp_path):
    state = ProjectState(id="p1", name="Project")
    before = state.version
    StateStore(tmp_path / "state.json").save(state)
    assert state.version == before + 1
