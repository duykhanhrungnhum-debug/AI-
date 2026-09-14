from ai_agent.core.project_state import ProjectPhase, ProjectState, StateStore, Step, StepStatus, TaskState


def test_state_store_round_trips_enums_and_progress(tmp_path):
    store = StateStore(tmp_path / "state.json")
    state = ProjectState(
        id="p1",
        name="demo",
        phase=ProjectPhase.ACTIVE,
        current_task_id="t1",
        tasks=[
            TaskState(
                id="t1",
                title="Build",
                current_step_id="s1",
                status=StepStatus.RUNNING,
                steps=[Step("s1", "Implement", StepStatus.RUNNING, attempts=1, result="working", evidence=["checkpoint"])],
            )
        ],
    )
    store.save(state)

    loaded = store.load()
    assert loaded.phase is ProjectPhase.ACTIVE
    assert loaded.current_task_id == "t1"
    assert loaded.current_task().status is StepStatus.RUNNING
    assert loaded.current_task().current_step().status is StepStatus.RUNNING
    assert loaded.current_task().current_step().evidence == ["checkpoint"]
