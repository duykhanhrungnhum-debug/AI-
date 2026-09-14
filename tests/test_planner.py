from ai_agent.core.planner import PlanStepSpec, Planner
from ai_agent.core.project_state import ProjectPhase, ProjectState, StepStatus


def test_planner_builds_ordered_steps():
    planner = Planner()
    steps = planner.build_steps([
        PlanStepSpec("s1", "First"),
        PlanStepSpec("s2", "Second"),
    ])
    assert [step.id for step in steps] == ["s1", "s2"]
    assert all(step.status == StepStatus.PENDING for step in steps)


def test_planner_rejects_duplicate_ids():
    planner = Planner()
    try:
        planner.build_steps([PlanStepSpec("s1", "First"), PlanStepSpec("s1", "Again")])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate step ids must be rejected")


def test_planner_applies_first_task_without_overwriting_existing_state():
    planner = Planner()
    state = ProjectState(id="p1", name="demo")
    task = planner.apply(
        state,
        "t1",
        "Build feature",
        [PlanStepSpec("s1", "Implement"), PlanStepSpec("s2", "Verify")],
    )
    assert state.current_task_id == "t1"
    assert state.phase == ProjectPhase.ACTIVE
    assert task.steps[0].status == StepStatus.PENDING
    assert planner.next_step_id(task) == "s1"
