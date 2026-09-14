from ai_agent.core.project_state import Step, StepStatus, TaskState
from ai_agent.core.recovery import RecoveryGuard, resume_task


def test_recovery_resumes_current_step():
    task = TaskState(
        id="t1",
        title="Long task",
        steps=[Step("s1", "Done", StepStatus.VERIFIED), Step("s2", "Continue", StepStatus.RUNNING)],
        current_step_id="s2",
        status=StepStatus.RUNNING,
    )
    assert resume_task(task, RecoveryGuard()) == "Resume step s2: Continue"


def test_identical_failure_eventually_stops_retry():
    guard = RecoveryGuard(max_identical_failures=2)
    guard.record("s1", "same-action", "failed", False)
    assert guard.may_retry("s1", "same-action")
    guard.record("s1", "same-action", "failed again", False)
    assert not guard.may_retry("s1", "same-action")
