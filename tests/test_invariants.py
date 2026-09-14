from ai_agent.core.invariants import CORE_INVARIANTS, assert_core_invariants


def test_core_invariants_exist():
    assert_core_invariants()
    assert len(CORE_INVARIANTS) >= 7


def test_completion_requires_verification_evidence():
    from ai_agent.core.models import Evidence, TaskResult, TaskStatus

    result = TaskResult(
        status=TaskStatus.SUCCEEDED,
        summary="Completed",
        evidence=[Evidence(kind="claim", description="No verification", verified=False)],
    )

    assert result.is_verified is False
