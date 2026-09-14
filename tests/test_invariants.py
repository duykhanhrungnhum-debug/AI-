from ai_agent.core.invariants import CORE_INVARIANTS, assert_core_invariants


def test_core_invariants_exist():
    assert_core_invariants()
    assert len(CORE_INVARIANTS) == 7


def test_core_invariants_cannot_be_silently_weakened():
    import ai_agent.core.invariants as invariants

    original = invariants.CORE_INVARIANTS
    try:
        invariants.CORE_INVARIANTS = original[:-1]
        try:
            invariants.assert_core_invariants()
        except RuntimeError:
            pass
        else:
            raise AssertionError("modified core invariants were accepted")
    finally:
        invariants.CORE_INVARIANTS = original

    invariants.assert_core_invariants()


def test_completion_requires_verification_evidence():
    from ai_agent.core.models import Evidence, TaskResult, TaskStatus

    result = TaskResult(
        status=TaskStatus.SUCCEEDED,
        summary="Completed",
        evidence=[Evidence(kind="claim", description="No verification", verified=False)],
    )

    assert result.is_verified is False
