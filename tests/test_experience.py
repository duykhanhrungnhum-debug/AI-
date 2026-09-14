from ai_agent.core.experience import Experience, ExperienceStore


def test_records_success_and_failure_and_retrieves_relevant_experience():
    store = ExperienceStore(max_items=3)
    success = store.record(Experience(
        goal="learn HTTP retrieval",
        outcome="verified source",
        lesson="require provenance before verification",
        success=True,
        evidence=["source hash"],
    ))
    failure = store.record(Experience(
        goal="learn HTTP retrieval",
        outcome="verification failed",
        lesson="do not trust a single unsupported source",
        success=False,
    ))

    assert success.success is True
    assert failure.success is False
    relevant = store.relevant("HTTP retrieval")
    assert [item.id for item in relevant] == [failure.id, success.id]


def test_store_is_bounded():
    store = ExperienceStore(max_items=2)
    first = store.record(Experience("alpha goal", "failed", "retry", False))
    store.record(Experience("beta goal", "failed", "retry", False))
    third = store.record(Experience("gamma goal", "success", "reuse", True))

    assert [item.id for item in store.all()] == [store.all()[0].id, third.id]
    assert first not in store.all()
