from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


def test_v65_worker_injects_flux_step_heartbeat():
    manager = WarmImageWorkerManagerV65(kernel_slug="ai-agent-image-warm-canary")
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="test-token",
        session_id="test-session",
    )

    assert "callback_on_step_end=_step_heartbeat" in source
    assert "progress_stage = (" in source
    assert "set_state(progress_stage, job_id)" in source
    assert "signal(progress_stage, job_id)" in source
    assert "step{int(step_index) + 1}" in source
