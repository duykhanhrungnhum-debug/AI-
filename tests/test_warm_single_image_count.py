from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6


def test_warm_worker_defaults_to_one_image_unless_multi_is_structural():
    source = WarmImageWorkerManagerV6()._worker_source(
        base_url="https://example.invalid",
        worker_token="token",
        session_id="session-v6",
    )
    assert "expected_count = requested_count(command)" in source
    assert "elif expected_count is None or expected_count == 1:" in source
    assert "prompts = [render_one_recaption(command)]" in source
    assert "prompts = render_batch_recaption(command)" in source
