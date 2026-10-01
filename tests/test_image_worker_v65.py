from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6
from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


def test_v65_worker_keeps_verified_v6_flux_call_unchanged():
    kwargs = {
        "base_url": "https://example.invalid",
        "worker_token": "test-token",
        "session_id": "test-session",
    }
    v6_source = WarmImageWorkerManagerV6(kernel_slug="ai-agent-image-warm-canary")._worker_source(**kwargs)
    v65_source = WarmImageWorkerManagerV65(kernel_slug="ai-agent-image-warm-canary")._worker_source(**kwargs)

    assert v65_source == v6_source
    assert "callback_on_step_end=_step_heartbeat" not in v65_source
    assert "def _step_heartbeat" not in v65_source
