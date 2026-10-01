from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6
from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


def test_v65_uses_verified_v6_watchdog_and_lease_defaults():
    manager = WarmImageWorkerManagerV65()
    assert manager.STAGE_TIMEOUTS == WarmImageWorkerManagerV6.STAGE_TIMEOUTS
    assert manager.STAGE_TIMEOUTS["generate"] == 120.0
    assert manager.STAGE_TIMEOUTS["recaption"] == 180.0
    assert manager.STAGE_TIMEOUTS["upload"] == 90.0
    assert manager.job_lease_seconds == 180
    assert manager.STAGE_TIMEOUTS["generate"] < manager.job_lease_seconds < 600


def test_v65_explicit_job_lease_override_is_honored():
    manager = WarmImageWorkerManagerV65(job_lease_seconds=420)
    assert manager.job_lease_seconds == 420
