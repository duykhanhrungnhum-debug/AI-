from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6
from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


def test_v65_generate_watchdog_allows_bounded_pre_denoise_warmup():
    assert WarmImageWorkerManagerV6.STAGE_TIMEOUTS["generate"] == 120.0
    assert WarmImageWorkerManagerV65.STAGE_TIMEOUTS["generate"] == 300.0
    assert WarmImageWorkerManagerV65.STAGE_TIMEOUTS["generate"] < 600.0
    assert WarmImageWorkerManagerV65.STAGE_TIMEOUTS["recaption"] == 180.0
    assert WarmImageWorkerManagerV65.STAGE_TIMEOUTS["upload"] == 90.0


def test_v65_default_job_lease_outlives_generate_watchdog_but_stays_bounded():
    manager = WarmImageWorkerManagerV65()
    assert manager.job_lease_seconds == 360
    assert manager.job_lease_seconds > manager.STAGE_TIMEOUTS["generate"]
    assert manager.job_lease_seconds < 600


def test_v65_explicit_job_lease_override_is_honored():
    manager = WarmImageWorkerManagerV65(job_lease_seconds=420)
    assert manager.job_lease_seconds == 420
