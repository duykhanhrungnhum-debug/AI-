from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6
from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


def test_v65_generate_watchdog_allows_bounded_pre_denoise_warmup():
    assert WarmImageWorkerManagerV6.STAGE_TIMEOUTS["generate"] == 120.0
    assert WarmImageWorkerManagerV65.STAGE_TIMEOUTS["generate"] == 300.0
    assert WarmImageWorkerManagerV65.STAGE_TIMEOUTS["generate"] < 600.0
    assert WarmImageWorkerManagerV65.STAGE_TIMEOUTS["recaption"] == 180.0
    assert WarmImageWorkerManagerV65.STAGE_TIMEOUTS["upload"] == 90.0
