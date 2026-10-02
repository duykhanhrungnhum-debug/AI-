import importlib
import inspect


def test_production_api_does_not_wire_warm_v65_or_start_selftests():
    module = importlib.import_module("ai_agent.chat_api")
    module = importlib.reload(module)
    source = inspect.getsource(module)

    assert "WarmImageWorkerManagerV65" not in source
    assert "AIKA_WARM_CANARY_SELFTEST" not in source
    assert "AIKA_COLD_IMAGE_SELFTEST" not in source
    assert "run_warm_canary_selftest" not in source
    assert "run_cold_image_selftest" not in source
