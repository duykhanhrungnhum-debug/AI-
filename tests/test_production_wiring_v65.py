import importlib
import inspect


def test_production_api_uses_simple_media_tools_only():
    module = importlib.import_module("ai_agent.chat_api")
    module = importlib.reload(module)
    source = inspect.getsource(module)

    assert "MEDIA_BROKER" in source
    assert "route_skill" in source
    assert '"/v1/chat/image"' in source
    assert '"/v1/chat/video"' in source

    # Production media must not grow warm/cold orchestration, worker leases,
    # benchmark hooks, or self-tests again. AIKA only calls a media tool.
    assert "WarmImageWorkerManager" not in source
    assert "/internal/image/" not in source
    assert "subject_lock_benchmark" not in source
    assert "AIKA_WARM_CANARY_SELFTEST" not in source
    assert "AIKA_COLD_IMAGE_SELFTEST" not in source
    assert "run_warm_canary_selftest" not in source
    assert "run_cold_image_selftest" not in source
