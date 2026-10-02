import importlib
import inspect
from pathlib import Path


def test_production_api_uses_aika_owned_media_tools_only():
    module = importlib.import_module("ai_agent.chat_api")
    module = importlib.reload(module)
    source = inspect.getsource(module)

    assert "AGENT_BROKER" in source
    assert "MEDIA_BROKER" in source
    assert "route_skill" not in source
    assert "skill_hint" not in source
    assert "chat_session" not in source
    assert '"/v1/chat/image"' in source
    assert '"/v1/chat/video"' in source
    assert "WarmImageWorkerManager" not in source
    assert "/internal/image/" not in source


def test_legacy_media_architecture_is_physically_removed():
    legacy_paths = (
        "src/ai_agent/chat_api_v6.py",
        "src/ai_agent/chat_session.py",
        "src/ai_agent/canary_selftest.py",
        "src/ai_agent/canary_soak.py",
        "src/ai_agent/router/skill_router.py",
        "src/ai_agent/workers/image_manager.py",
        "src/ai_agent/workers/image_manager_v6.py",
        "src/ai_agent/workers/image_manager_v65.py",
        "src/ai_agent/workers/image_worker_source_v6.py",
    )
    for raw_path in legacy_paths:
        assert not Path(raw_path).exists(), raw_path
