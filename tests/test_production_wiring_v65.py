import importlib

from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


def test_production_api_uses_v65_warm_manager(monkeypatch):
    monkeypatch.setenv("AIKA_WARM_CANARY_SELFTEST", "false")

    module = importlib.import_module("ai_agent.chat_api")
    module = importlib.reload(module)

    assert isinstance(module.CHAT_BROKER._warm_image, WarmImageWorkerManagerV65)
    assert module.CHAT_BROKER._warm_image.kernel_slug == "ai-agent-image-warm"
