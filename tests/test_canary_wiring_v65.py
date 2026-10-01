import importlib

from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


def test_canary_v65_override_is_final(monkeypatch):
    monkeypatch.setenv("AIKA_WARM_SOAK_COUNT", "0")
    monkeypatch.setenv("AIKA_WARM_CANARY_SELFTEST", "false")

    module = importlib.import_module("ai_agent.chat_api_v6")
    module = importlib.reload(module)

    assert isinstance(module.CHAT_BROKER._warm_image, WarmImageWorkerManagerV65)
    assert module.CHAT_BROKER._warm_image.kernel_slug == "ai-agent-image-warm-canary"
