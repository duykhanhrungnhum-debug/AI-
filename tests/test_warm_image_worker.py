from ai_agent.canary_selftest import _flag
from ai_agent.workers.image_manager import WarmImageWorkerManager


def test_warm_worker_is_off_by_default_and_tracks_launches(monkeypatch):
    monkeypatch.delenv("AIKA_IMAGE_WARM_WORKER", raising=False)
    manager = WarmImageWorkerManager()
    snapshot = manager.snapshot()
    assert snapshot.enabled is False
    assert snapshot.launch_count == 0
    assert snapshot.state == "offline"


def test_warm_worker_source_loads_models_once_and_processes_many_jobs():
    manager = WarmImageWorkerManager(idle_seconds=300)
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="test-token",
    )
    assert source.count("Flux2KleinPipeline.from_pretrained") == 1
    assert source.count("AutoModelForCausalLM.from_pretrained") == 1
    assert "while time.monotonic() - idle_started < float(CONFIG[\"idle_seconds\"]):" in source
    assert 'request("GET", "/internal/image/pull")' in source
    assert '"image_total": len(prompts)' in source
    assert '"image_index": index' in source


def test_canary_flag_is_explicit(monkeypatch):
    monkeypatch.delenv("AIKA_WARM_CANARY_SELFTEST", raising=False)
    assert _flag("AIKA_WARM_CANARY_SELFTEST") is False
    monkeypatch.setenv("AIKA_WARM_CANARY_SELFTEST", "true")
    assert _flag("AIKA_WARM_CANARY_SELFTEST") is True
