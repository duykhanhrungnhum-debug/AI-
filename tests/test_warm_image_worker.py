import inspect

from ai_agent.canary_selftest import _flag
from ai_agent.workers.image_manager import WarmImageWorkerManager


def test_warm_worker_is_off_by_default_and_tracks_launches(monkeypatch):
    monkeypatch.delenv("AIKA_IMAGE_WARM_WORKER", raising=False)
    manager = WarmImageWorkerManager()
    snapshot = manager.snapshot()
    assert snapshot.enabled is False
    assert snapshot.launch_count == 0
    assert snapshot.state == "offline"
    assert snapshot.stage == "offline"


def test_warm_worker_source_supervises_one_persistent_model_child():
    manager = WarmImageWorkerManager(idle_seconds=300)
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="test-token",
    )
    assert source.count("Flux2KleinPipeline.from_pretrained") == 1
    assert source.count("AutoModelForCausalLM.from_pretrained") == 1
    assert "aika_warm_child.py" in source
    assert "aika_warm_stage" in source
    assert "aika_warm_ready" in source
    assert "start_new_session=True" in source
    assert "os.killpg(proc.pid, signal_module.SIGKILL)" in source
    assert 'signal("repairing_dependencies")' in source
    assert 'timeout=300, label="dependency repair"' in source
    assert 'for attempt in range(2):' in source
    assert 'stage timeout: {current_stage}' in source
    assert 'if stage == "loading_models":' in source
    assert 'return 360' in source
    assert "while time.monotonic() - idle_started < float(CONFIG[\"idle_seconds\"]):" in source
    assert 'request("GET", "/internal/image/pull", timeout=30)' in source
    assert '"image_total": len(prompts)' in source
    assert '"image_index": index' in source
    assert 'timeout=180' in source
    assert 'signal("importing_torch")' in source
    assert 'signal("torch_ready")' in source
    assert 'signal("importing_diffusers")' in source
    assert 'signal("diffusers_ready")' in source
    assert 'signal("importing_transformers")' in source
    assert 'signal("transformers_ready")' in source
    assert 'signal("loading_models")' in source
    assert 'signal("ready")' in source


def test_warm_launch_title_is_collision_safe_and_counted_after_submit():
    source = inspect.getsource(WarmImageWorkerManager._launch)
    assert 'launch_title = f"AIKA Warm Image Worker {time.time_ns()}"' in source
    submit_pos = source.index("worker.submit_script(")
    count_pos = source.index("self._launch_count += 1")
    assert submit_pos < count_pos


def test_startup_error_heartbeat_sets_manager_error(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager()
    manager.heartbeat("error:models:RuntimeError: model load failed")
    snapshot = manager.snapshot()
    assert snapshot.state == "error"
    assert snapshot.stage.startswith("error:models:")
    assert snapshot.failures == 1
    assert snapshot.last_seen > 0
    assert manager.last_error() == "models:RuntimeError: model load failed"


def test_canary_flag_is_explicit(monkeypatch):
    monkeypatch.delenv("AIKA_WARM_CANARY_SELFTEST", raising=False)
    assert _flag("AIKA_WARM_CANARY_SELFTEST") is False
    monkeypatch.setenv("AIKA_WARM_CANARY_SELFTEST", "true")
    assert _flag("AIKA_WARM_CANARY_SELFTEST") is True
