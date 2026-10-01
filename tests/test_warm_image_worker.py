import inspect
import time

from ai_agent.canary_selftest import _flag
from ai_agent.workers.image_manager import WarmImageWorkerManager, WorkerState


def test_warm_worker_is_off_by_default_and_tracks_launches(monkeypatch):
    monkeypatch.delenv("AIKA_IMAGE_WARM_WORKER", raising=False)
    manager = WarmImageWorkerManager()
    snapshot = manager.snapshot()
    assert snapshot.enabled is False
    assert snapshot.launch_count == 0
    assert snapshot.state == "offline"
    assert snapshot.stage == "offline"
    assert snapshot.startup_started_at == 0.0


def test_warm_worker_source_matches_verified_cold_bootstrap_and_reuses_models():
    manager = WarmImageWorkerManager(idle_seconds=300)
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="test-token",
    )
    assert source.count("Flux2KleinPipeline.from_pretrained") == 1
    assert source.count("AutoModelForCausalLM.from_pretrained") == 1
    assert source.count("AutoTokenizer.from_pretrained") == 1
    assert "except (ImportError, AttributeError):" in source
    assert 'sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"' in source
    assert '"diffusers", "transformers>=4.57,<5", "accelerate<2", "safetensors"' in source
    assert 'pipe.enable_model_cpu_offload()' in source
    assert 'recaptioner.to("cuda")' in source
    assert 'recaptioner.to("cpu")' in source
    assert 'request("GET", "/internal/image/pull", timeout=30)' in source
    assert '"image_total": len(prompts)' in source
    assert '"image_index": index' in source
    assert 'timeout=180' in source
    assert 'signal("booting")' in source
    assert 'signal("dependencies_ready")' in source
    assert 'signal("recaption_model_loading")' in source
    assert 'signal("recaption_model_ready")' in source
    assert 'signal("image_model_loading")' in source
    assert 'signal("ready")' in source
    assert "while time.monotonic() - idle_started < float(CONFIG[\"idle_seconds\"]):" in source


def test_warm_launch_title_is_collision_safe_and_counted_after_submit():
    source = inspect.getsource(WarmImageWorkerManager._launch)
    assert 'launch_title = f"AIKA Warm Image Worker {time.time_ns()}"' in source
    submit_pos = source.index("worker.submit_script(")
    count_pos = source.index("self._launch_count += 1")
    assert submit_pos < count_pos


def test_stale_startup_becomes_error_and_opens_circuit(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager(startup_timeout_seconds=10, circuit_cooldown_seconds=60)
    with manager._lock:
        manager._state = WorkerState.STARTING
        manager._stage = "booting"
        manager._startup_started_at = time.time() - 11
    snapshot = manager.snapshot()
    assert snapshot.state == "error"
    assert snapshot.stage == "startup_timeout"
    assert snapshot.failures == 1
    assert snapshot.circuit_open_until > time.time()
    assert snapshot.startup_started_at == 0.0
    assert "exceeded 10 seconds" in manager.last_error()
    assert manager.ensure_started() is False


def test_ready_heartbeat_clears_startup_clock(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager()
    with manager._lock:
        manager._state = WorkerState.STARTING
        manager._startup_started_at = time.time()
    manager.heartbeat("ready")
    snapshot = manager.snapshot()
    assert snapshot.state == "ready"
    assert snapshot.startup_started_at == 0.0
    assert snapshot.failures == 0


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
