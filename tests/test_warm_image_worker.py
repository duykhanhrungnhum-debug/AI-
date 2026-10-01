import base64
import inspect
import time

import pytest

from ai_agent.canary_selftest import _flag
from ai_agent.chat_session import ChatSessionBroker
from ai_agent.workers.image_manager import WarmImageWorkerManager, WorkerState


def test_warm_worker_is_off_by_default_and_tracks_leases(monkeypatch):
    monkeypatch.delenv("AIKA_IMAGE_WARM_WORKER", raising=False)
    manager = WarmImageWorkerManager()
    snapshot = manager.snapshot()
    assert snapshot.enabled is False
    assert snapshot.launch_count == 0
    assert snapshot.state == "offline"
    assert snapshot.stage == "offline"
    assert snapshot.startup_started_at == 0.0
    assert snapshot.session_id == ""
    assert snapshot.current_job_id == ""
    assert snapshot.lease_expires_at == 0.0
    assert snapshot.job_lease_expires_at == 0.0


def test_warm_worker_source_serializes_qwen_then_flux_and_has_independent_heartbeat():
    manager = WarmImageWorkerManager(idle_seconds=300)
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="test-token",
        session_id="session-1",
    )
    assert source.count("Flux2KleinPipeline.from_pretrained") == 1
    assert source.count("AutoModelForCausalLM.from_pretrained") == 1
    assert source.count("AutoTokenizer.from_pretrained") == 1
    assert "except (ImportError, AttributeError):" in source
    assert 'sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"' in source
    assert '"diffusers", "transformers>=4.57,<5", "accelerate<2", "safetensors"' in source
    assert source.index("try:\n    import torch") < source.index("def request(")
    assert source.index("from diffusers import Flux2KleinPipeline") < source.index("def request(")
    assert source.index("Flux2KleinPipeline.from_pretrained") < source.index("def request(")
    assert source.index("pipe.enable_model_cpu_offload()") < source.index("def request(")
    assert source.index("def render_recaption") < source.index("AutoTokenizer.from_pretrained")
    assert 'device_map="auto"' in source
    assert 'recaptioner.to("cuda")' not in source
    assert 'recaptioner.to("cpu")' not in source
    assert source.index("del recaptioner") < source.index("image = pipe(")
    assert '"X-AIKA-Worker-Session": CONFIG["session_id"]' in source
    assert '"heartbeat_seconds": 10' in source
    assert "def heartbeat_loop():" in source
    assert "threading.Thread(target=heartbeat_loop" in source
    assert 'set_state("busy", job_id)' in source
    assert source.count('signal("busy", job_id)') >= 6
    assert 'set_state("ready", "")' in source
    assert 'request("GET", "/internal/image/pull", timeout=30)' in source
    assert '"image_total": len(prompts)' in source
    assert '"image_index": index' in source
    assert "while time.monotonic() - idle_started < float(CONFIG[\"idle_seconds\"]):" in source


def test_default_job_lease_covers_non_preemptible_gpu_sections():
    manager = WarmImageWorkerManager()
    assert manager.job_lease_seconds == 180
    assert manager.job_lease_seconds > manager.worker_lease_seconds


def test_warm_launch_title_is_collision_safe_and_counted_after_submit():
    source = inspect.getsource(WarmImageWorkerManager._launch)
    assert 'launch_title = f"AIKA Warm Image Worker {time.time_ns()}"' in source
    assert "session_id=session_id" in source
    submit_pos = source.index("worker.submit_script(")
    count_pos = source.index("self._launch_count += 1")
    assert submit_pos < count_pos


def test_stale_startup_becomes_error_and_opens_circuit(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager(startup_timeout_seconds=10, circuit_cooldown_seconds=60)
    with manager._lock:
        manager._session_id = "session-a"
        manager._state = WorkerState.STARTING
        manager._stage = "starting"
        manager._startup_started_at = time.time() - 11
    snapshot = manager.snapshot()
    assert snapshot.state == "error"
    assert snapshot.stage == "startup_timeout"
    assert snapshot.failures == 1
    assert snapshot.circuit_open_until > time.time()
    assert snapshot.startup_started_at == 0.0
    assert snapshot.session_id == ""
    assert "exceeded 10 seconds" in manager.last_error()
    assert manager.ensure_started() is False


def test_ready_heartbeat_establishes_session_lease(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager(worker_lease_seconds=30)
    with manager._lock:
        manager._session_id = "session-a"
        manager._state = WorkerState.STARTING
        manager._startup_started_at = time.time()
    assert manager.heartbeat("session-a", "ready") is True
    snapshot = manager.snapshot()
    assert snapshot.state == "ready"
    assert snapshot.session_id == "session-a"
    assert snapshot.startup_started_at == 0.0
    assert snapshot.lease_expires_at > time.time()
    assert snapshot.failures == 0


def test_stale_worker_heartbeat_is_rejected(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager()
    with manager._lock:
        manager._session_id = "session-a"
        manager._state = WorkerState.READY
        manager._lease_expires_at = time.time() + 30
    assert manager.heartbeat("session-b", "ready") is False
    assert manager.snapshot().session_id == "session-a"


def test_busy_job_lease_prevents_relaunch_when_worker_heartbeat_age_is_long(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager(worker_lease_seconds=30, job_lease_seconds=180)
    now = time.time()
    with manager._lock:
        manager._session_id = "session-a"
        manager._state = WorkerState.BUSY
        manager._stage = "busy"
        manager._last_seen = now - 120
        manager._lease_expires_at = now - 1
        manager._current_job_id = "job-a"
        manager._job_lease_expires_at = now + 120
    assert manager.ensure_started() is True
    snapshot = manager.snapshot()
    assert snapshot.session_id == "session-a"
    assert snapshot.current_job_id == "job-a"
    assert snapshot.state == "busy"
    assert snapshot.launch_count == 0


def test_busy_heartbeat_renews_worker_and_job_leases(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager(worker_lease_seconds=30, job_lease_seconds=180)
    with manager._lock:
        manager._session_id = "session-a"
        manager._state = WorkerState.READY
        manager._lease_expires_at = time.time() + 5
    assert manager.claim_job("session-a", "job-a") is True
    before = manager.snapshot()
    time.sleep(0.01)
    assert manager.heartbeat("session-a", "busy", current_job_id="job-a") is True
    after = manager.snapshot()
    assert after.state == "busy"
    assert after.current_job_id == "job-a"
    assert after.lease_expires_at >= before.lease_expires_at
    assert after.job_lease_expires_at >= before.job_lease_expires_at
    assert manager.job_lease_alive("job-a") is True


def test_second_job_waits_for_same_session_and_stale_result_is_rejected(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    broker = ChatSessionBroker()
    assert broker._warm_image.heartbeat("session-a", "ready") is True

    first = broker.create_job("AIKA tạo ảnh quả táo", message="AIKA tạo ảnh quả táo")
    pulled_first = broker.pull_image_job("session-a")
    assert pulled_first is not None
    assert pulled_first["job_id"] == first.job_id

    second = broker.create_job("AIKA tạo ảnh quả cam", message="AIKA tạo ảnh quả cam")
    assert broker._warm_image.snapshot().launch_count == 0
    assert broker.pull_image_job("session-a") is None

    png = base64.b64encode(b"\x89PNG\r\n\x1a\nlease-test").decode("ascii")
    with pytest.raises(PermissionError):
        broker.finish_image_job(
            {"job_id": first.job_id, "image_b64": png},
            session_id="session-stale",
        )

    broker.finish_image_job(
        {
            "job_id": first.job_id,
            "provider": "kaggle-image-warm",
            "model": "test-model",
            "image_b64": png,
            "image_index": 0,
            "image_total": 1,
        },
        session_id="session-a",
    )
    assert broker.get_job(first.job_id)["status"] == "done"
    pulled_second = broker.pull_image_job("session-a")
    assert pulled_second is not None
    assert pulled_second["job_id"] == second.job_id
    assert broker._warm_image.snapshot().session_id == "session-a"
    assert broker._warm_image.snapshot().launch_count == 0


def test_startup_error_heartbeat_sets_manager_error(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManager()
    with manager._lock:
        manager._session_id = "session-a"
        manager._state = WorkerState.STARTING
        manager._startup_started_at = time.time()
    assert manager.heartbeat(
        "session-a",
        "error:models:RuntimeError: model load failed",
    ) is True
    snapshot = manager.snapshot()
    assert snapshot.state == "error"
    assert snapshot.stage.startswith("error:models:")
    assert snapshot.failures == 1
    assert snapshot.last_seen > 0
    assert snapshot.session_id == ""
    assert manager.last_error() == "models:RuntimeError: model load failed"


def test_canary_flag_is_explicit(monkeypatch):
    monkeypatch.delenv("AIKA_WARM_CANARY_SELFTEST", raising=False)
    assert _flag("AIKA_WARM_CANARY_SELFTEST") is False
    monkeypatch.setenv("AIKA_WARM_CANARY_SELFTEST", "true")
    assert _flag("AIKA_WARM_CANARY_SELFTEST") is True
