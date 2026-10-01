import time

from ai_agent.workers.image_manager import WorkerState
from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6


def test_v6_source_keeps_both_models_loaded_once_and_hands_gpu_between_them():
    manager = WarmImageWorkerManagerV6(idle_seconds=300)
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="token",
        session_id="session-v6",
    )
    assert source.count("AutoTokenizer.from_pretrained") == 1
    assert source.count("AutoModelForCausalLM.from_pretrained") == 1
    assert source.count("Flux2KleinPipeline.from_pretrained") == 1
    assert "pipe.enable_model_cpu_offload()" in source
    assert "pipe.maybe_free_model_hooks()" in source
    assert "def acquire_qwen_gpu():" in source
    assert "def release_qwen_gpu():" in source
    assert 'recaptioner.to("cuda")' in source
    assert 'recaptioner.to("cpu")' in source
    assert 'device_map="auto"' not in source
    assert 'return_tensors="pt").to(recaptioner.device)' in source
    assert "torch.cuda.empty_cache()" in source
    assert 'User-Agent": "AIKA-Warm-Image/6.4"' in source
    assert 'max_new_tokens=int(CONFIG["recaption_max_new_tokens"])' in source
    assert '"recaption_max_new_tokens": 160' in source


def test_v6_qwen_and_flux_are_both_loaded_before_worker_registers_ready():
    manager = WarmImageWorkerManagerV6()
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="token",
        session_id="session-v6",
    )
    qwen_load = source.index("recaptioner = AutoModelForCausalLM.from_pretrained")
    flux_load = source.index("pipe = Flux2KleinPipeline.from_pretrained")
    ready = source.index('set_state("ready", "")')
    assert qwen_load < flux_load < ready
    assert source.count("recaptioner = AutoModelForCausalLM.from_pretrained") == 1


def test_v6_gpu_handoff_occurs_before_recaption_and_before_flux_generation():
    manager = WarmImageWorkerManagerV6()
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="token",
        session_id="session-v6",
    )
    acquire_call = source.index("acquire_qwen_gpu()", source.index("job_id = str(job"))
    release_call = source.index("release_qwen_gpu()", acquire_call)
    generate_loop = source.index("for index, prompt in enumerate(prompts):", release_call)
    assert acquire_call < release_call < generate_loop


def test_v6_source_reports_explicit_stages():
    manager = WarmImageWorkerManagerV6()
    source = manager._worker_source(
        base_url="https://example.invalid",
        worker_token="token",
        session_id="session-v6",
    )
    assert 'set_state("busy:recaption", job_id)' in source
    assert 'stage = f"busy:recaption:{item_index}/{item_total}"' in source
    assert 'stage = f"busy:generate:{index + 1}/{total}"' in source
    assert 'stage = f"busy:upload:{index + 1}/{total}"' in source
    assert 'request("POST", "/internal/image/result"' in source


def test_v6_busy_stage_is_preserved_without_becoming_starting(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManagerV6(worker_lease_seconds=30, job_lease_seconds=180)
    with manager._lock:
        manager._session_id = "session-v6"
        manager._state = WorkerState.READY
        manager._lease_expires_at = time.time() + 30
    assert manager.claim_job("session-v6", "job-1") is True
    assert manager.heartbeat(
        "session-v6",
        "busy:recaption",
        current_job_id="job-1",
    ) is True
    snap = manager.snapshot()
    assert snap.state == "busy"
    assert snap.stage == "recaption"
    assert snap.current_job_id == "job-1"


def test_v6_same_stage_heartbeat_does_not_reset_stage_deadline(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManagerV6(worker_lease_seconds=30, job_lease_seconds=180)
    with manager._lock:
        manager._session_id = "session-v6"
        manager._state = WorkerState.BUSY
        manager._stage = "recaption"
        manager._stage_started_at = time.time() - 30
        manager._lease_expires_at = time.time() + 30
        manager._current_job_id = "job-1"
        manager._job_lease_expires_at = time.time() + 180
    before = manager._stage_started_at
    assert manager.heartbeat(
        "session-v6",
        "busy:recaption",
        current_job_id="job-1",
    ) is True
    assert manager._stage_started_at == before


def test_v6_stalled_stage_fails_fast(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManagerV6(worker_lease_seconds=30, job_lease_seconds=300)
    with manager._lock:
        manager._session_id = "session-v6"
        manager._state = WorkerState.BUSY
        manager._stage = "recaption"
        manager._stage_started_at = time.time() - 181
        manager._lease_expires_at = time.time() + 30
        manager._current_job_id = "job-1"
        manager._job_lease_expires_at = time.time() + 120
    snap = manager.snapshot()
    assert snap.state == "error"
    assert snap.stage == "recaption_timeout"
    assert snap.session_id == ""
    assert "warm stage recaption exceeded 180s" in manager.last_error()


def test_v6_stage_transition_resets_stage_clock(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    manager = WarmImageWorkerManagerV6(worker_lease_seconds=30, job_lease_seconds=180)
    with manager._lock:
        manager._session_id = "session-v6"
        manager._state = WorkerState.BUSY
        manager._stage = "recaption"
        manager._stage_started_at = time.time() - 60
        manager._lease_expires_at = time.time() + 30
        manager._current_job_id = "job-1"
        manager._job_lease_expires_at = time.time() + 180
    old = manager._stage_started_at
    assert manager.heartbeat(
        "session-v6",
        "busy:generate:1/1",
        current_job_id="job-1",
    ) is True
    snap = manager.snapshot()
    assert snap.state == "busy"
    assert snap.stage == "generate:1/1"
    assert manager._stage_started_at > old
