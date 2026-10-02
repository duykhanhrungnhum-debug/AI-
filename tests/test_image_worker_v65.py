from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6
from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


def _sources():
    kwargs = {
        "base_url": "https://example.invalid",
        "worker_token": "test-token",
        "session_id": "test-session",
    }
    v6 = WarmImageWorkerManagerV6(kernel_slug="ai-agent-image-warm-canary")._worker_source(**kwargs)
    v65 = WarmImageWorkerManagerV65(kernel_slug="ai-agent-image-warm-canary")._worker_source(**kwargs)
    return v6, v65


def test_v65_keeps_qwen_cpu_only_and_flux_fp16():
    v6_source, v65_source = _sources()

    assert 'torch_dtype=torch.float16' in v65_source  # FLUX remains fp16.
    assert 'torch_dtype=torch.float32' in v65_source  # Qwen recaption is CPU-safe.
    assert 'recaptioner.to("cuda")' in v6_source
    assert 'recaptioner.to("cuda")' not in v65_source
    assert 'recaptioner.to("cpu")' not in v65_source
    assert "FLUX is the sole CUDA owner" in v65_source


def test_v65_selects_full_cuda_with_offload_fallback():
    _, v65_source = _sources()

    assert 'gpu_total_bytes >= 15 * 1024 ** 3' in v65_source
    assert 'pipe.to("cuda")' in v65_source
    assert 'flux_execution = "cuda"' in v65_source
    assert "pipe.enable_model_cpu_offload()" in v65_source
    assert 'flux_execution = "model_cpu_offload"' in v65_source


def test_v65_reports_gpu_preflight_in_generate_heartbeat():
    _, v65_source = _sources()

    assert "torch.cuda.mem_get_info()" in v65_source
    assert 'f"busy:generate:{index + 1}/{total}"' in v65_source
    assert 'f":gpu={gpu_slug}:mode={flux_execution}"' in v65_source
    assert 'f":free_mb={int(free_bytes // (1024 ** 2))}"' in v65_source
    assert 'f":total_mb={int(total_bytes // (1024 ** 2))}"' in v65_source


def test_v65_does_not_reintroduce_flux_callback_regression():
    _, v65_source = _sources()

    assert "callback_on_step_end=_step_heartbeat" not in v65_source
    assert "def _step_heartbeat" not in v65_source


def test_v65_preserves_generation_and_batch_flow():
    _, v65_source = _sources()

    for marker in (
        'stage = f"busy:upload:{index + 1}/{total}"',
        'image = pipe(',
        'request("POST", "/internal/image/result"',
        'explicit_items = split_explicit_items(command)',
    ):
        assert marker in v65_source
