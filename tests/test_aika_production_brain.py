from ai_agent.core.kaggle_worker import KaggleKernelStatus
from ai_agent.production_agent_broker import (
    DEFAULT_BRAIN_MACHINE_SHAPE,
    DEFAULT_BRAIN_MODEL,
    MIN_BRAIN_CUDA_MAJOR,
    ProductionAIKAAgentBroker,
)


def _production_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "example.test")
    monkeypatch.setenv("AI_AGENT_API_TOKEN", "api-token")


def test_production_brain_defaults_to_verified_runtime(monkeypatch):
    monkeypatch.delenv("AIKA_BRAIN_MACHINE_SHAPE", raising=False)
    monkeypatch.delenv("AIKA_BRAIN_MODEL_NAME", raising=False)
    monkeypatch.setenv("AI_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

    broker = ProductionAIKAAgentBroker()

    assert broker.machine_shape == DEFAULT_BRAIN_MACHINE_SHAPE == "NvidiaTeslaT4"
    assert broker.brain_model == DEFAULT_BRAIN_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"


def test_production_brain_submission_pins_machine_shape(monkeypatch):
    _production_env(monkeypatch)
    submitted = []

    class FakeWorker:
        def __init__(self, **kwargs):
            pass

        def status(self, slug):
            assert slug == "aika-brain-burst-v1"
            return KaggleKernelStatus("complete")

        def submit_script(self, **kwargs):
            submitted.append(kwargs)
            return object()

    monkeypatch.setattr("ai_agent.production_agent_broker.KaggleGpuWorker", FakeWorker)
    broker = ProductionAIKAAgentBroker()
    broker.kernel_slug = "aika-brain-burst-v1"
    broker.kernel_title = "AIKA Brain Burst V1"
    monkeypatch.setattr(broker, "_start_worker_monitor", lambda worker: None)
    broker._launch_worker()

    assert len(submitted) == 1
    assert submitted[0]["slug"] == "aika-brain-burst-v1"
    assert submitted[0]["title"] == "AIKA Brain Burst V1"
    assert submitted[0]["machine_shape"] == "NvidiaTeslaT4"
    assert submitted[0]["enable_gpu"] is True
    assert "Qwen/Qwen2.5-1.5B-Instruct" in submitted[0]["source"]


def test_production_worker_is_finite_burst_and_compiles():
    source = ProductionAIKAAgentBroker._production_worker_source(
        base_url="https://example.test",
        worker_token="secret",
        model=DEFAULT_BRAIN_MODEL,
    )
    assert "AIKA_BRAIN_GPU_PRECHECK" in source
    assert "AIKA_BRAIN_MODEL_READY" in source
    assert "AIKA_BRAIN_BURST_EXIT" in source
    assert 'request("POST", "/internal/chat/heartbeat", {"state": "ready"})' in source
    assert 'request("POST", "/internal/chat/heartbeat", {"state": "stopped"})' in source
    assert "loading_heartbeat" not in source
    assert "idle_polls" not in source
    assert "poll_seconds" not in source
    assert "torch.cuda.get_device_capability(0)" in source
    assert f"gpu_capability[0] < {MIN_BRAIN_CUDA_MAJOR}" in source
    compile(source, "aika-production-burst.py", "exec")


def test_production_boot_injects_one_burst_broker(monkeypatch):
    _production_env(monkeypatch)
    from ai_agent import agent_broker as broker_module
    from ai_agent import chat_api
    from ai_agent import production_boot

    broker = production_boot.configure()
    assert isinstance(broker, ProductionAIKAAgentBroker)
    assert chat_api.AGENT_BROKER is broker
    assert broker_module.AGENT_BROKER is broker
    assert broker.kernel_slug == "aika-brain-burst-v1"
    assert broker.kernel_title == "AIKA Brain Burst V1"
    assert broker.brain_model == DEFAULT_BRAIN_MODEL
