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
    # A legacy/global model setting must not silently move the AIKA control plane
    # back to the unstable 3B cold-start path.
    monkeypatch.setenv("AI_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

    broker = ProductionAIKAAgentBroker()

    assert broker.machine_shape == "NvidiaTeslaT4"
    assert broker.machine_shape == DEFAULT_BRAIN_MACHINE_SHAPE
    assert broker.brain_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert broker.brain_model == DEFAULT_BRAIN_MODEL


def test_production_brain_submission_pins_machine_shape(monkeypatch):
    _production_env(monkeypatch)
    submitted = []

    class FakeWorker:
        def __init__(self, **kwargs):
            pass

        def status(self, slug):
            assert slug == "aika-brain-prod-v2"
            return KaggleKernelStatus("cancelAcknowledged")

        def submit_script(self, **kwargs):
            submitted.append(kwargs)
            return object()

    monkeypatch.setattr(
        "ai_agent.production_agent_broker.KaggleGpuWorker", FakeWorker
    )
    broker = ProductionAIKAAgentBroker()
    monkeypatch.setattr(broker, "_start_worker_monitor", lambda worker: None)
    broker._launch_worker()

    assert len(submitted) == 1
    assert submitted[0]["slug"] == "aika-brain-prod-v2"
    assert submitted[0]["machine_shape"] == "NvidiaTeslaT4"
    assert submitted[0]["enable_gpu"] is True
    assert "Qwen/Qwen2.5-1.5B-Instruct" in submitted[0]["source"]


def test_production_worker_source_has_gpu_preflight_and_compiles():
    source = ProductionAIKAAgentBroker._production_worker_source(
        base_url="https://example.test",
        worker_token="secret",
        model=DEFAULT_BRAIN_MODEL,
    )
    assert "AIKA_BRAIN_GPU_PRECHECK" in source
    assert "torch.cuda.get_device_capability(0)" in source
    assert f"gpu_capability[0] < {MIN_BRAIN_CUDA_MAJOR}" in source
    assert "require CUDA compute capability >= 7.0" in source
    compile(source, "aika-production-worker.py", "exec")


def test_production_boot_injects_one_production_broker(monkeypatch):
    _production_env(monkeypatch)
    from ai_agent import agent_broker as broker_module
    from ai_agent import chat_api
    from ai_agent import production_boot

    broker = production_boot.configure()
    assert isinstance(broker, ProductionAIKAAgentBroker)
    assert chat_api.AGENT_BROKER is broker
    assert broker_module.AGENT_BROKER is broker
    assert broker.kernel_slug == "aika-brain-prod-v2"
    assert broker.kernel_title == "AIKA Brain Prod V2"
    assert broker.brain_model == DEFAULT_BRAIN_MODEL
