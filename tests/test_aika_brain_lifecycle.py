import time

from ai_agent.agent_broker import AIKAAgentBroker
from ai_agent.brain_lifecycle import (
    kernel_is_active,
    kernel_is_terminal,
    normalize_kernel_state,
)
from ai_agent.core.kaggle_worker import KaggleKernelStatus


def test_kernel_state_classification_handles_kaggle_cancel_acknowledged():
    assert normalize_kernel_state("cancelAcknowledged") == "cancelacknowledged"
    assert kernel_is_active(KaggleKernelStatus("RUNNING"))
    assert kernel_is_terminal(KaggleKernelStatus("cancelAcknowledged"))


def test_starting_worker_is_not_relaunched_inside_grace_window(monkeypatch):
    broker = AIKAAgentBroker()
    broker._worker_state = "starting"
    broker._worker_started_at = time.time()
    starts = []

    monkeypatch.setattr("threading.Thread.start", lambda self: starts.append(self.name))
    broker.ensure_worker()

    assert starts == []


def test_loading_heartbeat_keeps_starting_and_stopped_releases_worker():
    broker = AIKAAgentBroker()
    broker.heartbeat("loading")
    assert broker._worker_state == "starting"
    assert broker._worker_last_seen > 0

    broker.heartbeat("stopped")
    assert broker._worker_state == "idle"
    assert broker._worker_started_at == 0


def test_existing_active_kaggle_worker_is_reused_not_pushed(monkeypatch):
    submitted = []

    class FakeWorker:
        def __init__(self, **kwargs):
            pass

        def status(self, slug):
            assert slug == "aika-brain-prod-v2"
            return KaggleKernelStatus("RUNNING")

        def submit_script(self, **kwargs):
            submitted.append(kwargs)

    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "example.test")
    monkeypatch.setenv("AI_AGENT_API_TOKEN", "api-token")
    monkeypatch.setattr("ai_agent.agent_broker.KaggleGpuWorker", FakeWorker)

    broker = AIKAAgentBroker()
    monkeypatch.setattr(broker, "_start_worker_monitor", lambda worker: None)
    broker._launch_worker()

    assert submitted == []
    assert broker._worker_state == "starting"
    assert broker._worker_error == ""


def test_terminal_kaggle_worker_is_restarted_once(monkeypatch):
    submitted = []

    class FakeWorker:
        def __init__(self, **kwargs):
            pass

        def status(self, slug):
            return KaggleKernelStatus("cancelAcknowledged")

        def submit_script(self, **kwargs):
            submitted.append(kwargs)
            return object()

    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "example.test")
    monkeypatch.setenv("AI_AGENT_API_TOKEN", "api-token")
    monkeypatch.setattr("ai_agent.agent_broker.KaggleGpuWorker", FakeWorker)

    broker = AIKAAgentBroker()
    monkeypatch.setattr(broker, "_start_worker_monitor", lambda worker: None)
    broker._launch_worker()

    assert len(submitted) == 1
    assert submitted[0]["slug"] == "aika-brain-prod-v2"
    assert submitted[0]["title"] == "AIKA Brain Prod V2"


def test_worker_source_heartbeats_while_model_loads_and_on_exit():
    source = AIKAAgentBroker._worker_source(
        base_url="https://example.test",
        worker_token="secret",
        model="Qwen/Qwen2.5-3B-Instruct",
    )
    assert '"state": "loading"' in source
    assert '"state": "ready"' in source
    assert '"state": "stopped"' in source
    assert "loading_heartbeat" in source
