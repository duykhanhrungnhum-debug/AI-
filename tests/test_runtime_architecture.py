import ast
import base64
import re

import pytest

from ai_agent.command import CommandEnvelope
from ai_agent.jobs import JobManager, JobStatus
from ai_agent.router import route_skill
from ai_agent.workers.image_manager import WarmImageWorkerManager


def _warm_child_source(source: str) -> str:
    match = re.search(r"CHILD_SOURCE = base64\.b64decode\((.+?)\)\.decode\(\"utf-8\"\)", source)
    assert match is not None
    encoded = ast.literal_eval(match.group(1))
    return base64.b64decode(encoded).decode("utf-8")


def test_command_envelope_is_transport_neutral():
    command = CommandEnvelope(command="AIKA tạo ảnh một con trâu", source="operator")
    assert command.command.startswith("AIKA")
    assert command.source == "operator"
    assert command.command_id
    with pytest.raises(ValueError):
        CommandEnvelope(command="")


def test_skill_router_routes_at_skill_level_only():
    assert route_skill("AIKA tạo ảnh hai cô gái châu Á") == "image"
    assert route_skill("Vẽ hình một con sao la 3D") == "image"
    assert route_skill("Giải thích quy trình tạo ảnh") == "chat"
    assert route_skill("Xin chào AIKA") == "chat"
    assert route_skill("anything", skill_hint="image") == "image"


def test_job_manager_has_explicit_lifecycle():
    manager = JobManager(max_jobs=10)
    job = manager.create(skill="image", command="tạo ảnh", source="chat", job_id="job-1")
    assert job.status is JobStatus.QUEUED
    manager.mark_running("job-1", worker_id="warm-1")
    assert manager.get("job-1").status is JobStatus.RUNNING
    manager.mark_retrying("job-1", error="temporary")
    assert manager.get("job-1").retry_count == 1
    manager.mark_done("job-1", {"count": 1})
    snapshot = manager.snapshot("job-1")
    assert snapshot["status"] == "done"
    assert snapshot["result"] == {"count": 1}


def test_warm_worker_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AIKA_IMAGE_WARM_WORKER", raising=False)
    manager = WarmImageWorkerManager()
    assert manager.enabled is False
    assert manager.ensure_started() is False
    assert manager.snapshot().state == "offline"


def test_warm_worker_source_loads_models_once_and_polls_jobs():
    manager = WarmImageWorkerManager(idle_seconds=300)
    source = manager._worker_source(base_url="https://example.test", worker_token="secret")
    child = _warm_child_source(source)
    assert child.count("AutoTokenizer.from_pretrained") == 1
    assert child.count("AutoModelForCausalLM.from_pretrained") == 1
    assert child.count("Flux2KleinPipeline.from_pretrained") == 1
    assert '"/internal/image/pull"' in child
    assert '"/internal/image/result"' in child
    assert '"/internal/image/heartbeat"' in child
    assert "while time.monotonic() - idle_started" in child
    assert "recaptioner.to(\"cuda\")" in child
    assert "recaptioner.to(\"cpu\")" in child
    assert "pipe.enable_model_cpu_offload()" in child
