import pytest

from ai_agent.command import CommandEnvelope
from ai_agent.jobs import JobManager, JobStatus


def test_command_envelope_is_transport_neutral():
    command = CommandEnvelope(command="AIKA tạo ảnh một con trâu", source="operator")
    assert command.command.startswith("AIKA")
    assert command.source == "operator"
    assert command.command_id
    with pytest.raises(ValueError):
        CommandEnvelope(command="")


def test_job_manager_has_explicit_lifecycle_without_media_routing():
    manager = JobManager(max_jobs=10)
    job = manager.create(skill="image", command="tạo ảnh", source="aika", job_id="job-1")
    assert job.status is JobStatus.QUEUED
    manager.mark_running("job-1", worker_id="image-tool")
    assert manager.get("job-1").status is JobStatus.RUNNING
    manager.mark_retrying("job-1", error="temporary")
    assert manager.get("job-1").retry_count == 1
    manager.mark_done("job-1", {"count": 1})
    snapshot = manager.snapshot("job-1")
    assert snapshot["status"] == "done"
    assert snapshot["result"] == {"count": 1}
