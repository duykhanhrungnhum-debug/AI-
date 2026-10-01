from ai_agent.chat_session import ChatJob, ChatSessionBroker
from ai_agent.core.media_v2 import ImageResultV2


def _image_job(job_id: str = "job-fallback") -> ChatJob:
    return ChatJob(
        job_id=job_id,
        prompt="AIKA tạo ảnh quả táo đỏ",
        message="AIKA tạo ảnh quả táo đỏ",
        kind="image",
        status="processing",
        created_at=1.0,
    )


def test_warm_error_hands_off_to_cold_once(monkeypatch):
    monkeypatch.setenv("AIKA_IMAGE_WARM_WORKER", "true")
    broker = ChatSessionBroker()
    job = _image_job()
    with broker._lock:
        broker._jobs[job.job_id] = job

    assert broker._warm_image.heartbeat("session-a", "ready") is True
    assert broker._warm_image.claim_job("session-a", job.job_id) is True

    calls = []

    def record_fallback(job_id: str, *, reason: str = "") -> None:
        calls.append((job_id, reason))

    monkeypatch.setattr(broker, "_start_cold_fallback", record_fallback)
    broker.finish_image_job(
        {"job_id": job.job_id, "error": "RuntimeError: simulated warm failure"},
        session_id="session-a",
    )

    assert calls == [(job.job_id, "")]
    with broker._lock:
        assert broker._jobs[job.job_id].status == "pending"
        assert broker._jobs[job.job_id].error == "RuntimeError: simulated warm failure"
    assert broker._warm_image.snapshot().current_job_id == ""


def test_cold_fallback_executor_completes_job(monkeypatch):
    broker = ChatSessionBroker()
    job = _image_job("job-cold")
    with broker._lock:
        broker._jobs[job.job_id] = job

    fake = ImageResultV2(
        item_id="image-01",
        data=b"\x89PNG\r\n\x1a\nfake",
        prompt="a red apple",
        command=job.message,
        model="black-forest-labs/FLUX.2-klein-4B",
        seed=1,
        width=1024,
        height=1024,
        elapsed_seconds=29.0,
        evidence=("test",),
    )
    monkeypatch.setattr(
        broker._image_executor,
        "execute_command_cold",
        lambda *args, **kwargs: (fake,),
    )

    broker._run_image_job(job.job_id)
    state = broker.get_job(job.job_id)
    assert state["status"] == "done"
    assert state["provider"] == "kaggle-image-v2"
    assert state["image_count"] == 1
    assert state["elapsed_seconds"] == 29.0
