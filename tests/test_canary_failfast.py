import threading
from types import SimpleNamespace

import pytest

import ai_agent.canary_selftest as canary


class _FailedWarm:
    def snapshot(self):
        return SimpleNamespace(
            state="error",
            stage="recaption_timeout",
            launch_count=1,
            session_id="",
            current_job_id="",
            job_lease_expires_at=0.0,
            last_seen=0.0,
        )

    def last_error(self):
        return "warm stage recaption exceeded 180s"


def test_wait_job_aborts_immediately_when_warm_manager_is_error(monkeypatch):
    class Broker:
        _warm_image = _FailedWarm()

        def get_job(self, job_id):
            raise AssertionError("fail-fast path must not keep polling the job")

    monkeypatch.setattr(canary, "CHAT_BROKER", Broker())

    with pytest.raises(RuntimeError, match="warm stage recaption exceeded 180s"):
        canary._wait_job(
            "job-1",
            timeout_seconds=720,
            worker=object(),
            label="first",
        )


def test_canary_guard_blocks_cold_fallback(monkeypatch):
    job = SimpleNamespace(status="processing", error="")
    original_calls = []

    def original_fallback(job_id, *, reason=""):
        original_calls.append((job_id, reason))

    broker = SimpleNamespace(
        _warm_image=_FailedWarm(),
        _lock=threading.Lock(),
        _jobs={"job-1": job},
        _start_cold_fallback=original_fallback,
    )
    monkeypatch.setattr(canary, "CHAT_BROKER", broker)

    original = canary._install_no_cold_fallback_guard()
    assert original is original_fallback

    broker._start_cold_fallback("job-1")

    assert original_calls == []
    assert job.status == "error"
    assert "recaption exceeded 180s" in job.error
