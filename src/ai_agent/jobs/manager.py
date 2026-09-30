"""Thread-safe canonical job records for AIKA runtime work."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from uuid import uuid4


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    skill: str
    command: str
    source: str = "chat"
    job_id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    worker_id: str = ""
    retry_count: int = 0
    error: str = ""
    result: dict = field(default_factory=dict)


class JobManager:
    """Small runtime state store with deterministic state transitions."""

    def __init__(self, *, max_jobs: int = 200) -> None:
        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        self.max_jobs = max_jobs
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(self, *, skill: str, command: str, source: str = "chat", job_id: str = "") -> JobRecord:
        if not skill.strip():
            raise ValueError("skill is required")
        if not command.strip():
            raise ValueError("command is required")
        record = JobRecord(
            job_id=job_id.strip() or uuid4().hex,
            skill=skill.strip(),
            command=command.strip(),
            source=source.strip() or "chat",
        )
        with self._lock:
            self._jobs[record.job_id] = record
            self._trim_locked()
        return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return record

    def mark_running(self, job_id: str, *, worker_id: str = "") -> JobRecord:
        with self._lock:
            record = self._require_locked(job_id)
            record.status = JobStatus.RUNNING
            record.started_at = record.started_at or time.time()
            record.worker_id = worker_id.strip() or record.worker_id
            record.error = ""
            return record

    def mark_retrying(self, job_id: str, *, error: str = "") -> JobRecord:
        with self._lock:
            record = self._require_locked(job_id)
            record.status = JobStatus.RETRYING
            record.retry_count += 1
            record.error = error[:2000]
            return record

    def mark_done(self, job_id: str, result: dict | None = None) -> JobRecord:
        with self._lock:
            record = self._require_locked(job_id)
            record.status = JobStatus.DONE
            record.finished_at = time.time()
            record.error = ""
            record.result = dict(result or {})
            return record

    def mark_failed(self, job_id: str, error: str) -> JobRecord:
        with self._lock:
            record = self._require_locked(job_id)
            record.status = JobStatus.FAILED
            record.finished_at = time.time()
            record.error = error[:2000]
            return record

    def snapshot(self, job_id: str) -> dict:
        record = self.get(job_id)
        return {
            "job_id": record.job_id,
            "skill": record.skill,
            "source": record.source,
            "status": record.status.value,
            "worker_id": record.worker_id,
            "retry_count": record.retry_count,
            "error": record.error,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "result": dict(record.result),
        }

    def _require_locked(self, job_id: str) -> JobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    def _trim_locked(self) -> None:
        if len(self._jobs) <= self.max_jobs:
            return
        removable = sorted(self._jobs.values(), key=lambda item: item.created_at)
        for record in removable[: len(self._jobs) - self.max_jobs]:
            self._jobs.pop(record.job_id, None)
