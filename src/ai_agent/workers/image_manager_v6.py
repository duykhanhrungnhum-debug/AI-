"""AIKA Warm Image Worker v6 lifecycle.

V6 keeps lifecycle/leases separate from the self-contained Kaggle runtime. The
runtime source lives in image_worker_source_v6.py so model/batch tuning does not
risk changing worker-session state management.
"""
from __future__ import annotations

import time

from .image_manager import WarmImageWorkerManager, WorkerState
from .image_worker_source_v6 import build_v6_worker_source
from ai_agent.core.media_v2 import (
    IMAGE_BATCH_DELIMITER,
    IMAGE_GUIDANCE,
    IMAGE_MODEL,
    IMAGE_STEPS,
    RECAPTION_MODEL,
)


class WarmImageWorkerManagerV6(WarmImageWorkerManager):
    """Warm manager with explicit stage watchdogs and CPU-only recaption."""

    STAGE_TIMEOUTS = {
        "recaption": 180.0,
        "generate": 120.0,
        "upload": 90.0,
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stage_started_at = 0.0

    @staticmethod
    def _stage_family(stage: str) -> str:
        return (stage or "").split(":", 1)[0].strip().casefold()

    def _set_stage_locked(self, stage: str, now: float) -> None:
        stage = (stage or self._state.value).strip().casefold()
        if stage != self._stage:
            self._stage_started_at = now
        elif not self._stage_started_at:
            self._stage_started_at = now
        self._stage = stage

    def _clear_session_locked(self) -> None:
        super()._clear_session_locked()
        self._stage_started_at = 0.0

    def _expire_stale_locked(self, now: float) -> None:
        super()._expire_stale_locked(now)
        if self._state is not WorkerState.BUSY or not self._session_id:
            return
        family = self._stage_family(self._stage)
        timeout = self.STAGE_TIMEOUTS.get(family)
        if not timeout or not self._stage_started_at:
            return
        age = now - self._stage_started_at
        if age >= timeout:
            job_id = self._current_job_id
            self._record_failure_locked(
                f"warm stage {self._stage} exceeded {timeout:.0f}s"
                + (f" for {job_id}" if job_id else ""),
                stage=f"{family}_timeout",
            )

    def heartbeat(
        self,
        session_id: str,
        state: str,
        *,
        current_job_id: str = "",
    ) -> bool:
        raw = (state or "ready").strip().casefold()
        if raw.startswith("busy:"):
            stage = raw.split(":", 1)[1].strip() or "busy"
            with self._lock:
                previous_stage = self._stage
                previous_started_at = self._stage_started_at
            accepted = super().heartbeat(
                session_id,
                "busy",
                current_job_id=current_job_id,
            )
            if not accepted:
                return False
            now = time.time()
            with self._lock:
                if session_id != self._session_id:
                    return False
                if stage == previous_stage and previous_started_at:
                    self._stage = stage
                    self._stage_started_at = previous_started_at
                else:
                    self._set_stage_locked(stage, now)
            return True

        accepted = super().heartbeat(
            session_id,
            raw,
            current_job_id=current_job_id,
        )
        if not accepted:
            return False
        now = time.time()
        with self._lock:
            if session_id == self._session_id:
                self._set_stage_locked(raw, now)
        return True

    def claim_job(self, session_id: str, job_id: str) -> bool:
        accepted = super().claim_job(session_id, job_id)
        if accepted:
            with self._lock:
                self._set_stage_locked("claimed", time.time())
        return accepted

    def release_job(self, session_id: str, job_id: str) -> bool:
        accepted = super().release_job(session_id, job_id)
        if accepted:
            with self._lock:
                self._set_stage_locked("ready", time.time())
        return accepted

    def _worker_source(
        self,
        *,
        base_url: str,
        worker_token: str,
        session_id: str = "test-session",
    ) -> str:
        return build_v6_worker_source({
            "base_url": base_url.rstrip("/"),
            "worker_token": worker_token,
            "session_id": session_id,
            "image_model": IMAGE_MODEL,
            "recaption_model": RECAPTION_MODEL,
            "steps": IMAGE_STEPS,
            "guidance": IMAGE_GUIDANCE,
            "idle_seconds": self.idle_seconds,
            "poll_seconds": 3,
            "heartbeat_seconds": 10,
            "max_images": self.max_images_per_job,
            "delimiter": IMAGE_BATCH_DELIMITER,
            "recaption_max_new_tokens": 160,
        })
