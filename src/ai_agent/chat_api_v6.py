"""Canary-only AIKA chat API entrypoint for Warm Image Worker validation.

Production keeps the normal API entrypoint. The canary uses a fixed, separate
Kaggle kernel slug so soak/regression runs can never compete with the production
warm worker session.
"""
from __future__ import annotations

import os
import threading

from ai_agent.chat_session import CHAT_BROKER
from ai_agent.workers.image_manager_v65 import WarmImageWorkerManagerV65


# Canary isolation must not be overridable by Railway/shared variables.
CANARY_KERNEL_SLUG = "ai-agent-image-warm-canary"

# Replace only the warm image lifecycle. All routing, job storage, auth, chat,
# translation, TTS and other skills remain unchanged.
CHAT_BROKER._warm_image = WarmImageWorkerManagerV65(kernel_slug=CANARY_KERNEL_SLUG)


def _canary_selftest_enabled() -> bool:
    return os.environ.get("AIKA_WARM_CANARY_SELFTEST", "").strip().casefold() in {
        "1", "true", "yes", "on"
    }


def _soak_enabled() -> bool:
    try:
        return int(os.environ.get("AIKA_WARM_SOAK_COUNT", "0")) > 0
    except ValueError:
        return False


def _install_canary_fail_fast() -> None:
    """Disable cold fallback for isolated canary validation."""
    if not (_canary_selftest_enabled() or _soak_enabled()):
        return

    def fail_fast(job_id: str, *, reason: str = "") -> None:
        with CHAT_BROKER._lock:
            job = CHAT_BROKER._jobs.get(job_id)
            worker_error = (job.error if job is not None else "").strip()
        detail = (
            reason
            or worker_error
            or CHAT_BROKER._warm_image.last_error()
            or "warm canary failure"
        ).strip()
        CHAT_BROKER._warm_image.invalidate_job(job_id, detail)
        with CHAT_BROKER._lock:
            job = CHAT_BROKER._jobs.get(job_id)
            if job is None or job.status in {"done", "error"}:
                return
            job.status = "error"
            job.error = ("warm canary fail-fast: " + detail)[:2000]

    CHAT_BROKER._start_cold_fallback = fail_fast


_install_canary_fail_fast()

from ai_agent.chat_api import main as _main  # noqa: E402


def main() -> None:
    if _soak_enabled():
        from ai_agent.canary_soak import run_warm_soak

        threading.Thread(
            target=run_warm_soak,
            name="aika-warm-soak",
            daemon=True,
        ).start()
    _main()


if __name__ == "__main__":
    main()
