"""Canary-only AIKA chat API entrypoint for Warm Image Worker v6.

Production keeps the existing entrypoint until v6 passes real reuse + batch E2E.
The canary must fail immediately when Warm fails; it must never hide a Warm
regression behind the production Cold Image V2 fallback.
"""
from __future__ import annotations

import os

from ai_agent.chat_session import CHAT_BROKER
from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6


# Replace only the warm image lifecycle. All routing, job storage, auth, chat,
# translation, TTS and other skills remain unchanged.
CHAT_BROKER._warm_image = WarmImageWorkerManagerV6()


def _canary_selftest_enabled() -> bool:
    return os.environ.get("AIKA_WARM_CANARY_SELFTEST", "").strip().casefold() in {
        "1", "true", "yes", "on"
    }


def _install_canary_fail_fast() -> None:
    """Disable cold fallback only for the isolated Warm canary self-test.

    Production keeps the normal fallback path. In canary mode, any Warm
    timeout/failure must surface immediately so the test cannot spend another
    several minutes running Cold Image V2 after Warm is already known-bad.
    """
    if not _canary_selftest_enabled():
        return

    def fail_fast(job_id: str, *, reason: str = "") -> None:
        detail = (reason or CHAT_BROKER._warm_image.last_error() or "warm canary failure").strip()
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
    _main()


if __name__ == "__main__":
    main()
