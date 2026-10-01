"""AIKA Warm Image Worker v6.5 compatibility manager.

V6.5 keeps the verified V6 worker execution path unchanged. An earlier v6.5
experiment injected ``callback_on_step_end`` into FLUX to obtain denoise-step
heartbeats, but deployed canaries repeatedly stalled before the first denoise
step while the verified V6 path had already completed first/reuse/batch E2E.

The compatibility class remains so production/canary wiring does not churn, but
its worker source now deliberately delegates to V6 without pipeline callbacks.
V6's normal stage watchdog and leases remain the failure boundary.
"""
from __future__ import annotations

from .image_manager_v6 import WarmImageWorkerManagerV6


class WarmImageWorkerManagerV65(WarmImageWorkerManagerV6):
    """Compatibility wrapper around the verified V6 warm-image execution path."""

    def _worker_source(
        self,
        *,
        base_url: str,
        worker_token: str,
        session_id: str = "test-session",
    ) -> str:
        return super()._worker_source(
            base_url=base_url,
            worker_token=worker_token,
            session_id=session_id,
        )
