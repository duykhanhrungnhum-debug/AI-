"""Stable production wrapper for Image V2.

The normal Kaggle status endpoint is not reliable for the current private-token
setup (``kernels.get`` can return 403 even while a submitted kernel runs).  This
provider therefore treats the output artifact itself as the source of truth.

Each run injects a unique token into the worker log and final report, so an old
report from the previous kernel version can never be mistaken for the current
run.  Polling is bounded to <10 minutes and logs are used only for early failure
diagnostics, never as a required control-plane dependency.
"""
from __future__ import annotations

from hashlib import sha256
import json
import time
from uuid import uuid4

from .media_v2 import KaggleImageV2Provider


class StableKaggleImageV2Provider(KaggleImageV2Provider):
    """Image V2 provider that verifies completion from the current output report."""

    max_wait_seconds: float = 540.0

    def _instrument_source(self, source: str, run_token: str) -> str:
        future = "from __future__ import annotations\n"
        if future not in source:
            raise RuntimeError("Image V2 source no longer has the expected future import")
        source = source.replace(
            future,
            future
            + f'\nRUN_TOKEN = {run_token!r}\n'
            + 'print("AIKA_IMAGE_RUN " + RUN_TOKEN, flush=True)\n',
            1,
        )
        report_write = 'Path("/kaggle/working/image_v2_report.json").write_text('
        if report_write not in source:
            raise RuntimeError("Image V2 source no longer has the expected report writer")
        return source.replace(
            report_write,
            'report["run_token"] = RUN_TOKEN\n' + report_write,
            1,
        )

    def _run_source(self, source: str):
        run_token = uuid4().hex
        source = self._instrument_source(source, run_token)
        kernel_title = (
            "AI Agent Image V2"
            if self.kernel_slug == "ai-agent-image-v2"
            else f"AI Agent Image V2 {sha256(self.kernel_slug.encode('utf-8')).hexdigest()[:8]}"
        )
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=kernel_title,
            source=source,
            enable_internet=True,
            enable_gpu=True,
            is_private=True,
        )

        poll_interval = max(1.0, float(self.poll_interval or 1.0))
        configured_window = max(1, int(self.max_poll_attempts)) * poll_interval
        deadline = time.monotonic() + min(self.max_wait_seconds, configured_window)
        last_logs = ""

        while time.monotonic() < deadline:
            try:
                raw = self.worker.download_output_file(self.kernel_slug, "image_v2_report.json")
                report = json.loads(raw.decode("utf-8"))
                if isinstance(report, dict) and report.get("run_token") == run_token:
                    items = report.get("items")
                    if not isinstance(items, dict):
                        raise ValueError("image V2 report is invalid")
                    gpu_name = str(report.get("gpu_name") or "").strip()
                    if not gpu_name:
                        raise ValueError("image V2 report has no GPU evidence")
                    return submission, report, gpu_name
            except FileNotFoundError:
                pass
            except RuntimeError as exc:
                # Output can legitimately be absent while the current kernel is running.
                # Authentication/permission failures are not transient and should fail fast.
                text = str(exc)
                if "HTTP 404" not in text and "HTTP 409" not in text:
                    raise
            except json.JSONDecodeError:
                pass

            try:
                logs = self.worker.logs(self.kernel_slug)
                if logs:
                    last_logs = logs[-12000:]
                    current = last_logs.split("AIKA_IMAGE_RUN " + run_token, 1)
                    if len(current) == 2:
                        tail = current[1]
                        if "Traceback (most recent call last)" in tail:
                            raise RuntimeError("image V2 worker failed: " + tail[-8000:])
            except RuntimeError as exc:
                # Logs are diagnostic only. A permission error here must not block output polling.
                if "image V2 worker failed:" in str(exc):
                    raise
            except Exception:
                pass

            time.sleep(poll_interval)

        detail = ""
        if last_logs:
            detail = ": " + last_logs[-4000:]
        raise TimeoutError(
            f"image V2 run {run_token[:8]} exceeded {min(self.max_wait_seconds, configured_window):.0f}s{detail}"
        )
