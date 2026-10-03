"""Production-specific AIKA brain broker.

The generic AIKA broker owns job/tool semantics and durable worker lifecycle.
This adapter owns only the production accelerator contract: AIKA's text brain
runs on an explicitly requested Kaggle accelerator rather than whatever Kaggle
happens to choose by default.
"""
from __future__ import annotations

import json
import os
import time

from .agent_broker import AIKAAgentBroker
from .brain_lifecycle import probe_kernel
from .core.kaggle_worker import KaggleGpuWorker


DEFAULT_BRAIN_MACHINE_SHAPE = "NvidiaTeslaT4"
MIN_BRAIN_CUDA_MAJOR = 7


class ProductionAIKAAgentBroker(AIKAAgentBroker):
    """AIKA broker with an explicit, verified production GPU contract."""

    def __init__(self) -> None:
        super().__init__()
        self.machine_shape = os.environ.get(
            "AIKA_BRAIN_MACHINE_SHAPE", DEFAULT_BRAIN_MACHINE_SHAPE
        ).strip()
        if not self.machine_shape:
            raise ValueError("AIKA_BRAIN_MACHINE_SHAPE must not be empty")

    def _launch_worker(self) -> None:
        try:
            token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
            username = os.environ.get("KAGGLE_USERNAME", "").strip()
            model = os.environ.get("AI_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct").strip()
            public_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
            worker_token = os.environ.get("AI_AGENT_API_TOKEN", "").strip()
            if not token:
                raise RuntimeError("KAGGLE_API_TOKEN is required")
            if not username:
                raise RuntimeError("KAGGLE_USERNAME is required")
            if not public_domain:
                raise RuntimeError("RAILWAY_PUBLIC_DOMAIN is required")
            if not worker_token:
                raise RuntimeError("AI_AGENT_API_TOKEN is required")

            worker = KaggleGpuWorker(
                api_token=token,
                username=username,
                timeout=120,
                submission_retry_attempts=2,
                submission_retry_delay_seconds=10,
            )

            # Kaggle is the source of truth across Railway process restarts.
            # Reuse an existing active worker instead of pushing over it.
            probe = probe_kernel(worker, self.kernel_slug)
            if probe.active:
                with self._lock:
                    self._worker_state = "starting"
                    self._worker_error = ""
                    if self._worker_started_at <= 0:
                        self._worker_started_at = time.time()
                self._start_worker_monitor(worker)
                return
            if probe.exists and not probe.terminal:
                status = probe.status.status if probe.status else "unknown"
                raise RuntimeError(f"unexpected Kaggle brain state: {status}")

            source = self._production_worker_source(
                base_url="https://" + public_domain,
                worker_token=worker_token,
                model=model,
            )
            try:
                worker.submit_script(
                    slug=self.kernel_slug,
                    title=self.kernel_title,
                    source=source,
                    machine_shape=self.machine_shape,
                    enable_internet=True,
                    enable_gpu=True,
                    is_private=True,
                )
            except RuntimeError as exc:
                # A concurrent API request can race the first submission. A 409
                # is safe only when the exact production kernel is now active.
                message = str(exc).casefold()
                if "kaggle http 409" in message or "already in use" in message:
                    race_probe = probe_kernel(worker, self.kernel_slug)
                    if race_probe.active:
                        with self._lock:
                            self._worker_state = "starting"
                            self._worker_error = ""
                            self._worker_started_at = time.time()
                        self._start_worker_monitor(worker)
                        return
                raise

            with self._lock:
                self._worker_state = "starting"
                self._worker_error = ""
                self._worker_started_at = time.time()
            self._start_worker_monitor(worker)
        except Exception as exc:
            self._mark_pending_jobs_error(str(exc))
        finally:
            with self._lock:
                self._launching = False

    @staticmethod
    def _production_worker_source(*, base_url: str, worker_token: str, model: str) -> str:
        """Add a hardware preflight before the generic worker loads model weights."""
        source = AIKAAgentBroker._worker_source(
            base_url=base_url,
            worker_token=worker_token,
            model=model,
        )
        marker = (
            'if not torch.cuda.is_available():\n'
            '    raise RuntimeError("CUDA GPU is not available")\n'
        )
        if marker not in source:
            raise RuntimeError("AIKA brain worker preflight insertion point changed")
        replacement = marker + (
            'gpu_name = torch.cuda.get_device_name(0)\n'
            'gpu_capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))\n'
            'gpu_arch_list = list(torch.cuda.get_arch_list())\n'
            'print("AIKA_BRAIN_GPU_PRECHECK " + json.dumps({\n'
            '    "gpu_name": gpu_name,\n'
            '    "compute_capability": list(gpu_capability),\n'
            '    "torch_cuda": str(torch.version.cuda or ""),\n'
            '    "torch_arch_list": gpu_arch_list,\n'
            '}), flush=True)\n'
            f'if gpu_capability[0] < {MIN_BRAIN_CUDA_MAJOR}:\n'
            '    raise RuntimeError(\n'
            '        "Unsupported AIKA brain GPU compute capability "\n'
            '        + str(gpu_capability)\n'
            '        + "; require CUDA compute capability >= 7.0"\n'
            '    )\n'
        )
        return source.replace(marker, replacement, 1)


__all__ = [
    "DEFAULT_BRAIN_MACHINE_SHAPE",
    "MIN_BRAIN_CUDA_MAJOR",
    "ProductionAIKAAgentBroker",
]
