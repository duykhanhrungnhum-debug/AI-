"""Production bootstrap for AIKA with a dedicated Kaggle brain notebook.

The brain remains AIKA. This module only gives the production brain worker a
stable notebook identity that cannot collide with legacy test/canary notebooks.
"""
from __future__ import annotations

import os
from types import MethodType

from ai_agent.agent_broker import AGENT_BROKER
from ai_agent.chat_api import main as chat_main
from ai_agent.core.kaggle_worker import KaggleGpuWorker


PRODUCTION_BRAIN_SLUG = "aika-brain-prod-v2"
PRODUCTION_BRAIN_TITLE = "AIKA Brain Prod V2"


def _launch_production_worker(self) -> None:
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
        source = self._worker_source(
            base_url="https://" + public_domain,
            worker_token=worker_token,
            model=model,
        )
        worker.submit_script(
            slug=PRODUCTION_BRAIN_SLUG,
            title=PRODUCTION_BRAIN_TITLE,
            source=source,
            enable_internet=True,
            enable_gpu=True,
            is_private=True,
        )
        with self._lock:
            self._worker_state = "starting"
    except Exception as exc:
        with self._lock:
            self._worker_state = "error"
            self._worker_error = str(exc)[:1000]
            for job in self._jobs.values():
                if job.status == "pending":
                    job.status = "error"
                    job.error = self._worker_error
    finally:
        with self._lock:
            self._launching = False


def configure() -> None:
    AGENT_BROKER.kernel_slug = PRODUCTION_BRAIN_SLUG
    AGENT_BROKER._launch_worker = MethodType(_launch_production_worker, AGENT_BROKER)


def main() -> None:
    configure()
    chat_main()


if __name__ == "__main__":
    main()
