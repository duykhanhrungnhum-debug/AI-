"""Production-specific AIKA brain broker.

AIKA's production text brain runs as a short Kaggle *burst* job. Kaggle is a
batch compute platform, so production deliberately does not keep a notebook
alive as an RPC server. One burst loads the verified control-plane model,
processes every currently queued AIKA job, reports results, and exits so GPU is
released immediately when there is no work.
"""
from __future__ import annotations

import json
import os
import textwrap
import threading
import time

from .agent_broker import AIKAAgentBroker, AIKA_SYSTEM_PROMPT
from .brain_lifecycle import probe_kernel
from .core.kaggle_worker import KaggleGpuWorker


DEFAULT_BRAIN_MACHINE_SHAPE = "NvidiaTeslaT4"
DEFAULT_BRAIN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MIN_BRAIN_CUDA_MAJOR = 7


class ProductionAIKAAgentBroker(AIKAAgentBroker):
    """AIKA broker using finite, on-demand Kaggle GPU bursts."""

    def __init__(self) -> None:
        super().__init__()
        self.machine_shape = os.environ.get(
            "AIKA_BRAIN_MACHINE_SHAPE", DEFAULT_BRAIN_MACHINE_SHAPE
        ).strip()
        self.brain_model = os.environ.get(
            "AIKA_BRAIN_MODEL_NAME", DEFAULT_BRAIN_MODEL
        ).strip()
        if not self.machine_shape:
            raise ValueError("AIKA_BRAIN_MACHINE_SHAPE must not be empty")
        if not self.brain_model:
            raise ValueError("AIKA_BRAIN_MODEL_NAME must not be empty")

    def heartbeat(self, state: str = "ready") -> None:
        """Track a burst and safely relaunch only if work arrived during exit."""
        normalized = state.strip().casefold()
        super().heartbeat(state)
        if normalized not in {"stopped", "stopping", "idle_exit", "offline"}:
            return
        with self._lock:
            has_pending = any(job.status == "pending" for job in self._jobs.values())
        if has_pending:
            threading.Thread(
                target=self._ensure_after_burst_exit,
                name="aika-brain-burst-relaunch",
                daemon=True,
            ).start()

    def _ensure_after_burst_exit(self) -> None:
        # The stopped callback is emitted immediately before the Kaggle process
        # exits. Give Kaggle a short window to publish terminal state so the
        # next burst never mistakes the old process for a reusable live worker.
        time.sleep(12.0)
        self.ensure_worker()

    def _launch_worker(self) -> None:
        try:
            token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
            username = os.environ.get("KAGGLE_USERNAME", "").strip()
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

            # External Kaggle state is authoritative across Railway restarts.
            # If a burst is already booting/running, reuse it rather than push
            # a competing notebook version that would cancel the first one.
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
                model=self.brain_model,
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
                # Only treat a 409 as a harmless race if the exact production
                # burst is now genuinely active.
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
        """Return a finite Kaggle worker: load, drain queue, exit."""
        cfg = json.dumps(
            {
                "base_url": base_url.rstrip("/"),
                "worker_token": worker_token,
                "model": model,
                "max_new_tokens": 512,
                "system_prompt": AIKA_SYSTEM_PROMPT,
            },
            ensure_ascii=False,
        )
        return textwrap.dedent(
            f"""
            from __future__ import annotations
            import json
            import subprocess
            import sys
            import time
            import urllib.error
            import urllib.request

            CONFIG = json.loads({cfg!r})

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers<5", "accelerate<2", "safetensors", "sentencepiece",
                ])
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

            def request(method, path, payload=None):
                body = json.dumps(payload).encode("utf-8") if payload is not None else None
                req = urllib.request.Request(
                    CONFIG["base_url"] + path,
                    data=body,
                    method=method,
                    headers={{
                        "Authorization": "Bearer " + CONFIG["worker_token"],
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "AIKA-Brain/2.0",
                    }},
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        raw = response.read()
                        return json.loads(raw.decode("utf-8")) if raw else None
                except urllib.error.HTTPError as exc:
                    if exc.code == 204:
                        return None
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"HTTP {{exc.code}} {{path}}: {{detail[:500]}}") from exc

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
            gpu_arch_list = list(torch.cuda.get_arch_list())
            print("AIKA_BRAIN_GPU_PRECHECK " + json.dumps({{
                "gpu_name": gpu_name,
                "compute_capability": list(gpu_capability),
                "torch_cuda": str(torch.version.cuda or ""),
                "torch_arch_list": gpu_arch_list,
                "model": CONFIG["model"],
            }}), flush=True)
            if gpu_capability[0] < {MIN_BRAIN_CUDA_MAJOR}:
                raise RuntimeError(
                    "Unsupported AIKA brain GPU compute capability "
                    + str(gpu_capability)
                    + "; require CUDA compute capability >= 7.0"
                )

            started = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model"],
                dtype=torch.float16,
                device_map="auto",
            )
            model.eval()
            print(
                "AIKA_BRAIN_MODEL_READY "
                + json.dumps({{"model": CONFIG["model"], "seconds": round(time.perf_counter() - started, 3)}}),
                flush=True,
            )
            request("POST", "/internal/chat/heartbeat", {{"state": "ready"}})

            processed = 0
            try:
                while True:
                    job = request("GET", "/internal/chat/pull")
                    if not job:
                        break
                    job_id = str(job["job_id"])
                    prompt = str(job["prompt"])
                    try:
                        messages = [
                            {{"role": "system", "content": CONFIG["system_prompt"]}},
                            {{"role": "user", "content": prompt}},
                        ]
                        rendered = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=int(CONFIG["max_new_tokens"]),
                            do_sample=False,
                            repetition_penalty=1.05,
                        )
                        new_tokens = generated[:, inputs.input_ids.shape[1]:]
                        text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                        if not text:
                            raise RuntimeError("model returned empty text")
                        request("POST", "/internal/chat/result", {{
                            "job_id": job_id,
                            "text": text,
                            "provider": "kaggle-aika-brain",
                            "model": CONFIG["model"],
                        }})
                    except Exception as exc:
                        request("POST", "/internal/chat/result", {{
                            "job_id": job_id,
                            "error": type(exc).__name__ + ": " + str(exc),
                        }})
                    processed += 1
            finally:
                try:
                    request("POST", "/internal/chat/heartbeat", {{"state": "stopped"}})
                except Exception:
                    pass
            print("AIKA_BRAIN_BURST_EXIT processed=" + str(processed), flush=True)
            """
        ).strip() + "\n"


__all__ = [
    "DEFAULT_BRAIN_MACHINE_SHAPE",
    "DEFAULT_BRAIN_MODEL",
    "MIN_BRAIN_CUDA_MAJOR",
    "ProductionAIKAAgentBroker",
]
