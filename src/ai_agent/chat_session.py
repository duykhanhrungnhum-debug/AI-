"""Asynchronous Kaggle-backed chat sessions for the web chat UI."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import secrets
import threading
import time
import textwrap
from urllib.error import HTTPError
from uuid import uuid4

from .core.kaggle_worker import KaggleGpuWorker


@dataclass
class ChatJob:
    job_id: str
    prompt: str
    status: str = "pending"
    text: str = ""
    provider: str = ""
    model: str = ""
    error: str = ""
    created_at: float = 0.0


class ChatSessionBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ChatJob] = {}
        self._worker_state = "idle"
        self._worker_error = ""
        self._worker_last_seen = 0.0
        self._launching = False
        self.kernel_slug = "ai-agent-chat-session"

    def create_job(self, prompt: str) -> ChatJob:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt is required")
        job = ChatJob(job_id=uuid4().hex, prompt=prompt, created_at=time.time())
        with self._lock:
            self._jobs[job.job_id] = job
            if len(self._jobs) > 100:
                oldest = sorted(self._jobs.values(), key=lambda j: j.created_at)[:-80]
                for item in oldest:
                    self._jobs.pop(item.job_id, None)
        self.ensure_worker()
        return job

    def ensure_worker(self) -> None:
        with self._lock:
            alive = self._worker_state == "running" and time.time() - self._worker_last_seen < 45
            if alive or self._launching:
                return
            self._launching = True
            self._worker_state = "starting"
            self._worker_error = ""
        threading.Thread(target=self._launch_worker, name="kaggle-chat-launcher", daemon=True).start()

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

            worker = KaggleGpuWorker(api_token=token, username=username, timeout=120)
            try:
                status = worker.status(self.kernel_slug)
                if not status.terminal:
                    with self._lock:
                        self._worker_state = "starting"
                    return
            except Exception:
                pass

            source = self._worker_source(
                base_url="https://" + public_domain,
                worker_token=worker_token,
                model=model,
            )
            worker.submit_script(
                slug=self.kernel_slug,
                title="AI Agent Chat Session",
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

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                "job_id": job.job_id,
                "status": job.status,
                "text": job.text,
                "provider": job.provider,
                "model": job.model,
                "error": job.error,
                "worker_state": self._worker_state,
                "worker_error": self._worker_error,
            }

    def pull_job(self) -> dict | None:
        with self._lock:
            for job in sorted(self._jobs.values(), key=lambda j: j.created_at):
                if job.status == "pending":
                    job.status = "processing"
                    return {"job_id": job.job_id, "prompt": job.prompt}
        return None

    def heartbeat(self) -> None:
        with self._lock:
            self._worker_state = "running"
            self._worker_error = ""
            self._worker_last_seen = time.time()

    def finish_job(self, payload: dict) -> None:
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("job_id is required")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            error = str(payload.get("error", "")).strip()
            if error:
                job.status = "error"
                job.error = error[:2000]
            else:
                text = str(payload.get("text", "")).strip()
                if not text:
                    raise ValueError("text is required")
                job.status = "done"
                job.text = text
                job.provider = str(payload.get("provider", "kaggle-chat"))
                job.model = str(payload.get("model", ""))
            self._worker_state = "running"
            self._worker_last_seen = time.time()

    @staticmethod
    def _worker_source(*, base_url: str, worker_token: str, model: str) -> str:
        cfg = json.dumps({
            "base_url": base_url.rstrip("/"),
            "worker_token": worker_token,
            "model": model,
            "max_new_tokens": 512,
            "idle_polls": 60,
            "poll_seconds": 5,
        }, ensure_ascii=False)
        return textwrap.dedent(f"""
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
                        "User-Agent": "AI-Agent-Kaggle-Chat/1.0",
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
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model"],
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model.eval()
            request("POST", "/internal/chat/heartbeat", {{"state": "ready"}})

            idle = 0
            heartbeat_tick = 0
            while idle < int(CONFIG["idle_polls"]):
                job = request("GET", "/internal/chat/pull")
                if not job:
                    idle += 1
                    heartbeat_tick += 1
                    if heartbeat_tick >= 3:
                        request("POST", "/internal/chat/heartbeat", {{"state": "idle"}})
                        heartbeat_tick = 0
                    time.sleep(float(CONFIG["poll_seconds"]))
                    continue
                idle = 0
                heartbeat_tick = 0
                job_id = str(job["job_id"])
                prompt = str(job["prompt"])
                try:
                    messages = [{{"role": "user", "content": prompt}}]
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=int(CONFIG["max_new_tokens"]),
                        do_sample=False,
                        repetition_penalty=1.08,
                        no_repeat_ngram_size=3,
                    )
                    new_tokens = generated[:, inputs.input_ids.shape[1]:]
                    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                    if not text:
                        raise RuntimeError("model returned empty text")
                    request("POST", "/internal/chat/result", {{
                        "job_id": job_id,
                        "text": text,
                        "provider": "kaggle-chat-session",
                        "model": CONFIG["model"],
                    }})
                except Exception as exc:
                    request("POST", "/internal/chat/result", {{
                        "job_id": job_id,
                        "error": type(exc).__name__ + ": " + str(exc),
                    }})
            print("AI_AGENT_CHAT_SESSION_IDLE_EXIT")
        """).strip() + "\n"


CHAT_BROKER = ChatSessionBroker()
