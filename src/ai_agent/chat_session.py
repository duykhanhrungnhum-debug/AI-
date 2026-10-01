"""Asynchronous AIKA chat sessions with direct skill routing."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
import threading
import time
import textwrap
from uuid import uuid4

from .core.kaggle_worker import KaggleGpuWorker
from .executors.image import ImageExecutor
from .router.skill_router import route_skill
from .workers.image_manager import WarmImageWorkerManager


AIKA_SYSTEM_PROMPT = (
    "You are AIKA, the AI assistant of the user's AI project. "
    "Your name is AIKA. If asked your name or identity, answer AIKA. "
    "Do not identify yourself as ChatGPT; ChatGPT is a separate assistant that can operate this project."
)


def classify_chat_intent(message: str) -> str:
    """Backward-compatible alias for the central skill router."""
    return route_skill(message)


@dataclass
class ChatJob:
    job_id: str
    prompt: str
    message: str = ""
    kind: str = "chat"
    status: str = "pending"
    text: str = ""
    provider: str = ""
    model: str = ""
    error: str = ""
    created_at: float = 0.0
    image_data: bytes | None = None
    image_mime: str = "image/png"
    generation_prompt: str = ""
    elapsed_seconds: float = 0.0
    image_datas: list[bytes] = field(default_factory=list)
    image_mimes: list[str] = field(default_factory=list)
    generation_prompts: list[str] = field(default_factory=list)
    image_elapsed_seconds: list[float] = field(default_factory=list)
    expected_images: int = 0


class ChatSessionBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ChatJob] = {}
        self._worker_state = "idle"
        self._worker_error = ""
        self._worker_last_seen = 0.0
        self._launching = False
        self.kernel_slug = "ai-agent-chat-session"
        self.image_kernel_slug = "ai-agent-image-v2"
        self._image_executor = ImageExecutor(kernel_slug=self.image_kernel_slug)
        self._warm_image = WarmImageWorkerManager()
        self._cold_fallbacks: set[str] = set()

    def create_job(self, prompt: str, *, message: str = "") -> ChatJob:
        prompt = prompt.strip()
        message = message.strip()
        if not prompt and not message:
            raise ValueError("prompt or message is required")
        if not prompt:
            prompt = message
        kind = classify_chat_intent(message) if message else "chat"
        job = ChatJob(
            job_id=uuid4().hex,
            prompt=prompt,
            message=message,
            kind=kind,
            status="pending",
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            if len(self._jobs) > 60:
                oldest = sorted(self._jobs.values(), key=lambda j: j.created_at)[:-45]
                for item in oldest:
                    self._jobs.pop(item.job_id, None)
        if kind == "image":
            if self._warm_image.ensure_started():
                threading.Thread(
                    target=self._watch_warm_image_job,
                    args=(job.job_id,),
                    name=f"aika-image-watch-{job.job_id[:8]}",
                    daemon=True,
                ).start()
            else:
                self._start_cold_fallback(job.job_id)
        else:
            self.ensure_worker()
        return job

    def _clear_image_outputs_locked(self, job: ChatJob) -> None:
        job.image_data = None
        job.image_mime = "image/png"
        job.generation_prompt = ""
        job.elapsed_seconds = 0.0
        job.image_datas.clear()
        job.image_mimes.clear()
        job.generation_prompts.clear()
        job.image_elapsed_seconds.clear()
        job.expected_images = 0

    def _start_cold_fallback(self, job_id: str, *, reason: str = "") -> None:
        if reason:
            self._warm_image.invalidate_job(job_id, reason)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {"done", "error"} or job_id in self._cold_fallbacks:
                return
            self._cold_fallbacks.add(job_id)
            self._clear_image_outputs_locked(job)
            job.status = "processing"
        threading.Thread(
            target=self._run_image_job,
            args=(job_id,),
            name=f"aika-image-cold-{job_id[:8]}",
            daemon=True,
        ).start()

    def _watch_warm_image_job(self, job_id: str) -> None:
        """Watch leases, not elapsed generation time, and fall back only on failure."""
        hard_timeout = float(os.environ.get("AIKA_WARM_IMAGE_JOB_TIMEOUT", "900"))
        hard_deadline = time.time() + max(300.0, hard_timeout)
        while time.time() < hard_deadline:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status in {"done", "error"}:
                    return
                status = job.status

            if status == "processing":
                if self._warm_image.job_lease_alive(job_id):
                    time.sleep(5)
                    continue
                self._start_cold_fallback(
                    job_id,
                    reason="warm job lease expired while processing",
                )
                return

            snapshot = self._warm_image.snapshot()
            if snapshot.state == "error" or (
                snapshot.circuit_open_until and snapshot.circuit_open_until > time.time()
            ):
                self._start_cold_fallback(job_id)
                return

            # A pending job may have been queued while the sole warm worker is
            # busy with an earlier job. ensure_started() is lease-aware and must
            # never launch a second worker while the current session/job lease lives.
            if not self._warm_image.ensure_started():
                self._start_cold_fallback(job_id)
                return
            time.sleep(5)

        self._start_cold_fallback(
            job_id,
            reason=f"warm image job exceeded hard timeout {hard_timeout:.0f}s",
        )

    def _run_image_job(self, job_id: str) -> None:
        try:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status == "done":
                    return
                command = job.message or job.prompt
            results = self._image_executor.execute_command_cold(
                command,
                width=1024,
                height=1024,
                max_images=6,
            )
            if not results:
                raise RuntimeError("cold image executor returned no images")
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status == "done":
                    return
                self._clear_image_outputs_locked(job)
                job.status = "done"
                job.text = (
                    "AIKA đã tạo ảnh xong."
                    if len(results) == 1
                    else f"AIKA đã tạo xong {len(results)} ảnh riêng."
                )
                job.provider = "kaggle-image-v2"
                job.model = results[0].model
                job.image_datas = [result.data for result in results]
                job.image_mimes = ["image/png" for _ in results]
                job.generation_prompts = [result.prompt for result in results]
                job.image_elapsed_seconds = [result.elapsed_seconds for result in results]
                job.expected_images = len(results)
                job.image_data = results[0].data
                job.image_mime = "image/png"
                job.generation_prompt = results[0].prompt
                job.elapsed_seconds = sum(result.elapsed_seconds for result in results)
                job.error = ""
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None and job.status != "done":
                    job.status = "error"
                    job.error = f"{type(exc).__name__}: {exc}"[:2000]
        finally:
            with self._lock:
                self._cold_fallbacks.discard(job_id)

    def pull_image_job(self, session_id: str) -> dict | None:
        """Lease exactly one pending image job to the authenticated warm session."""
        session_id = session_id.strip()
        if not session_id:
            raise PermissionError("worker session id is required")
        with self._lock:
            candidates = [
                job.job_id
                for job in sorted(self._jobs.values(), key=lambda j: j.created_at)
                if job.kind == "image" and job.status == "pending"
            ]

        for job_id in candidates:
            if not self._warm_image.claim_job(session_id, job_id):
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status != "pending":
                    self._warm_image.release_job(session_id, job_id)
                    continue
                job.status = "processing"
                command = job.message or job.prompt
                return {
                    "job_id": job.job_id,
                    "command": command,
                    "seed": ImageExecutor.seed_for(command),
                    "width": 1024,
                    "height": 1024,
                }
        return None

    def image_heartbeat(
        self,
        session_id: str,
        state: str = "ready",
        *,
        current_job_id: str = "",
    ) -> None:
        if not self._warm_image.heartbeat(
            session_id,
            state,
            current_job_id=current_job_id,
        ):
            raise PermissionError("stale warm worker session")

    def finish_image_job(self, payload: dict, *, session_id: str) -> None:
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("job_id is required")
        if not self._warm_image.validate_job_session(session_id, job_id):
            raise PermissionError("stale warm worker or expired job lease")

        error = str(payload.get("error", "")).strip()
        if error:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    raise KeyError(job_id)
                if job.status == "done":
                    return
                job.status = "pending"
                job.error = error[:2000]
            self._warm_image.release_job(session_id, job_id)
            self._start_cold_fallback(job_id)
            return

        encoded = str(payload.get("image_b64", "")).strip()
        if not encoded:
            raise ValueError("image_b64 is required")
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("image_b64 is invalid") from exc
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("warm image result is not PNG")

        index = int(payload.get("image_index", 0))
        total = int(payload.get("image_total", 1))
        if total <= 0 or total > 8 or index < 0 or index >= total:
            raise ValueError("invalid warm image index/total")

        complete = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == "done":
                return
            if job.expected_images not in {0, total}:
                raise ValueError("warm image total changed during job")
            if job.expected_images == 0:
                job.expected_images = total
                job.image_datas = [b"" for _ in range(total)]
                job.image_mimes = ["image/png" for _ in range(total)]
                job.generation_prompts = ["" for _ in range(total)]
                job.image_elapsed_seconds = [0.0 for _ in range(total)]
            job.provider = str(payload.get("provider", "kaggle-image-warm"))
            job.model = str(payload.get("model", ""))
            job.image_datas[index] = data
            job.image_mimes[index] = str(payload.get("image_mime", "image/png"))
            job.generation_prompts[index] = str(payload.get("generation_prompt", ""))
            job.image_elapsed_seconds[index] = float(payload.get("elapsed_seconds") or 0.0)
            job.error = ""
            if index == 0:
                job.image_data = data
                job.image_mime = job.image_mimes[index]
                job.generation_prompt = job.generation_prompts[index]
            if all(job.image_datas):
                job.status = "done"
                job.elapsed_seconds = sum(job.image_elapsed_seconds)
                job.text = (
                    "AIKA đã tạo ảnh xong."
                    if total == 1
                    else f"AIKA đã tạo xong {total} ảnh riêng."
                )
                complete = True
        if complete:
            self._warm_image.release_job(session_id, job_id)

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

            worker = KaggleGpuWorker(
                api_token=token,
                username=username,
                timeout=120,
                submission_retry_attempts=2,
                submission_retry_delay_seconds=10,
            )
            launch_started = time.time()
            try:
                status = worker.status(self.kernel_slug)
                if not status.terminal:
                    deadline = time.time() + 45
                    while time.time() < deadline:
                        with self._lock:
                            if self._worker_last_seen >= launch_started:
                                return
                        time.sleep(5)
                        try:
                            status = worker.status(self.kernel_slug)
                            if status.terminal:
                                break
                        except Exception:
                            break
            except Exception:
                pass

            source = self._worker_source(
                base_url="https://" + public_domain,
                worker_token=worker_token,
                model=model,
            )
            worker.submit_script(
                slug=self.kernel_slug,
                title="AIKA Chat Session",
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
                    if job.kind == "chat" and job.status == "pending":
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
            if job.kind == "image" and self._warm_image.enabled:
                image_snapshot = self._warm_image.snapshot()
                image_state = image_snapshot.state
                image_error = self._warm_image.last_error()
            else:
                image_snapshot = None
                image_state = job.status
                image_error = ""
            complete_images = len(job.image_datas) if job.status == "done" and job.image_datas else (1 if job.status == "done" and job.image_data else 0)
            image_urls = [
                f"/v1/chat/image?job_id={job.job_id}&index={index}"
                for index in range(complete_images)
            ]
            return {
                "job_id": job.job_id,
                "kind": job.kind,
                "status": job.status,
                "text": job.text,
                "provider": job.provider,
                "model": job.model,
                "error": job.error,
                "has_image": bool(image_urls),
                "image_count": len(image_urls),
                "image_urls": image_urls,
                "image_url": image_urls[0] if image_urls else "",
                "generation_prompt": job.generation_prompt,
                "generation_prompts": list(job.generation_prompts),
                "elapsed_seconds": job.elapsed_seconds,
                "worker_state": self._worker_state if job.kind == "chat" else image_state,
                "worker_error": self._worker_error if job.kind == "chat" else image_error,
                "warm_image_enabled": self._warm_image.enabled if job.kind == "image" else False,
                "worker_session_id": image_snapshot.session_id if image_snapshot else "",
                "worker_current_job_id": image_snapshot.current_job_id if image_snapshot else "",
            }

    def get_image(self, job_id: str, *, index: int = 0) -> tuple[bytes, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != "done":
                raise ValueError("image is not ready")
            if job.image_datas:
                if index < 0 or index >= len(job.image_datas):
                    raise IndexError(index)
                return job.image_datas[index], job.image_mimes[index]
            if index != 0 or not job.image_data:
                raise IndexError(index)
            return job.image_data, job.image_mime

    def pull_job(self) -> dict | None:
        with self._lock:
            for job in sorted(self._jobs.values(), key=lambda j: j.created_at):
                if job.kind == "chat" and job.status == "pending":
                    job.status = "processing"
                    return {"job_id": job.job_id, "prompt": job.prompt}
        return None

    def heartbeat(self, state: str = "ready") -> None:
        normalized = state.strip().lower()
        with self._lock:
            self._worker_state = "starting" if normalized in {"booting", "starting", "loading"} else "running"
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
            if job.kind != "chat":
                raise ValueError("only chat jobs can be finished by the chat worker")
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
            "system_prompt": AIKA_SYSTEM_PROMPT,
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
                        "User-Agent": "AIKA-Kaggle-Chat/1.0",
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

            request("POST", "/internal/chat/heartbeat", {{"state": "booting"}})
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
                    messages = [
                        {{"role": "system", "content": CONFIG["system_prompt"]}},
                        {{"role": "user", "content": prompt}},
                    ]
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
            print("AIKA_CHAT_SESSION_IDLE_EXIT")
        """).strip() + "\n"


CHAT_BROKER = ChatSessionBroker()
