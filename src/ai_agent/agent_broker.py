"""AIKA brain broker with explicit tool calling.

Production requests always reach AIKA first. AIKA may answer directly or emit a
structured call to one of its tools. This module never classifies user intent
with keywords and never contains image/video execution logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import threading
import time
import textwrap
from uuid import uuid4

from .brain_lifecycle import kernel_is_active, kernel_is_terminal, probe_kernel
from .core.kaggle_worker import KaggleGpuWorker


AIKA_SYSTEM_PROMPT = """You are AIKA, the AI assistant of the user's AI project.
Your name is AIKA. You are the brain; image and video generation are tools you may call.

Available tools:
- image: create one or more still images.
- video: create a video. The video tool may be upgraded internally without changing how you call it.

Rules:
1. Understand the user's request yourself. Do not rely on keyword routing.
2. If the user asks you to CREATE media, choose the appropriate tool yourself.
3. Questions, explanations, status checks, analysis, and discussion must be answered normally and must NOT call a media tool.
4. For a tool call, output ONLY one JSON object with exactly this shape:
   {"tool":"image","input":"faithful execution instruction"}
   or
   {"tool":"video","input":"faithful execution instruction"}
5. Preserve every named subject, count, identity, style, action, setting, and constraint from the user. Never substitute or invent a different subject.
6. Do not mention Kaggle, GPU, FLUX, Wan, LTX, routing, brokers, or implementation details to the user unless explicitly asked.
7. If no tool is needed, answer the user normally in their language. Do not wrap normal answers in JSON.
"""


@dataclass
class AgentJob:
    job_id: str
    prompt: str
    status: str = "pending"
    text: str = ""
    provider: str = ""
    model: str = ""
    error: str = ""
    created_at: float = 0.0
    decision: str = ""
    tool: str = ""


def _json_object(text: str) -> dict | None:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_tool_call(text: str) -> tuple[str, str] | None:
    """Parse AIKA's explicit tool decision; plain text is a normal answer."""
    payload = _json_object(text)
    if payload is None:
        return None
    tool = str(payload.get("tool", "")).strip().casefold()
    instruction = str(payload.get("input", "")).strip()
    if tool not in {"image", "video"}:
        return None
    if not instruction:
        raise ValueError("AIKA tool call is missing input")
    return tool, instruction


class AIKAAgentBroker:
    """One AIKA brain session; media execution is delegated to tool brokers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, AgentJob] = {}
        self._worker_state = "idle"
        self._worker_error = ""
        self._worker_last_seen = 0.0
        self._worker_started_at = 0.0
        self._launching = False
        self._monitoring = False
        self.kernel_slug = os.environ.get("AIKA_BRAIN_KERNEL_SLUG", "aika-brain-prod-v2").strip()
        self.kernel_title = os.environ.get("AIKA_BRAIN_KERNEL_TITLE", "AIKA Brain Prod V2").strip()
        self._startup_grace_seconds = float(os.environ.get("AIKA_BRAIN_STARTUP_GRACE_SECONDS", "900"))
        self._heartbeat_stale_seconds = float(os.environ.get("AIKA_BRAIN_HEARTBEAT_STALE_SECONDS", "45"))

    def create_job(self, prompt: str) -> AgentJob:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt is required")
        job = AgentJob(
            job_id=uuid4().hex,
            prompt=prompt,
            status="pending",
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            if len(self._jobs) > 60:
                oldest = sorted(self._jobs.values(), key=lambda item: item.created_at)[:-45]
                for item in oldest:
                    self._jobs.pop(item.job_id, None)
        self.ensure_worker()
        return job

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                "job_id": job.job_id,
                "kind": "agent",
                "status": job.status,
                "text": job.text,
                "provider": job.provider,
                "model": job.model,
                "error": job.error,
                "worker_state": self._worker_state,
                "worker_error": self._worker_error,
                "decision": job.decision,
                "tool": job.tool,
                "has_image": False,
                "image_count": 0,
                "image_urls": [],
                "image_url": "",
                "has_video": False,
                "video_url": "",
                "artifact_urls": [],
                "generation_prompt": "",
                "generation_prompts": [],
                "elapsed_seconds": 0.0,
                "per_artifact_seconds": [],
            }

    def pull_job(self) -> dict | None:
        with self._lock:
            for job in sorted(self._jobs.values(), key=lambda item: item.created_at):
                if job.status == "pending":
                    job.status = "processing"
                    return {"job_id": job.job_id, "prompt": job.prompt}
        return None

    def heartbeat(self, state: str = "ready") -> None:
        normalized = state.strip().casefold()
        now = time.time()
        with self._lock:
            if normalized in {"booting", "starting", "loading"}:
                self._worker_state = "starting"
                if self._worker_started_at <= 0:
                    self._worker_started_at = now
            elif normalized in {"stopped", "stopping", "idle_exit", "offline"}:
                self._worker_state = "idle"
                self._worker_started_at = 0.0
            else:
                self._worker_state = "running"
                self._worker_started_at = 0.0
            self._worker_error = ""
            self._worker_last_seen = now

    def finish_job(self, payload: dict) -> None:
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("job_id is required")
        error = str(payload.get("error", "")).strip()
        text = str(payload.get("text", "")).strip()

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if error:
                job.status = "error"
                job.error = error[:2000]
                self._worker_state = "running"
                self._worker_last_seen = time.time()
                return
            if not text:
                raise ValueError("text is required")

        call = parse_tool_call(text)
        if call is None:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "done"
                job.text = text
                job.provider = str(payload.get("provider", "kaggle-aika-brain"))
                job.model = str(payload.get("model", ""))
                job.decision = "respond"
                job.error = ""
                self._worker_state = "running"
                self._worker_last_seen = time.time()
            return

        tool, instruction = call
        try:
            from .media_broker import MEDIA_BROKER

            MEDIA_BROKER.create_job(tool, instruction, job_id=job_id)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "error"
                job.error = f"tool dispatch failed: {type(exc).__name__}: {exc}"[:2000]
                job.decision = "tool"
                job.tool = tool
            raise

        with self._lock:
            job = self._jobs[job_id]
            job.status = "delegated"
            job.text = f"AIKA đang dùng công cụ {tool}."
            job.provider = str(payload.get("provider", "kaggle-aika-brain"))
            job.model = str(payload.get("model", ""))
            job.decision = "tool"
            job.tool = tool
            job.error = ""
            self._worker_state = "running"
            self._worker_last_seen = time.time()

    def ensure_worker(self) -> None:
        now = time.time()
        with self._lock:
            heartbeat_fresh = (
                self._worker_last_seen > 0
                and now - self._worker_last_seen < self._heartbeat_stale_seconds
            )
            running_alive = self._worker_state == "running" and heartbeat_fresh
            starting_alive = self._worker_state == "starting" and (
                heartbeat_fresh
                or (
                    self._worker_started_at > 0
                    and now - self._worker_started_at < self._startup_grace_seconds
                )
            )
            if running_alive or starting_alive or self._launching:
                return
            self._launching = True
            self._worker_state = "starting"
            self._worker_started_at = now
            self._worker_error = ""
        threading.Thread(target=self._launch_worker, name="aika-brain-launcher", daemon=True).start()

    def _mark_pending_jobs_error(self, message: str) -> None:
        with self._lock:
            self._worker_state = "error"
            self._worker_error = message[:1000]
            self._worker_started_at = 0.0
            for job in self._jobs.values():
                if job.status == "pending":
                    job.status = "error"
                    job.error = self._worker_error

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
                raise RuntimeError(f"unexpected Kaggle brain state: {probe.status.status if probe.status else 'unknown'}")

            source = self._worker_source(
                base_url="https://" + public_domain,
                worker_token=worker_token,
                model=model,
            )
            try:
                worker.submit_script(
                    slug=self.kernel_slug,
                    title=self.kernel_title,
                    source=source,
                    enable_internet=True,
                    enable_gpu=True,
                    is_private=True,
                )
            except RuntimeError as exc:
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

    def _start_worker_monitor(self, worker: KaggleGpuWorker) -> None:
        with self._lock:
            if self._monitoring:
                return
            self._monitoring = True
        threading.Thread(
            target=self._monitor_worker_startup,
            args=(worker,),
            name="aika-brain-monitor",
            daemon=True,
        ).start()

    def _monitor_worker_startup(self, worker: KaggleGpuWorker) -> None:
        deadline = time.time() + self._startup_grace_seconds
        try:
            while time.time() < deadline:
                with self._lock:
                    state = self._worker_state
                    last_seen = self._worker_last_seen
                if state == "running" and last_seen > 0 and time.time() - last_seen < self._heartbeat_stale_seconds:
                    return

                time.sleep(15.0)
                try:
                    probe = probe_kernel(worker, self.kernel_slug)
                except Exception as exc:
                    self._mark_pending_jobs_error(f"Kaggle brain status check failed: {exc}")
                    return

                if not probe.exists:
                    self._mark_pending_jobs_error("Kaggle brain notebook disappeared during startup")
                    return
                if probe.active:
                    continue
                if probe.terminal:
                    status = probe.status.status if probe.status else "unknown"
                    detail = probe.status.failure_message if probe.status else ""
                    suffix = f": {detail}" if detail else ""
                    self._mark_pending_jobs_error(f"Kaggle brain stopped during startup ({status}){suffix}")
                    return
                status = probe.status.status if probe.status else "unknown"
                self._mark_pending_jobs_error(f"unexpected Kaggle brain state during startup: {status}")
                return

            self._mark_pending_jobs_error("Kaggle brain startup timed out before ready heartbeat")
        finally:
            with self._lock:
                self._monitoring = False

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
            import threading
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
                        "User-Agent": "AIKA-Brain/1.0",
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

            loading_done = threading.Event()

            def loading_heartbeat():
                while not loading_done.wait(15.0):
                    try:
                        request("POST", "/internal/chat/heartbeat", {{"state": "loading"}})
                    except Exception:
                        pass

            request("POST", "/internal/chat/heartbeat", {{"state": "booting"}})
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            heartbeat_thread = threading.Thread(target=loading_heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
                model = AutoModelForCausalLM.from_pretrained(
                    CONFIG["model"],
                    dtype=torch.float16,
                    device_map="auto",
                )
                model.eval()
            finally:
                loading_done.set()
                heartbeat_thread.join(timeout=2.0)

            request("POST", "/internal/chat/heartbeat", {{"state": "ready"}})

            idle = 0
            heartbeat_tick = 0
            try:
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
            finally:
                try:
                    request("POST", "/internal/chat/heartbeat", {{"state": "stopped"}})
                except Exception:
                    pass
            print("AIKA_BRAIN_SESSION_IDLE_EXIT")
        """).strip() + "\n"


AGENT_BROKER = AIKAAgentBroker()
