"""Warm Kaggle image-worker lifecycle for AIKA.

The warm path is additive and gated by AIKA_IMAGE_WARM_WORKER. The verified cold
Image V2 provider remains the fallback until the warm path is promoted.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import json
import os
import textwrap
import threading
import time

from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_v2 import (
    IMAGE_BATCH_DELIMITER,
    IMAGE_GUIDANCE,
    IMAGE_MODEL,
    IMAGE_STEPS,
    RECAPTION_MODEL,
)


class WorkerState(str, Enum):
    OFFLINE = "offline"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass(frozen=True)
class WorkerSnapshot:
    enabled: bool
    state: str
    stage: str
    last_seen: float
    failures: int
    circuit_open_until: float
    kernel_slug: str
    launch_count: int


class WarmImageWorkerManager:
    """Launch one reusable Kaggle image worker and monitor its heartbeat."""

    def __init__(
        self,
        *,
        kernel_slug: str = "ai-agent-image-warm",
        idle_seconds: int = 300,
        heartbeat_ttl: int = 60,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: int = 300,
        max_images_per_job: int = 6,
    ) -> None:
        if idle_seconds <= 0:
            raise ValueError("idle_seconds must be positive")
        if heartbeat_ttl <= 0:
            raise ValueError("heartbeat_ttl must be positive")
        if max_images_per_job <= 0 or max_images_per_job > 8:
            raise ValueError("max_images_per_job must be between 1 and 8")
        self.kernel_slug = kernel_slug
        self.idle_seconds = idle_seconds
        self.heartbeat_ttl = heartbeat_ttl
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds
        self.max_images_per_job = max_images_per_job
        self._lock = threading.Lock()
        self._state = WorkerState.OFFLINE
        self._stage = WorkerState.OFFLINE.value
        self._last_seen = 0.0
        self._failures = 0
        self._circuit_open_until = 0.0
        self._launching = False
        self._last_error = ""
        self._launch_count = 0

    @property
    def enabled(self) -> bool:
        return os.environ.get("AIKA_IMAGE_WARM_WORKER", "").strip().casefold() in {
            "1", "true", "yes", "on"
        }

    def snapshot(self) -> WorkerSnapshot:
        with self._lock:
            return WorkerSnapshot(
                enabled=self.enabled,
                state=self._state.value,
                stage=self._stage,
                last_seen=self._last_seen,
                failures=self._failures,
                circuit_open_until=self._circuit_open_until,
                kernel_slug=self.kernel_slug,
                launch_count=self._launch_count,
            )

    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def is_healthy(self) -> bool:
        with self._lock:
            return (
                self.enabled
                and self._state in {WorkerState.READY, WorkerState.BUSY, WorkerState.IDLE}
                and time.time() - self._last_seen < self.heartbeat_ttl
            )

    def ensure_started(self) -> bool:
        """Start/reuse the warm worker. Returns False when warm mode is unavailable."""
        if not self.enabled:
            return False
        now = time.time()
        with self._lock:
            if self._circuit_open_until > now:
                return False
            alive = (
                self._state in {WorkerState.STARTING, WorkerState.READY, WorkerState.BUSY, WorkerState.IDLE}
                and (self._state == WorkerState.STARTING or now - self._last_seen < self.heartbeat_ttl)
            )
            if alive or self._launching:
                return True
            self._launching = True
            self._state = WorkerState.STARTING
            self._stage = WorkerState.STARTING.value
            self._last_error = ""
        threading.Thread(target=self._launch, name="aika-image-warm-launch", daemon=True).start()
        return True

    def heartbeat(self, state: str) -> None:
        raw = state.strip()
        normalized = raw.casefold()
        if normalized.startswith("error:"):
            detail = raw.split(":", 1)[1].strip() or "warm worker startup failed"
            self.record_failure(detail)
            with self._lock:
                self._last_seen = time.time()
                self._stage = normalized[:2000]
            return
        mapping = {
            "booting": WorkerState.STARTING,
            "supervisor_starting": WorkerState.STARTING,
            "child_booting": WorkerState.STARTING,
            "repairing_dependencies": WorkerState.STARTING,
            "installing_dependencies": WorkerState.STARTING,
            "dependencies_ready": WorkerState.STARTING,
            "importing_torch": WorkerState.STARTING,
            "torch_ready": WorkerState.STARTING,
            "importing_diffusers": WorkerState.STARTING,
            "diffusers_ready": WorkerState.STARTING,
            "importing_transformers": WorkerState.STARTING,
            "transformers_ready": WorkerState.STARTING,
            "loading_models": WorkerState.STARTING,
            "ready": WorkerState.READY,
            "busy": WorkerState.BUSY,
            "idle": WorkerState.IDLE,
            "shutting_down": WorkerState.SHUTTING_DOWN,
            "offline": WorkerState.OFFLINE,
        }
        with self._lock:
            self._state = mapping.get(normalized, WorkerState.STARTING)
            self._stage = normalized or WorkerState.STARTING.value
            self._last_seen = time.time()
            if self._state in {WorkerState.READY, WorkerState.BUSY, WorkerState.IDLE}:
                self._failures = 0
                self._circuit_open_until = 0.0
                self._last_error = ""

    def record_failure(self, error: str) -> None:
        with self._lock:
            self._failures += 1
            self._state = WorkerState.ERROR
            self._stage = WorkerState.ERROR.value
            self._last_error = error[:2000]
            if self._failures >= self.circuit_failure_threshold:
                self._circuit_open_until = time.time() + self.circuit_cooldown_seconds

    def mark_offline(self) -> None:
        with self._lock:
            self._state = WorkerState.OFFLINE
            self._stage = WorkerState.OFFLINE.value
            self._last_seen = time.time()

    def _launch(self) -> None:
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
                submission_retry_attempts=3,
                submission_retry_delay_seconds=20,
            )
            launch_started = time.time()
            try:
                status = worker.status(self.kernel_slug)
                if not status.terminal:
                    deadline = time.time() + 45
                    while time.time() < deadline:
                        with self._lock:
                            if self._last_seen >= launch_started:
                                return
                        time.sleep(5)
                        status = worker.status(self.kernel_slug)
                        if status.terminal:
                            break
            except Exception:
                pass

            launch_title = f"AIKA Warm Image Worker {time.time_ns()}"
            worker.submit_script(
                slug=self.kernel_slug,
                title=launch_title,
                source=self._worker_source(
                    base_url="https://" + public_domain,
                    worker_token=worker_token,
                ),
                enable_internet=True,
                enable_gpu=True,
                is_private=True,
            )
            with self._lock:
                self._launch_count += 1
        except Exception as exc:
            self.record_failure(f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._launching = False

    def _worker_source(self, *, base_url: str, worker_token: str) -> str:
        config = {
            "base_url": base_url.rstrip("/"),
            "worker_token": worker_token,
            "image_model": IMAGE_MODEL,
            "recaption_model": RECAPTION_MODEL,
            "steps": IMAGE_STEPS,
            "guidance": IMAGE_GUIDANCE,
            "idle_seconds": self.idle_seconds,
            "poll_seconds": 3,
            "max_images": self.max_images_per_job,
            "delimiter": IMAGE_BATCH_DELIMITER,
        }
        config_json = json.dumps(config, ensure_ascii=False)

        child_template = r'''
from __future__ import annotations

import base64
from io import BytesIO
import gc
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

CONFIG = json.loads(__CONFIG_JSON__)
STAGE_FILE = Path("/tmp/aika_warm_stage")
READY_FILE = Path("/tmp/aika_warm_ready")
ERROR_FILE = Path("/tmp/aika_warm_error")


def request(method, path, payload=None, *, timeout=60):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        CONFIG["base_url"] + path,
        data=body,
        method=method,
        headers={
            "Authorization": "Bearer " + CONFIG["worker_token"],
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AIKA-Warm-Image/2.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as exc:
        if exc.code == 204:
            return None
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail[:1000]}") from exc


def signal(state):
    try:
        STAGE_FILE.write_text(state, encoding="utf-8")
    except Exception:
        pass
    for attempt in range(3):
        try:
            request("POST", "/internal/image/heartbeat", {"state": state}, timeout=5)
            return True
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return False


def startup_error(stage, exc):
    detail = (type(exc).__name__ + ": " + str(exc)).replace("\n", " ")[:700]
    try:
        ERROR_FILE.write_text(stage + ":" + detail, encoding="utf-8")
    except Exception:
        pass
    signal("error:" + stage + ":" + detail)


def recaption_batch(command):
    delimiter = CONFIG["delimiter"]
    contract = (
        "Preserve exactly the requested subject or species, number of subjects, visual style, setting, framing, "
        "important attributes, and explicit exclusions. Preserve culturally specific names instead of substituting an "
        "item from another culture. Vietnamese 'áo dài' must remain Vietnamese áo dài and preserve its long tunic panels "
        "over separate trousers when requested. Do not generalize a named subject. "
    )
    instruction = (
        "Convert the USER REQUEST into faithful English image-generation descriptions. " + contract +
        "Only create multiple descriptions when the user explicitly asks for multiple separate images, pictures, files, "
        "or variants. Multiple subjects requested together in one image must stay in ONE description. When multiple "
        "separate images are requested, output exactly one complete description per image in the user's requested order, "
        "separated only by the exact delimiter " + delimiter + ". Do not number or label the descriptions. Do not output "
        "JSON, explanation, scoring, or commentary. Never exceed " + str(CONFIG["max_images"]) + " descriptions.\n"
        "USER REQUEST: " + command
    )
    recaptioner.to("cuda")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = recaptioner.generate(
            **inputs,
            max_new_tokens=640,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.03,
        )
    planned = tokenizer.batch_decode(
        generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip().strip('"')
    del generated, inputs
    recaptioner.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    prompts = [part.strip().strip('"') for part in planned.split(delimiter) if part.strip()]
    if not prompts:
        raise RuntimeError("recaption returned no image descriptions")
    if len(prompts) > int(CONFIG["max_images"]):
        raise RuntimeError("recaption exceeded max_images")
    return prompts


signal("child_booting")
try:
    signal("importing_torch")
    import torch
    signal("torch_ready")

    signal("importing_diffusers")
    from diffusers import Flux2KleinPipeline
    signal("diffusers_ready")

    signal("importing_transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    signal("transformers_ready")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    signal("loading_models")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["recaption_model"])
    recaptioner = AutoModelForCausalLM.from_pretrained(
        CONFIG["recaption_model"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    recaptioner.eval()
    pipe = Flux2KleinPipeline.from_pretrained(CONFIG["image_model"], torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
except Exception as exc:
    startup_error("startup", exc)
    raise

READY_FILE.write_text("ready", encoding="utf-8")
signal("ready")

idle_started = time.monotonic()
while time.monotonic() - idle_started < float(CONFIG["idle_seconds"]):
    try:
        job = request("GET", "/internal/image/pull", timeout=30)
    except Exception:
        time.sleep(float(CONFIG["poll_seconds"]))
        continue
    if not job:
        signal("idle")
        time.sleep(float(CONFIG["poll_seconds"]))
        continue

    idle_started = time.monotonic()
    job_id = str(job["job_id"])
    command = str(job["command"])
    seed = int(job["seed"])
    width = int(job.get("width", 1024))
    height = int(job.get("height", 1024))
    signal("busy")
    try:
        prompts = recaption_batch(command)
        for index, prompt in enumerate(prompts):
            item_started = time.perf_counter()
            generator = torch.Generator(device="cpu").manual_seed((seed + index) % (2 ** 32))
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=float(CONFIG["guidance"]),
                num_inference_steps=int(CONFIG["steps"]),
                generator=generator,
            ).images[0]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            request("POST", "/internal/image/result", {
                "job_id": job_id,
                "provider": "kaggle-image-warm",
                "model": CONFIG["image_model"],
                "image_index": index,
                "image_total": len(prompts),
                "image_mime": "image/png",
                "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "generation_prompt": prompt,
                "elapsed_seconds": round(time.perf_counter() - item_started, 3),
            }, timeout=180)
    except Exception as exc:
        try:
            recaptioner.to("cpu")
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            request("POST", "/internal/image/result", {
                "job_id": job_id,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }, timeout=60)
        except Exception:
            pass
    signal("ready")

signal("shutting_down")
'''
        child_source = (
            textwrap.dedent(child_template)
            .replace("__CONFIG_JSON__", repr(config_json))
            .strip()
            + "\n"
        )
        child_b64 = base64.b64encode(child_source.encode("utf-8")).decode("ascii")

        supervisor_template = r'''
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import signal as signal_module
import subprocess
import sys
import time
import urllib.error
import urllib.request

CONFIG = json.loads(__CONFIG_JSON__)
CHILD_SOURCE = base64.b64decode(__CHILD_B64__).decode("utf-8")
CHILD_PATH = Path("/tmp/aika_warm_child.py")
READY_FILE = Path("/tmp/aika_warm_ready")
ERROR_FILE = Path("/tmp/aika_warm_error")
STAGE_FILE = Path("/tmp/aika_warm_stage")
LOG_FILE = Path("/tmp/aika_warm_child.log")


def request(method, path, payload=None, *, timeout=60):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        CONFIG["base_url"] + path,
        data=body,
        method=method,
        headers={
            "Authorization": "Bearer " + CONFIG["worker_token"],
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AIKA-Warm-Supervisor/2.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as exc:
        if exc.code == 204:
            return None
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail[:1000]}") from exc


def signal(state):
    for attempt in range(3):
        try:
            request("POST", "/internal/image/heartbeat", {"state": state}, timeout=5)
            return True
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return False


def report_error(stage, detail):
    clean = str(detail).replace("\n", " ")[:700]
    signal("error:" + stage + ":" + clean)


def kill_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal_module.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def hard_run(command, *, timeout, label):
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + float(timeout)
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            kill_group(proc)
            raise TimeoutError(f"{label} exceeded {timeout} seconds")
        time.sleep(1)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} exited with code {proc.returncode}")


def install_dependencies():
    signal("repairing_dependencies")
    hard_run([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "diffusers", "transformers>=4.57,<5", "accelerate<2", "safetensors",
        "sentencepiece", "Pillow<13",
    ], timeout=300, label="dependency repair")
    signal("dependencies_ready")


def stage_name():
    try:
        return STAGE_FILE.read_text(encoding="utf-8").strip() or "child_booting"
    except Exception:
        return "child_booting"


def stage_budget(stage):
    if stage in {"child_booting", "importing_torch"}:
        return 120
    if stage in {"importing_diffusers", "importing_transformers"}:
        return 120
    if stage == "loading_models":
        return 360
    return 420


def start_child():
    for path in (READY_FILE, ERROR_FILE, STAGE_FILE):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    LOG_FILE.write_text("", encoding="utf-8")
    log_handle = LOG_FILE.open("ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, str(CHILD_PATH)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_handle


def wait_ready(proc):
    current_stage = "child_booting"
    stage_started = time.monotonic()
    while True:
        if READY_FILE.exists():
            return True, "ready"
        if ERROR_FILE.exists():
            try:
                return False, ERROR_FILE.read_text(encoding="utf-8")[-1500:]
            except Exception:
                return False, "child startup error"
        if proc.poll() is not None:
            try:
                tail = LOG_FILE.read_text(encoding="utf-8", errors="replace")[-3000:]
            except Exception:
                tail = ""
            return False, f"child exited {proc.returncode}: {tail}"

        observed = stage_name()
        if observed != current_stage:
            current_stage = observed
            stage_started = time.monotonic()
        budget = stage_budget(current_stage)
        if time.monotonic() - stage_started > budget:
            kill_group(proc)
            return False, f"stage timeout: {current_stage} exceeded {budget}s"
        time.sleep(2)


signal("supervisor_starting")
CHILD_PATH.write_text(CHILD_SOURCE, encoding="utf-8")
last_error = ""
child = None
handle = None
for attempt in range(2):
    child, handle = start_child()
    ok, detail = wait_ready(child)
    if ok:
        break
    last_error = detail
    kill_group(child)
    try:
        handle.close()
    except Exception:
        pass
    if attempt == 0:
        try:
            install_dependencies()
        except Exception as exc:
            report_error("dependencies", f"{type(exc).__name__}: {exc}")
            raise
else:
    report_error("child", last_error or "warm child failed to become ready")
    raise RuntimeError(last_error or "warm child failed to become ready")

try:
    while child.poll() is None:
        time.sleep(5)
    if child.returncode != 0:
        try:
            tail = LOG_FILE.read_text(encoding="utf-8", errors="replace")[-3000:]
        except Exception:
            tail = ""
        report_error("runtime", f"child exited {child.returncode}: {tail}")
        raise RuntimeError(f"warm child exited {child.returncode}")
finally:
    try:
        handle.close()
    except Exception:
        pass
'''
        return (
            textwrap.dedent(supervisor_template)
            .replace("__CONFIG_JSON__", repr(config_json))
            .replace("__CHILD_B64__", repr(child_b64))
            .strip()
            + "\n"
        )
