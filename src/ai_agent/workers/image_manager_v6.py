"""AIKA Warm Image Worker v6 canary runtime.

V6 keeps the verified image models but gives the GPU a single owner:
Qwen3-1.7B stays CPU-resident for recaption and FLUX remains the only model
allowed to execute on CUDA.  The runtime also reports explicit job stages so a
stalled recaption/generation/upload is invalidated instead of waiting silently.

This module is canary-only until the real Kaggle E2E passes reuse + batch.
"""
from __future__ import annotations

import json
import textwrap
import time

from .image_manager import WarmImageWorkerManager, WorkerState
from ai_agent.core.media_v2 import (
    IMAGE_BATCH_DELIMITER,
    IMAGE_GUIDANCE,
    IMAGE_MODEL,
    IMAGE_STEPS,
    RECAPTION_MODEL,
)


class WarmImageWorkerManagerV6(WarmImageWorkerManager):
    """Warm manager with explicit stage watchdogs and CPU-only recaption."""

    STAGE_TIMEOUTS = {
        "recaption": 180.0,
        "generate": 120.0,
        "upload": 90.0,
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stage_started_at = 0.0

    @staticmethod
    def _stage_family(stage: str) -> str:
        return (stage or "").split(":", 1)[0].strip().casefold()

    def _set_stage_locked(self, stage: str, now: float) -> None:
        stage = (stage or self._state.value).strip().casefold()
        if stage != self._stage:
            self._stage_started_at = now
        elif not self._stage_started_at:
            self._stage_started_at = now
        self._stage = stage

    def _clear_session_locked(self) -> None:
        super()._clear_session_locked()
        self._stage_started_at = 0.0

    def _expire_stale_locked(self, now: float) -> None:
        super()._expire_stale_locked(now)
        if self._state is not WorkerState.BUSY or not self._session_id:
            return
        family = self._stage_family(self._stage)
        timeout = self.STAGE_TIMEOUTS.get(family)
        if not timeout or not self._stage_started_at:
            return
        age = now - self._stage_started_at
        if age >= timeout:
            job_id = self._current_job_id
            self._record_failure_locked(
                f"warm stage {self._stage} exceeded {timeout:.0f}s"
                + (f" for {job_id}" if job_id else ""),
                stage=f"{family}_timeout",
            )

    def heartbeat(
        self,
        session_id: str,
        state: str,
        *,
        current_job_id: str = "",
    ) -> bool:
        raw = (state or "ready").strip().casefold()
        if raw.startswith("busy:"):
            stage = raw.split(":", 1)[1].strip() or "busy"
            accepted = super().heartbeat(
                session_id,
                "busy",
                current_job_id=current_job_id,
            )
            if not accepted:
                return False
            now = time.time()
            with self._lock:
                if session_id != self._session_id:
                    return False
                self._set_stage_locked(stage, now)
            return True

        accepted = super().heartbeat(
            session_id,
            raw,
            current_job_id=current_job_id,
        )
        if not accepted:
            return False
        now = time.time()
        with self._lock:
            if session_id == self._session_id:
                self._set_stage_locked(raw, now)
        return True

    def claim_job(self, session_id: str, job_id: str) -> bool:
        accepted = super().claim_job(session_id, job_id)
        if accepted:
            with self._lock:
                self._set_stage_locked("claimed", time.time())
        return accepted

    def release_job(self, session_id: str, job_id: str) -> bool:
        accepted = super().release_job(session_id, job_id)
        if accepted:
            with self._lock:
                self._set_stage_locked("ready", time.time())
        return accepted

    def _worker_source(
        self,
        *,
        base_url: str,
        worker_token: str,
        session_id: str = "test-session",
    ) -> str:
        config = json.dumps({
            "base_url": base_url.rstrip("/"),
            "worker_token": worker_token,
            "session_id": session_id,
            "image_model": IMAGE_MODEL,
            "recaption_model": RECAPTION_MODEL,
            "steps": IMAGE_STEPS,
            "guidance": IMAGE_GUIDANCE,
            "idle_seconds": self.idle_seconds,
            "poll_seconds": 3,
            "heartbeat_seconds": 10,
            "max_images": self.max_images_per_job,
            "delimiter": IMAGE_BATCH_DELIMITER,
        }, ensure_ascii=False)

        template = r'''
from __future__ import annotations

import base64
from io import BytesIO
import gc
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

CONFIG = json.loads(__CONFIG_JSON__)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
except (ImportError, AttributeError):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "diffusers", "transformers>=4.57,<5", "accelerate<2", "safetensors",
        "sentencepiece", "Pillow<13",
    ])
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required")

# Qwen stays CPU-resident for the whole session. It never competes with FLUX
# for CUDA memory. This costs some CPU latency but removes the failure mode that
# repeatedly stalled recaption when FLUX was already warm.
tokenizer = AutoTokenizer.from_pretrained(CONFIG["recaption_model"])
recaptioner = AutoModelForCausalLM.from_pretrained(
    CONFIG["recaption_model"],
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
recaptioner.eval()

# FLUX is the sole GPU owner and remains warm across jobs.
pipe = Flux2KleinPipeline.from_pretrained(
    CONFIG["image_model"],
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()


def request(method, path, payload=None, *, timeout=60):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        CONFIG["base_url"] + path,
        data=body,
        method=method,
        headers={
            "Authorization": "Bearer " + CONFIG["worker_token"],
            "X-AIKA-Worker-Session": CONFIG["session_id"],
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AIKA-Warm-Image/6.0",
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


state_lock = threading.Lock()
worker_state = "ready"
current_job_id = ""
stop_heartbeat = threading.Event()


def set_state(state, job_id=""):
    global worker_state, current_job_id
    with state_lock:
        worker_state = state
        current_job_id = job_id


def current_state():
    with state_lock:
        return worker_state, current_job_id


def signal(state=None, job_id=None):
    if state is None or job_id is None:
        live_state, live_job = current_state()
        state = live_state if state is None else state
        job_id = live_job if job_id is None else job_id
    payload = {"state": state, "current_job_id": job_id}
    for attempt in range(3):
        try:
            request("POST", "/internal/image/heartbeat", payload, timeout=5)
            return True
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return False


def heartbeat_loop():
    while not stop_heartbeat.wait(float(CONFIG["heartbeat_seconds"])):
        signal()


def render_recaption(command):
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
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Inputs and Qwen remain on CPU by design.
    inputs = tokenizer([rendered], return_tensors="pt")
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
    gc.collect()
    prompts = [part.strip().strip('"') for part in planned.split(delimiter) if part.strip()]
    if not prompts:
        raise RuntimeError("recaption returned no image descriptions")
    if len(prompts) > int(CONFIG["max_images"]):
        raise RuntimeError("recaption exceeded max_images")
    return prompts


set_state("ready", "")
if not signal("ready", ""):
    raise RuntimeError("could not register warm worker session")
heartbeat_thread = threading.Thread(target=heartbeat_loop, name="aika-warm-heartbeat", daemon=True)
heartbeat_thread.start()

idle_started = time.monotonic()
last_idle_signal = 0.0
try:
    while time.monotonic() - idle_started < float(CONFIG["idle_seconds"]):
        try:
            job = request("GET", "/internal/image/pull", timeout=30)
        except Exception:
            time.sleep(float(CONFIG["poll_seconds"]))
            continue
        if not job:
            now = time.monotonic()
            if now - last_idle_signal >= 15:
                set_state("idle", "")
                signal("idle", "")
                last_idle_signal = now
            time.sleep(float(CONFIG["poll_seconds"]))
            continue

        idle_started = time.monotonic()
        last_idle_signal = 0.0
        job_id = str(job["job_id"])
        command = str(job["command"])
        seed = int(job["seed"])
        width = int(job.get("width", 1024))
        height = int(job.get("height", 1024))

        try:
            set_state("busy:recaption", job_id)
            signal("busy:recaption", job_id)
            prompts = render_recaption(command)
            signal("busy:recaption", job_id)

            total = len(prompts)
            for index, prompt in enumerate(prompts):
                stage = f"busy:generate:{index + 1}/{total}"
                set_state(stage, job_id)
                signal(stage, job_id)
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

                stage = f"busy:upload:{index + 1}/{total}"
                set_state(stage, job_id)
                signal(stage, job_id)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                request("POST", "/internal/image/result", {
                    "job_id": job_id,
                    "provider": "kaggle-image-warm",
                    "model": CONFIG["image_model"],
                    "image_index": index,
                    "image_total": total,
                    "image_mime": "image/png",
                    "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "generation_prompt": prompt,
                    "elapsed_seconds": round(time.perf_counter() - item_started, 3),
                }, timeout=90)
        except Exception as exc:
            try:
                request("POST", "/internal/image/result", {
                    "job_id": job_id,
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                }, timeout=30)
            except Exception:
                pass
        finally:
            set_state("ready", "")
            signal("ready", "")
finally:
    set_state("shutting_down", "")
    signal("shutting_down", "")
    stop_heartbeat.set()
    heartbeat_thread.join(timeout=2)
    signal("offline", "")
'''
        return textwrap.dedent(template).replace("__CONFIG_JSON__", repr(config)).strip() + "\n"
