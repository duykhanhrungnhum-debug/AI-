"""Build the self-contained Kaggle source for AIKA Warm Image Worker v6."""
from __future__ import annotations

import json
import textwrap


def build_v6_worker_source(config: dict) -> str:
    config_json = json.dumps(config, ensure_ascii=False)
    template = r'''
from __future__ import annotations

import base64
from io import BytesIO
import gc
import json
import os
import re
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

# Bound CPU scheduling and output length so recaption has a predictable ceiling.
# Qwen stays CPU-resident; FLUX is the only CUDA owner.
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

tokenizer = AutoTokenizer.from_pretrained(CONFIG["recaption_model"])
recaptioner = AutoModelForCausalLM.from_pretrained(
    CONFIG["recaption_model"],
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
recaptioner.eval()

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
            "User-Agent": "AIKA-Warm-Image/6.2",
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


VI_ORDINALS = {"nhất": 1, "một": 1, "hai": 2, "ba": 3, "tư": 4, "bốn": 4, "năm": 5, "sáu": 6}
EN_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
VI_NUMBERS = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "sáu": 6}
EN_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def requested_count(command):
    text = " ".join(command.casefold().split())
    digit = re.search(r"\b([1-6])\s*(?:ảnh|hình|image|images|picture|pictures|file|files)\b", text)
    if digit:
        return int(digit.group(1))
    for word, value in {**VI_NUMBERS, **EN_NUMBERS}.items():
        if re.search(rf"\b{re.escape(word)}\s*(?:ảnh|hình|image|images|picture|pictures|file|files)\b", text):
            return value
    return None


def split_explicit_items(command):
    patterns = [
        re.compile(r"(?i)\b(?:ảnh|hình)\s+(?:thứ\s+)?(nhất|một|hai|ba|tư|bốn|năm|sáu)\b"),
        re.compile(r"(?i)\b(?:ảnh|hình)\s+(?:số\s*)?([1-6])\b"),
        re.compile(r"(?i)\b(first|second|third|fourth|fifth|sixth)\s+(?:image|picture|file)\b"),
        re.compile(r"(?i)\b(?:image|picture|file)\s*#?\s*([1-6])\b"),
    ]
    found = []
    for pattern in patterns:
        for match in pattern.finditer(command):
            token = match.group(1).casefold()
            if token.isdigit():
                index = int(token)
            else:
                index = VI_ORDINALS.get(token) or EN_ORDINALS.get(token) or 0
            if index:
                found.append((match.start(), match.end(), index))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    markers = []
    last_end = -1
    for item in found:
        if item[0] < last_end:
            continue
        markers.append(item)
        last_end = item[1]
    if len(markers) < 2 or len(markers) > int(CONFIG["max_images"]):
        return None
    if [item[2] for item in markers] != list(range(1, len(markers) + 1)):
        return None
    expected = requested_count(command)
    if expected is not None and expected != len(markers):
        return None
    items = []
    for pos, (_, end, _) in enumerate(markers):
        next_start = markers[pos + 1][0] if pos + 1 < len(markers) else len(command)
        body = command[end:next_start].strip(" \t\r\n,;:.-")
        body = re.sub(r"(?i)^(?:là|is)\s+", "", body).strip()
        if not body:
            return None
        items.append(body)
    return items


BASE_CONTRACT = (
    "Preserve exactly the requested subject or species, number of subjects, visual style, setting, framing, "
    "important attributes, and explicit exclusions. Preserve culturally specific names instead of substituting an "
    "item from another culture. Vietnamese 'áo dài' must remain Vietnamese áo dài and preserve its long tunic panels "
    "over separate trousers when requested. Do not generalize a named subject. "
)


def qwen_generate(instruction):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([rendered], return_tensors="pt")
    with torch.inference_mode():
        generated = recaptioner.generate(
            **inputs,
            max_new_tokens=int(CONFIG["recaption_max_new_tokens"]),
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.03,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.batch_decode(
        generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip().strip('"')
    del generated, inputs
    gc.collect()
    return text


def render_one_recaption(command):
    instruction = (
        "Rewrite the USER REQUEST as exactly one faithful, concise English image-generation description. "
        + BASE_CONTRACT
        + "Keep it under 120 words. Output only that one description. Do not number it, label it, explain it, or output JSON.\n"
        + "USER REQUEST: " + command
    )
    text = qwen_generate(instruction).strip().strip('"')
    if len(text) < 12:
        raise RuntimeError("recaption returned an empty/invalid description")
    return text


def render_batch_recaption(command):
    delimiter = CONFIG["delimiter"]
    instruction = (
        "Convert the USER REQUEST into concise English image-generation descriptions. " + BASE_CONTRACT +
        "Only create multiple descriptions when the user explicitly asks for multiple separate images, pictures, files, "
        "or variants. Multiple subjects requested together in one image must stay in ONE description. When multiple "
        "separate images are requested, output exactly one description per image in the user's requested order, each under "
        "120 words, separated only by the exact delimiter " + delimiter + ". Do not number or label the descriptions. "
        "Do not output JSON, explanation, scoring, or commentary. Never exceed " + str(CONFIG["max_images"]) + " descriptions.\n"
        "USER REQUEST: " + command
    )
    planned = qwen_generate(instruction)
    prompts = [part.strip().strip('"') for part in planned.split(delimiter) if part.strip()]
    if not prompts:
        raise RuntimeError("recaption returned no image descriptions")
    if len(prompts) > int(CONFIG["max_images"]):
        raise RuntimeError("recaption exceeded max_images")
    expected = requested_count(command)
    if expected is not None and len(prompts) != expected:
        raise RuntimeError(f"recaption returned {len(prompts)} images; command explicitly requested {expected}")
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
            explicit_items = split_explicit_items(command)
            if explicit_items:
                prompts = []
                item_total = len(explicit_items)
                for item_index, item_command in enumerate(explicit_items, 1):
                    stage = f"busy:recaption:{item_index}/{item_total}"
                    set_state(stage, job_id)
                    signal(stage, job_id)
                    prompts.append(render_one_recaption(item_command))
                    signal(stage, job_id)
            else:
                set_state("busy:recaption", job_id)
                signal("busy:recaption", job_id)
                prompts = render_batch_recaption(command)
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
    return textwrap.dedent(template).replace("__CONFIG_JSON__", repr(config_json)).strip() + "\n"
