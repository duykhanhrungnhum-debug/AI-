"""Media generation V2: natural language -> concise recaption -> one image model.

Production deliberately has no visual critic, detector, species table, style router,
best-of-N voting, or automatic aesthetic retry. Quality is certified offline by
regression before a model/version is promoted; production stays fast and simple.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import textwrap
import time

from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker


IMAGE_MODEL = "black-forest-labs/FLUX.2-klein-4B"
RECAPTION_MODEL = "Qwen/Qwen3-1.7B"
IMAGE_STEPS = 4
IMAGE_GUIDANCE = 1.0
IMAGE_BATCH_DELIMITER = "<AIKA_IMAGE_BREAK>"


@dataclass(frozen=True)
class ImageRequestV2:
    item_id: str
    command: str
    seed: int
    width: int = 1024
    height: int = 1024

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id is required")
        if not self.command.strip():
            raise ValueError("command is required")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("image dimensions must be positive and divisible by 16")


@dataclass(frozen=True)
class ImageResultV2:
    item_id: str
    data: bytes
    prompt: str
    command: str
    model: str
    seed: int
    width: int
    height: int
    elapsed_seconds: float
    evidence: tuple[str, ...]


@dataclass
class KaggleImageV2Provider:
    worker: KaggleGpuWorker
    kernel_slug: str = "ai-agent-image-v2"
    poll_interval: float = 3.0
    max_poll_attempts: int = 600
    image_model: str = IMAGE_MODEL
    recaption_model: str = RECAPTION_MODEL

    def generate_many(
        self,
        requests: tuple[ImageRequestV2, ...] | list[ImageRequestV2],
    ) -> tuple[ImageResultV2, ...]:
        """Generate an explicit batch and pay model load only once."""
        assert_core_invariants()
        requests = tuple(requests)
        if not requests:
            raise ValueError("at least one image request is required")
        ids = [request.item_id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("image request ids must be unique")

        config = {
            "mode": "explicit",
            "image_model": self.image_model,
            "recaption_model": self.recaption_model,
            "steps": IMAGE_STEPS,
            "guidance": IMAGE_GUIDANCE,
            "items": [
                {
                    "id": request.item_id,
                    "command": request.command,
                    "seed": request.seed,
                    "width": request.width,
                    "height": request.height,
                }
                for request in requests
            ],
        }
        submission, report, gpu_name = self._run_source(self._build_worker_source_from_config(config))
        results: list[ImageResultV2] = []
        for request in requests:
            item = report["items"].get(request.item_id)
            if not isinstance(item, dict):
                raise ValueError(f"image V2 report missing {request.item_id}")
            results.append(self._read_result(
                submission_ref=submission.ref,
                gpu_name=gpu_name,
                item_id=request.item_id,
                command=request.command,
                seed=request.seed,
                width=request.width,
                height=request.height,
                item=item,
            ))
        return tuple(results)

    def generate_command(
        self,
        command: str,
        *,
        seed: int,
        width: int = 1024,
        height: int = 1024,
        max_images: int = 6,
    ) -> tuple[ImageResultV2, ...]:
        """Generate one or more separate images requested in one natural command.

        Qwen performs one language pass inside the same Kaggle session. It emits
        one plain-text description per explicitly requested separate image using a
        fixed delimiter; a request containing multiple subjects in one picture is
        deliberately kept as one image. FLUX is then loaded once for the batch.
        """
        assert_core_invariants()
        command = command.strip()
        if not command:
            raise ValueError("command is required")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("image dimensions must be positive and divisible by 16")
        if max_images <= 0 or max_images > 8:
            raise ValueError("max_images must be between 1 and 8")

        config = {
            "mode": "natural_batch",
            "image_model": self.image_model,
            "recaption_model": self.recaption_model,
            "steps": IMAGE_STEPS,
            "guidance": IMAGE_GUIDANCE,
            "command": command,
            "seed": seed,
            "width": width,
            "height": height,
            "max_images": max_images,
            "delimiter": IMAGE_BATCH_DELIMITER,
        }
        submission, report, gpu_name = self._run_source(self._build_worker_source_from_config(config))
        items = report.get("items")
        if not isinstance(items, dict) or not items:
            raise ValueError("image V2 report contains no generated items")
        if len(items) > max_images:
            raise ValueError("image V2 report exceeds max_images")

        results: list[ImageResultV2] = []
        for item_id, item in items.items():
            if not isinstance(item, dict):
                raise ValueError(f"image V2 report item {item_id} is invalid")
            item_seed = int(item.get("seed", seed))
            item_width = int(item.get("width", width))
            item_height = int(item.get("height", height))
            results.append(self._read_result(
                submission_ref=submission.ref,
                gpu_name=gpu_name,
                item_id=str(item_id),
                command=command,
                seed=item_seed,
                width=item_width,
                height=item_height,
                item=item,
            ))
        return tuple(results)

    def _run_source(self, source: str):
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
        for _ in range(self.max_poll_attempts):
            status = self.worker.status(self.kernel_slug)
            if status.terminal:
                if not status.successful:
                    detail = status.failure_message or status.status
                    try:
                        logs = self.worker.logs(self.kernel_slug)
                        if logs.strip():
                            detail = logs[-12000:]
                    except Exception:
                        pass
                    raise RuntimeError(f"image V2 worker failed: {detail}")
                break
            if self.poll_interval:
                time.sleep(self.poll_interval)
        else:
            raise TimeoutError("timed out waiting for image V2 worker")

        report = json.loads(
            self.worker.download_output_file(self.kernel_slug, "image_v2_report.json").decode("utf-8")
        )
        if not isinstance(report, dict) or not isinstance(report.get("items"), dict):
            raise ValueError("image V2 report is invalid")
        gpu_name = str(report.get("gpu_name") or "").strip()
        if not gpu_name:
            raise ValueError("image V2 report has no GPU evidence")
        return submission, report, gpu_name

    def _read_result(
        self,
        *,
        submission_ref: str,
        gpu_name: str,
        item_id: str,
        command: str,
        seed: int,
        width: int,
        height: int,
        item: dict,
    ) -> ImageResultV2:
        filename = str(item.get("filename") or "")
        prompt = str(item.get("prompt") or "").strip()
        if not filename or not prompt:
            raise ValueError(f"image V2 result incomplete for {item_id}")
        data = self.worker.download_output_file(self.kernel_slug, filename)
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"image V2 output is not PNG for {item_id}")
        digest = sha256(data).hexdigest()
        if digest != item.get("sha256"):
            raise ValueError(f"image V2 hash mismatch for {item_id}")
        return ImageResultV2(
            item_id=item_id,
            data=data,
            prompt=prompt,
            command=command,
            model=self.image_model,
            seed=seed,
            width=width,
            height=height,
            elapsed_seconds=float(item.get("elapsed_seconds") or 0.0),
            evidence=(
                f"kaggle_kernel:{submission_ref}",
                f"gpu:{gpu_name}",
                f"recaption_model:{self.recaption_model}",
                f"image_model:{self.image_model}",
                f"steps:{IMAGE_STEPS}",
                f"guidance:{IMAGE_GUIDANCE}",
                f"sha256:{digest}",
            ),
        )

    def _build_worker_source(self, requests: tuple[ImageRequestV2, ...]) -> str:
        """Backward-compatible explicit-batch source builder used by tests/tools."""
        config = {
            "mode": "explicit",
            "image_model": self.image_model,
            "recaption_model": self.recaption_model,
            "steps": IMAGE_STEPS,
            "guidance": IMAGE_GUIDANCE,
            "items": [
                {
                    "id": request.item_id,
                    "command": request.command,
                    "seed": request.seed,
                    "width": request.width,
                    "height": request.height,
                }
                for request in requests
            ],
        }
        return self._build_worker_source_from_config(config)

    def _build_command_worker_source(
        self,
        command: str,
        *,
        seed: int,
        width: int = 1024,
        height: int = 1024,
        max_images: int = 6,
    ) -> str:
        config = {
            "mode": "natural_batch",
            "image_model": self.image_model,
            "recaption_model": self.recaption_model,
            "steps": IMAGE_STEPS,
            "guidance": IMAGE_GUIDANCE,
            "command": command,
            "seed": seed,
            "width": width,
            "height": height,
            "max_images": max_images,
            "delimiter": IMAGE_BATCH_DELIMITER,
        }
        return self._build_worker_source_from_config(config)

    def _build_worker_source_from_config(self, config: dict) -> str:
        config_json = json.dumps(config, ensure_ascii=False)
        template = r'''
from __future__ import annotations

from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CONFIG = json.loads(__CONFIG_JSON__)
STARTED = time.perf_counter()
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def progress(step, **extra):
    payload = {"step": step, "elapsed_seconds": round(time.perf_counter() - STARTED, 3), **extra}
    print("AI_AGENT_MEDIA_V2 " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def render_recaption(tokenizer, recaptioner, instruction):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(recaptioner.device)
    with torch.inference_mode():
        generated = recaptioner.generate(
            **inputs,
            max_new_tokens=640,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.03,
        )
    text = tokenizer.batch_decode(
        generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip().strip('"')
    del generated, inputs
    return text


try:
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
except (ImportError, AttributeError):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "diffusers", "transformers>=4.57,<5", "accelerate<2", "safetensors", "sentencepiece", "Pillow<13",
    ])
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required")
gpu_name = torch.cuda.get_device_name(0)
progress("ready", gpu=gpu_name)

progress("recaption_loading", model=CONFIG["recaption_model"])
tokenizer = AutoTokenizer.from_pretrained(CONFIG["recaption_model"])
recaptioner = AutoModelForCausalLM.from_pretrained(
    CONFIG["recaption_model"],
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
recaptioner.eval()

base_contract = (
    "Preserve exactly the requested subject or species, number of subjects, visual style, setting, framing, "
    "important attributes, and explicit exclusions. Preserve culturally specific names and untranslated proper "
    "terms verbatim instead of substituting an item from another culture. For named garments, foods, places, or "
    "art forms, keep the original name and optionally add a short English gloss. For a culturally specific named "
    "garment, add its canonical silhouette and construction details when known confidently. In particular, Vietnamese "
    "'áo dài' must remain 'Vietnamese áo dài' and must be described as a fitted high-collared long-sleeved tunic with "
    "long front and back panels, high side slits, worn over separate loose full-length trousers; it is not a one-piece "
    "dress, hanbok, qipao, or cheongsam. Do not generalize a named subject. Do not invent body parts, objects, text, "
    "logos, or requirements that the user did not request."
)

items = []
prompts = {}
if CONFIG.get("mode") == "natural_batch":
    delimiter = CONFIG["delimiter"]
    instruction = (
        "Convert the USER REQUEST into faithful English image-generation descriptions. "
        + base_contract + " "
        "Only create multiple descriptions when the user explicitly asks for multiple separate images, pictures, files, "
        "or variants. Multiple subjects requested together in one image must stay in ONE description. When multiple "
        "separate images are requested, output exactly one complete description per image in the user's requested order, "
        "separated only by the exact delimiter " + delimiter + ". Do not number or label the descriptions. Do not output "
        "JSON, explanation, scoring, or commentary. Never exceed " + str(CONFIG["max_images"]) + " descriptions.\n"
        "USER REQUEST: " + CONFIG["command"]
    )
    planned = render_recaption(tokenizer, recaptioner, instruction)
    parts = [part.strip().strip('"') for part in planned.split(delimiter) if part.strip()]
    if not parts:
        raise RuntimeError("recaption returned no image descriptions")
    if len(parts) > int(CONFIG["max_images"]):
        raise RuntimeError("recaption exceeded max_images")
    for index, prompt in enumerate(parts):
        item_id = f"image-{index + 1:02d}"
        item = {
            "id": item_id,
            "command": CONFIG["command"],
            "seed": (int(CONFIG["seed"]) + index) % (2 ** 32),
            "width": int(CONFIG["width"]),
            "height": int(CONFIG["height"]),
        }
        items.append(item)
        prompts[item_id] = prompt
        progress("recaptioned", item_id=item_id, item_index=index + 1, item_total=len(parts))
else:
    items = list(CONFIG["items"])
    for index, item in enumerate(items):
        instruction = (
            "Rewrite the USER REQUEST as one concise, vivid English image-generation description. "
            + base_contract + " "
            "Output only the final English description, with no labels, JSON, explanation, scoring, or commentary.\n"
            "USER REQUEST: " + item["command"]
        )
        prompt = render_recaption(tokenizer, recaptioner, instruction)
        if len(prompt) < 12:
            raise RuntimeError("recaption returned an empty/invalid description for " + item["id"])
        prompts[item["id"]] = prompt
        progress("recaptioned", item_id=item["id"], item_index=index + 1, item_total=len(items))

for item_id, prompt in prompts.items():
    if len(prompt) < 12:
        raise RuntimeError("recaption returned an empty/invalid description for " + item_id)

del recaptioner, tokenizer
gc.collect()
torch.cuda.empty_cache()

progress("image_model_loading", model=CONFIG["image_model"])
dtype = torch.float16
pipe = Flux2KleinPipeline.from_pretrained(CONFIG["image_model"], torch_dtype=dtype)
pipe.enable_model_cpu_offload()
progress("image_model_ready", model=CONFIG["image_model"], dtype=str(dtype), execution="model_cpu_offload")

reports = {}
for index, item in enumerate(items):
    item_started = time.perf_counter()
    progress("generating", item_id=item["id"], item_index=index + 1, item_total=len(items))
    generator = torch.Generator(device="cpu").manual_seed(int(item["seed"]))
    image = pipe(
        prompt=prompts[item["id"]],
        height=int(item["height"]),
        width=int(item["width"]),
        guidance_scale=float(CONFIG["guidance"]),
        num_inference_steps=int(CONFIG["steps"]),
        generator=generator,
    ).images[0]
    filename = item["id"] + ".png"
    path = Path("/kaggle/working") / filename
    image.save(path, format="PNG")
    data = path.read_bytes()
    reports[item["id"]] = {
        "filename": filename,
        "prompt": prompts[item["id"]],
        "sha256": sha256(data).hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - item_started, 3),
        "seed": int(item["seed"]),
        "width": int(item["width"]),
        "height": int(item["height"]),
    }
    progress("generated", item_id=item["id"], elapsed_seconds=reports[item["id"]]["elapsed_seconds"])

report = {
    "architecture": "natural_request->recaption_batch->flux2_klein",
    "runtime_visual_qa": False,
    "gpu_name": gpu_name,
    "image_model": CONFIG["image_model"],
    "recaption_model": CONFIG["recaption_model"],
    "steps": CONFIG["steps"],
    "guidance": CONFIG["guidance"],
    "worker_total_seconds": round(time.perf_counter() - STARTED, 3),
    "items": reports,
}
Path("/kaggle/working/image_v2_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
progress("complete", worker_total_seconds=report["worker_total_seconds"])
'''
        return textwrap.dedent(template).replace("__CONFIG_JSON__", repr(config_json)).strip() + "\n"
