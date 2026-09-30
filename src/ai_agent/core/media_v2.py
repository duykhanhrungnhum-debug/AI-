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
        assert_core_invariants()
        requests = tuple(requests)
        if not requests:
            raise ValueError("at least one image request is required")
        ids = [request.item_id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("image request ids must be unique")

        kernel_title = (
            "AI Agent Image V2"
            if self.kernel_slug == "ai-agent-image-v2"
            else f"AI Agent Image V2 {sha256(self.kernel_slug.encode('utf-8')).hexdigest()[:8]}"
        )
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=kernel_title,
            source=self._build_worker_source(requests),
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

        results: list[ImageResultV2] = []
        for request in requests:
            item = report["items"].get(request.item_id)
            if not isinstance(item, dict):
                raise ValueError(f"image V2 report missing {request.item_id}")
            filename = str(item.get("filename") or "")
            prompt = str(item.get("prompt") or "").strip()
            if not filename or not prompt:
                raise ValueError(f"image V2 result incomplete for {request.item_id}")
            data = self.worker.download_output_file(self.kernel_slug, filename)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"image V2 output is not PNG for {request.item_id}")
            digest = sha256(data).hexdigest()
            if digest != item.get("sha256"):
                raise ValueError(f"image V2 hash mismatch for {request.item_id}")
            results.append(ImageResultV2(
                item_id=request.item_id,
                data=data,
                prompt=prompt,
                command=request.command,
                model=self.image_model,
                seed=request.seed,
                width=request.width,
                height=request.height,
                elapsed_seconds=float(item.get("elapsed_seconds") or 0.0),
                evidence=(
                    f"kaggle_kernel:{submission.ref}",
                    f"gpu:{gpu_name}",
                    f"recaption_model:{self.recaption_model}",
                    f"image_model:{self.image_model}",
                    f"steps:{IMAGE_STEPS}",
                    f"guidance:{IMAGE_GUIDANCE}",
                    f"sha256:{digest}",
                ),
            ))
        return tuple(results)

    def _build_worker_source(self, requests: tuple[ImageRequestV2, ...]) -> str:
        config = {
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

# One language step only: turn the user's natural request into a faithful English
# generation description. No JSON, classification, anatomy schema, or keyword list.
progress("recaption_loading", model=CONFIG["recaption_model"])
tokenizer = AutoTokenizer.from_pretrained(CONFIG["recaption_model"])
recaptioner = AutoModelForCausalLM.from_pretrained(
    CONFIG["recaption_model"],
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
recaptioner.eval()
prompts = {}
for index, item in enumerate(CONFIG["items"]):
    instruction = (
        "Rewrite the USER REQUEST as one concise, vivid English image-generation description. "
        "Preserve exactly the requested subject or species, number of subjects, visual style, setting, framing, "
        "important attributes, and explicit exclusions. Preserve culturally specific names and untranslated proper "
        "terms verbatim instead of substituting an item from another culture. For named garments, foods, places, or "
        "art forms, keep the original name and optionally add a short English gloss. In particular, Vietnamese 'áo dài' "
        "must remain 'Vietnamese áo dài' and must never be rewritten as hanbok, qipao, cheongsam, or another garment. "
        "Do not generalize a named subject. Do not invent body parts, objects, text, logos, or requirements that the user "
        "did not request. Output only the final English description, with no labels, JSON, explanation, scoring, or "
        "commentary.\nUSER REQUEST: " + item["command"]
    )
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
            max_new_tokens=320,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.03,
        )
    prompt = tokenizer.batch_decode(
        generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip().strip('"')
    if len(prompt) < 12:
        raise RuntimeError("recaption returned an empty/invalid description for " + item["id"])
    prompts[item["id"]] = prompt
    progress("recaptioned", item_id=item["id"], item_index=index + 1, item_total=len(CONFIG["items"]))
    del generated, inputs

del recaptioner, tokenizer
gc.collect()
torch.cuda.empty_cache()

# One unified image model for photo, 3D, illustration, and later image editing.
progress("image_model_loading", model=CONFIG["image_model"])
dtype = torch.float16
pipe = Flux2KleinPipeline.from_pretrained(CONFIG["image_model"], torch_dtype=dtype)
# A Kaggle T4 has 14.56 GiB VRAM while the full fp16 pipeline peaks slightly
# above that. Diffusers model CPU offload keeps the architecture unchanged and
# moves only the component currently executing onto the GPU.
pipe.enable_model_cpu_offload()
progress("image_model_ready", model=CONFIG["image_model"], dtype=str(dtype), execution="model_cpu_offload")

reports = {}
for index, item in enumerate(CONFIG["items"]):
    item_started = time.perf_counter()
    progress("generating", item_id=item["id"], item_index=index + 1, item_total=len(CONFIG["items"]))
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
    }
    progress("generated", item_id=item["id"], elapsed_seconds=reports[item["id"]]["elapsed_seconds"])

report = {
    "architecture": "natural_request->single_recaption->flux2_klein",
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