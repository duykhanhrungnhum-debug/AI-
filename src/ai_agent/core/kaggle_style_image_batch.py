"""Semantic image generation with style-specialized engines in one Kaggle GPU worker.

The router is semantic and species-agnostic.  The worker compiles every command
once, then loads the minimum set of image engines sequentially in the same GPU
session.  Photographic/default work uses RealVisXL while 3D mascot work uses
Playground v2.5.  This avoids the old two-worker queue failure while no longer
forcing one generator to cover incompatible visual domains.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
import textwrap
import time

from .image_model import ImageArtifact
from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker


_ITEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


@dataclass(frozen=True)
class StyleRoutedImageItem:
    item_id: str
    command: str
    seed: int
    fallback_profile: str = "general_premium"

    def __post_init__(self) -> None:
        if not _ITEM_ID.fullmatch(self.item_id):
            raise ValueError("item_id must be a simple 1-64 character slug")
        if not self.command.strip():
            raise ValueError("command is required")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class StyleRoutedCandidate:
    item_id: str
    candidate_index: int
    artifact: ImageArtifact
    report: dict


@dataclass(frozen=True)
class StyleRoutedImageBatchOutput:
    candidates: dict[str, tuple[StyleRoutedCandidate, ...]]
    plans: dict[str, dict]
    batch_report: dict


@dataclass
class KaggleStyleRoutedImageBatchProvider:
    """Plan semantically, then run style-specialized engines in one GPU job."""

    worker: KaggleGpuWorker
    profile_engines: dict[str, dict]
    profile_positive_constraints: dict[str, str]
    profile_negative_constraints: dict[str, str]
    profile_dimensions: dict[str, tuple[int, int]]
    kernel_slug: str = "ai-agent-style-routed-image-batch"
    poll_interval: float = 3.0
    max_poll_attempts: int = 600
    planner_fast_model: str = "Qwen/Qwen3-0.6B"
    planner_final_model: str = "Qwen/Qwen3-1.7B"
    provider: str = "kaggle-gpu-style-routed-image-batch"

    def __post_init__(self) -> None:
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        keys = set(self.profile_engines)
        if not keys:
            raise ValueError("profile_engines must not be empty")
        for mapping in (
            self.profile_positive_constraints,
            self.profile_negative_constraints,
            self.profile_dimensions,
        ):
            if set(mapping) != keys:
                raise ValueError("style-routed profile maps must have matching keys")
        for profile, engine in self.profile_engines.items():
            required = {"engine_id", "model", "steps", "guidance", "scheduler", "candidate_count"}
            if not isinstance(engine, dict) or not required.issubset(engine):
                raise ValueError(f"invalid engine configuration for {profile}")
            if engine["scheduler"] not in {"default", "dpm_karras", "edm_dpm"}:
                raise ValueError(f"invalid scheduler for {profile}")
            if int(engine["steps"]) <= 0 or int(engine["candidate_count"]) <= 0:
                raise ValueError(f"invalid step/candidate configuration for {profile}")

    def generate_batch(
        self,
        items: tuple[StyleRoutedImageItem, ...] | list[StyleRoutedImageItem],
    ) -> StyleRoutedImageBatchOutput:
        assert_core_invariants()
        items = tuple(items)
        if not items:
            raise ValueError("at least one style-routed image item is required")
        ids = [item.item_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("style-routed image item ids must be unique")

        source = self._build_worker_source(items)
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=self.kernel_slug.replace("-", " ").title(),
            source=source,
            enable_internet=True,
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
                            detail = logs[-8000:]
                    except Exception:
                        pass
                    raise RuntimeError(f"Kaggle style-routed image worker failed: {detail}")
                break
            if self.poll_interval:
                time.sleep(self.poll_interval)
        else:
            raise TimeoutError("Timed out waiting for style-routed image worker")

        report = self._parse_report(
            self.worker.download_output_file(self.kernel_slug, "style_batch_report.json")
        )
        gpu_name = str(report.get("gpu_name") or "").strip()
        if not gpu_name:
            raise ValueError("style-routed report has no GPU evidence")
        report_items = report.get("items")
        if not isinstance(report_items, dict):
            raise ValueError("style-routed report has no item map")

        candidates: dict[str, tuple[StyleRoutedCandidate, ...]] = {}
        plans: dict[str, dict] = {}
        for item in items:
            entry = report_items.get(item.item_id)
            if not isinstance(entry, dict):
                raise ValueError(f"style-routed report missing item: {item.item_id}")
            profile = str(entry.get("semantic_profile") or "")
            if profile not in self.profile_engines:
                raise ValueError(f"style-routed profile invalid: {item.item_id}")
            plans[item.item_id] = entry
            raw_candidates = entry.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise ValueError(f"style-routed candidates missing: {item.item_id}")
            materialized: list[StyleRoutedCandidate] = []
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    raise ValueError(f"style-routed candidate invalid: {item.item_id}")
                index = int(raw_candidate.get("candidate_index", -1))
                filename = str(raw_candidate.get("filename") or "")
                if index < 0 or not filename:
                    raise ValueError(f"style-routed candidate metadata invalid: {item.item_id}")
                image = self.worker.download_output_file(self.kernel_slug, filename)
                digest = sha256(image).hexdigest()
                if digest != raw_candidate.get("image_sha256"):
                    raise ValueError(f"style-routed candidate hash mismatch: {item.item_id}")
                model = str(raw_candidate.get("model") or "")
                artifact = ImageArtifact(
                    data=image,
                    mime_type="image/png",
                    provider=self.provider,
                    model=model,
                    evidence=(
                        f"kaggle_kernel:{submission.ref}",
                        f"gpu:{gpu_name}",
                        f"item_id:{item.item_id}",
                        f"candidate_index:{index}",
                        f"image_sha256:{digest}",
                        f"semantic_profile:{profile}",
                        f"subject_class:{entry.get('subject_class')}",
                        f"style_class:{entry.get('style_class')}",
                        f"subject_name:{entry.get('subject_name')}",
                        f"engine_id:{raw_candidate.get('engine_id')}",
                        f"model:{model}",
                    ),
                )
                materialized.append(StyleRoutedCandidate(
                    item_id=item.item_id,
                    candidate_index=index,
                    artifact=artifact,
                    report=raw_candidate,
                ))
            candidates[item.item_id] = tuple(materialized)

        return StyleRoutedImageBatchOutput(
            candidates=candidates,
            plans=plans,
            batch_report=report,
        )

    @staticmethod
    def _parse_report(data: bytes) -> dict:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("style-routed report is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("style-routed report must be an object")
        return payload

    def _build_worker_source(self, items: tuple[StyleRoutedImageItem, ...]) -> str:
        config = {
            "profile_engines": self.profile_engines,
            "profile_positive_constraints": self.profile_positive_constraints,
            "profile_negative_constraints": self.profile_negative_constraints,
            "profile_dimensions": {
                key: list(value) for key, value in self.profile_dimensions.items()
            },
            "planner_fast_model": self.planner_fast_model,
            "planner_final_model": self.planner_final_model,
            "items": [
                {
                    "item_id": item.item_id,
                    "command": item.command,
                    "seed": item.seed,
                    "fallback_profile": item.fallback_profile,
                }
                for item in items
            ],
        }
        config_json = json.dumps(config, ensure_ascii=False)
        template = r'''
from __future__ import annotations

from hashlib import sha256
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIG = json.loads(__CONFIG_JSON__)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    import torch
    from PIL import Image
    from diffusers import (
        AutoPipelineForText2Image,
        DPMSolverMultistepScheduler,
        EDMDPMSolverMultistepScheduler,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "diffusers<1", "transformers>=4.57,<5", "accelerate<2", "safetensors", "Pillow<13",
    ])
    import torch
    from PIL import Image
    from diffusers import (
        AutoPipelineForText2Image,
        DPMSolverMultistepScheduler,
        EDMDPMSolverMultistepScheduler,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available")

started = time.perf_counter()
gpu_name = torch.cuda.get_device_name(0)


def normalize_subject(value):
    token = (value or "").strip().casefold().replace("_", " ").replace("-", " ")
    if token in {"animal", "animals", "creature", "creatures", "pet", "wildlife"}:
        return "animal"
    if token in {"human", "person", "people", "man", "woman"}:
        return "human"
    if token in {"general", "object", "scene", "other"}:
        return "general"
    return ""


def normalize_style(value):
    token = (value or "").strip().casefold().replace("_", " ").replace("-", " ")
    if token in {"photo", "photograph", "photorealistic", "realistic photo", "real photo"}:
        return "photo"
    if token in {"3d", "3 d", "3d render", "cgi", "cg"}:
        return "3d"
    if token in {"mascot", "3d mascot"}:
        return "mascot"
    if token in {"illustration", "drawing", "painting", "painted"}:
        return "illustration"
    if token in {"general", "other"}:
        return "general"
    return ""


def looks_vietnamese(text):
    chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    return any(ch in chars for ch in (text or "").casefold())


def placeholder(text):
    value = " ".join((text or "").strip().casefold().split())
    if not value:
        return True
    bad = {
        "english image description", "english image prompt", "english prompt",
        "image description", "subject name", "exact english subject name",
    }
    return value in bad or "<" in value or ">" in value


def parse_compiled(raw):
    raw = (raw or "").strip().strip(chr(96)).strip()
    result = {"prompt": "", "subject_class": "", "style_class": "", "subject_name": ""}
    if not raw:
        return result
    cleaned = raw
    if cleaned.casefold().startswith("json"):
        cleaned = cleaned[4:].lstrip("\n :")
    first, last = cleaned.find("{"), cleaned.rfind("}")
    candidates = [cleaned]
    if 0 <= first < last:
        candidates.insert(0, cleaned[first:last + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        result["prompt"] = str(payload.get("PROMPT") or payload.get("prompt") or "").strip()
        result["subject_class"] = normalize_subject(str(payload.get("SUBJECT_CLASS") or payload.get("subject_class") or ""))
        result["style_class"] = normalize_style(str(payload.get("STYLE_CLASS") or payload.get("style_class") or ""))
        result["subject_name"] = str(payload.get("SUBJECT_NAME") or payload.get("subject_name") or payload.get("species") or "").strip()
        if result["prompt"]:
            return result
    aliases = {
        "PROMPT": "prompt", "PROMT": "prompt", "PROMP": "prompt",
        "SUBJECTCLASS": "subject_class", "SUBJECT": "subject_class",
        "STYLECLASS": "style_class", "STYLE": "style_class",
        "SUBJECTNAME": "subject_name", "SPECIES": "subject_name", "ENTITY": "subject_name",
    }
    for line in cleaned.replace("：", ":").splitlines():
        line = line.strip().lstrip("-*#> ").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = "".join(ch for ch in key.strip().upper() if ch.isalpha())
        canonical = aliases.get(key)
        if canonical and value.strip() and not result[canonical]:
            result[canonical] = value.strip().strip(chr(96)).strip()
    result["subject_class"] = normalize_subject(result["subject_class"])
    result["style_class"] = normalize_style(result["style_class"])
    return result


def choose_profile(subject, style, fallback):
    if style in {"3d", "mascot"}:
        return "mascot_premium"
    if subject == "animal":
        return "animal_photo_premium"
    if subject == "human":
        return "human_photo_premium"
    return fallback if fallback in CONFIG["profile_engines"] else "general_premium"


def compile_items(model_name, pending):
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    ready = time.perf_counter()
    outputs = {}
    for item in pending:
        instruction = (
            "MEDIA_COMMAND_COMPILE\n"
            "Understand USER_COMMAND semantically. Do not depend on a fixed list of animal species. "
            "Preserve the exact named species/entity and requested visual style. Translate the visual request into one concise English still-image prompt. "
            "SUBJECT_CLASS must be animal, human, or general. STYLE_CLASS must be photo, 3d, mascot, illustration, or general. "
            "SUBJECT_NAME must be the exact English common name of the requested main subject/species. "
            "Do not invent anatomy counts or appendages. Do not copy schema placeholders. "
            "Return exactly four labeled lines and no commentary:\n"
            "SUBJECT_CLASS: animal|human|general\n"
            "STYLE_CLASS: photo|3d|mascot|illustration|general\n"
            "SUBJECT_NAME: exact English subject name\n"
            "PROMPT: complete English image description\n"
            "USER_COMMAND: " + item["command"]
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=190,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.04,
            )
        text = tokenizer.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        parsed = parse_compiled(text)
        valid = (
            len(parsed["prompt"]) >= 20
            and not looks_vietnamese(parsed["prompt"])
            and not placeholder(parsed["prompt"])
            and not placeholder(parsed["subject_name"])
            and parsed["subject_class"] in {"animal", "human", "general"}
            and parsed["style_class"] in {"photo", "3d", "mascot", "illustration", "general"}
        )
        outputs[item["item_id"]] = {"valid": valid, "raw": text, "parsed": parsed}
        del generated, inputs
    timing = {
        "model_load_seconds": round(ready - load_started, 3),
        "total_seconds": round(time.perf_counter() - load_started, 3),
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, timing


planner, fast_timing = compile_items(CONFIG["planner_fast_model"], CONFIG["items"])
failed = [item for item in CONFIG["items"] if not planner[item["item_id"]]["valid"]]
final_timing = None
if failed:
    fallback, final_timing = compile_items(CONFIG["planner_final_model"], failed)
    planner.update(fallback)

compiled = []
for item in CONFIG["items"]:
    result = planner[item["item_id"]]
    if not result["valid"]:
        raise RuntimeError("semantic planner failed for item " + item["item_id"])
    parsed = result["parsed"]
    profile = choose_profile(parsed["subject_class"], parsed["style_class"], item["fallback_profile"])
    engine = CONFIG["profile_engines"][profile]
    positive = CONFIG["profile_positive_constraints"][profile]
    negative = CONFIG["profile_negative_constraints"][profile]
    width, height = CONFIG["profile_dimensions"][profile]
    prompt = parsed["prompt"].rstrip(" .") + ". Quality requirements: " + positive.rstrip(" .") + "."
    if profile == "mascot_premium":
        prompt += (
            " Species-anatomy guard: keep only anatomy naturally belonging to the named species; no duplicated or extra horn-like, antler-like, tusk-like, ear-like, limb-like, tail-like, wing-like or fin-like appendages. "
            "Cute stylization may change proportions but must not create extra body parts. Keep the complete character centered with generous clean border margin and no text, logo, signature, badge or watermark."
        )
        negative = negative.rstrip(" ,") + ", extra horns, extra antlers, extra tusks, duplicated appendages, extra ears, extra legs, extra tails, wings on non-winged subject, fins on non-finned subject, pseudo-logo, pseudo-signature, corner badge, watermark"
    compiled.append({
        **item,
        "prompt": prompt,
        "negative_prompt": negative,
        "subject_class": parsed["subject_class"],
        "style_class": parsed["style_class"],
        "subject_name": parsed["subject_name"],
        "semantic_profile": profile,
        "planner_raw": result["raw"],
        "planner_model": CONFIG["planner_final_model"] if item in failed else CONFIG["planner_fast_model"],
        "width": int(width),
        "height": int(height),
        "engine": engine,
    })

# Group by engine_id so each image model is loaded at most once per GPU session.
engine_groups = {}
for item in compiled:
    engine_groups.setdefault(item["engine"]["engine_id"], []).append(item)

item_reports = {}
engine_timings = {}
for engine_id, engine_items in engine_groups.items():
    engine = engine_items[0]["engine"]
    model_load_started = time.perf_counter()
    kwargs = {"torch_dtype": torch.float16}
    if engine.get("variant"):
        kwargs["variant"] = engine["variant"]
    pipe = AutoPipelineForText2Image.from_pretrained(engine["model"], **kwargs)
    if engine["scheduler"] == "dpm_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif engine["scheduler"] == "edm_dpm":
        pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()
    model_ready = time.perf_counter()

    for item in engine_items:
        candidates = []
        for candidate_index in range(int(engine["candidate_count"])):
            gen_started = time.perf_counter()
            seed = int(item["seed"]) + candidate_index
            generator = torch.Generator(device="cpu").manual_seed(seed)
            final_width, final_height = item["width"], item["height"]
            safe_crop = item["semantic_profile"] == "mascot_premium"
            if safe_crop:
                render_width = ((int(final_width * 1.125) + 7) // 8) * 8
                render_height = ((int(final_height * 1.125) + 7) // 8) * 8
            else:
                render_width, render_height = final_width, final_height
            image = pipe(
                prompt=item["prompt"],
                negative_prompt=item["negative_prompt"] or None,
                width=render_width,
                height=render_height,
                num_inference_steps=int(engine["steps"]),
                guidance_scale=float(engine["guidance"]),
                generator=generator,
            ).images[0]
            if safe_crop:
                left = max(0, (image.width - final_width) // 2)
                top = max(0, (image.height - final_height) // 2)
                image = image.crop((left, top, left + final_width, top + final_height))
            filename = item["item_id"] + "__candidate_" + str(candidate_index) + ".png"
            path = Path("/kaggle/working") / filename
            image.save(path, format="PNG")
            data = path.read_bytes()
            candidates.append({
                "candidate_index": candidate_index,
                "filename": filename,
                "seed": seed,
                "image_sha256": sha256(data).hexdigest(),
                "engine_id": engine_id,
                "model": engine["model"],
                "scheduler": engine["scheduler"],
                "render_width": render_width,
                "render_height": render_height,
                "safe_border_crop": safe_crop,
                "generation_seconds": round(time.perf_counter() - gen_started, 3),
            })
        item_reports[item["item_id"]] = {
            "prompt": item["prompt"],
            "negative_prompt": item["negative_prompt"],
            "subject_class": item["subject_class"],
            "style_class": item["style_class"],
            "subject_name": item["subject_name"],
            "semantic_profile": item["semantic_profile"],
            "planner_raw": item["planner_raw"],
            "planner_model": item["planner_model"],
            "width": item["width"],
            "height": item["height"],
            "engine_id": engine_id,
            "candidates": candidates,
        }

    engine_timings[engine_id] = {
        "model": engine["model"],
        "model_load_seconds": round(model_ready - model_load_started, 3),
        "total_seconds": round(time.perf_counter() - model_load_started, 3),
    }
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

report = {
    "gpu_name": gpu_name,
    "planner_fast_timing": fast_timing,
    "planner_final_timing": final_timing,
    "engine_timings": engine_timings,
    "worker_total_seconds": round(time.perf_counter() - started, 3),
    "items": item_reports,
}
Path("/kaggle/working/style_batch_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("AI_AGENT_STYLE_ROUTED_IMAGE_BATCH_OK")
print(json.dumps({
    "gpu_name": gpu_name,
    "engine_ids": list(engine_groups),
    "item_count": len(item_reports),
    "worker_total_seconds": report["worker_total_seconds"],
}, ensure_ascii=False, indent=2))
'''
        return textwrap.dedent(template).replace("__CONFIG_JSON__", repr(config_json)).strip() + "\n"
