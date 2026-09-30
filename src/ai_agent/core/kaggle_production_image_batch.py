"""Production image generation: semantic router -> style engine -> detector gate.

Design goals:
- one trusted semantic planner (Qwen3-1.7B), no species whitelist;
- style-specialized image engines loaded sequentially in one Kaggle GPU session;
- structured live progress streamed from Kaggle logs;
- optional zero-shot object-count hard gates supplied only by trusted task/test contracts;
- original user command is preserved for downstream VLM QA.
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
class ObjectCountGate:
    label: str
    expected_count: int
    query: str | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("gate label is required")
        if self.expected_count < 0:
            raise ValueError("expected_count must be non-negative")
        if self.query is not None and not self.query.strip():
            raise ValueError("gate query must be non-empty when provided")


@dataclass(frozen=True)
class ProductionImageItem:
    item_id: str
    command: str
    seed: int
    fallback_profile: str = "general_premium"
    hard_gates: tuple[ObjectCountGate, ...] = ()

    def __post_init__(self) -> None:
        if not _ITEM_ID.fullmatch(self.item_id):
            raise ValueError("item_id must be a simple 1-64 character slug")
        if not self.command.strip():
            raise ValueError("command is required")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class ProductionImageCandidate:
    item_id: str
    candidate_index: int
    artifact: ImageArtifact
    report: dict


@dataclass(frozen=True)
class ProductionImageBatchOutput:
    candidates: dict[str, tuple[ProductionImageCandidate, ...]]
    plans: dict[str, dict]
    batch_report: dict


@dataclass
class KaggleProductionImageBatchProvider:
    worker: KaggleGpuWorker
    profile_engines: dict[str, dict]
    profile_positive_constraints: dict[str, str]
    profile_negative_constraints: dict[str, str]
    profile_dimensions: dict[str, tuple[int, int]]
    kernel_slug: str = "ai-agent-production-image-batch"
    poll_interval: float = 3.0
    max_poll_attempts: int = 700
    planner_model: str = "Qwen/Qwen3-1.7B"
    detector_model: str = "IDEA-Research/grounding-dino-tiny"
    detector_box_threshold: float = 0.25
    detector_text_threshold: float = 0.20
    progress_poll_every: int = 5
    provider: str = "kaggle-gpu-production-image-batch"

    def __post_init__(self) -> None:
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if not self.planner_model.strip() or not self.detector_model.strip():
            raise ValueError("planner_model and detector_model are required")
        if not 0 <= self.detector_box_threshold <= 1 or not 0 <= self.detector_text_threshold <= 1:
            raise ValueError("detector thresholds must be in [0, 1]")
        if self.progress_poll_every <= 0:
            raise ValueError("progress_poll_every must be positive")
        keys = set(self.profile_engines)
        if not keys:
            raise ValueError("profile_engines must not be empty")
        for mapping in (
            self.profile_positive_constraints,
            self.profile_negative_constraints,
            self.profile_dimensions,
        ):
            if set(mapping) != keys:
                raise ValueError("production image profile maps must have matching keys")

    def generate_batch(
        self,
        items: tuple[ProductionImageItem, ...] | list[ProductionImageItem],
    ) -> ProductionImageBatchOutput:
        assert_core_invariants()
        items = tuple(items)
        if not items:
            raise ValueError("at least one production image item is required")
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("production image item ids must be unique")

        source = self._build_worker_source(items)
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=self.kernel_slug.replace("-", " ").title(),
            source=source,
            enable_internet=True,
            is_private=True,
        )
        seen_progress: set[str] = set()
        for poll_index in range(self.max_poll_attempts):
            status = self.worker.status(self.kernel_slug)
            if poll_index % self.progress_poll_every == 0 or status.terminal:
                self._stream_progress(seen_progress)
            if status.terminal:
                if not status.successful:
                    detail = status.failure_message or status.status
                    try:
                        logs = self.worker.logs(self.kernel_slug)
                        if logs.strip():
                            detail = logs[-8000:]
                    except Exception:
                        pass
                    raise RuntimeError(f"Kaggle production image worker failed: {detail}")
                break
            if self.poll_interval:
                time.sleep(self.poll_interval)
        else:
            raise TimeoutError("Timed out waiting for production image worker")

        report = self._parse_report(
            self.worker.download_output_file(self.kernel_slug, "production_image_report.json")
        )
        gpu_name = str(report.get("gpu_name") or "").strip()
        if not gpu_name:
            raise ValueError("production image report has no GPU evidence")
        report_items = report.get("items")
        if not isinstance(report_items, dict):
            raise ValueError("production image report has no item map")

        candidates: dict[str, tuple[ProductionImageCandidate, ...]] = {}
        plans: dict[str, dict] = {}
        for item in items:
            entry = report_items.get(item.item_id)
            if not isinstance(entry, dict):
                raise ValueError(f"production image report missing item: {item.item_id}")
            profile = str(entry.get("semantic_profile") or "")
            if profile not in self.profile_engines:
                raise ValueError(f"production image profile invalid: {item.item_id}")
            plans[item.item_id] = entry
            raw_candidates = entry.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise ValueError(f"production image candidates missing: {item.item_id}")
            materialized: list[ProductionImageCandidate] = []
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    continue
                index = int(raw_candidate.get("candidate_index", -1))
                filename = str(raw_candidate.get("filename") or "")
                if index < 0 or not filename:
                    raise ValueError(f"production image candidate metadata invalid: {item.item_id}")
                image = self.worker.download_output_file(self.kernel_slug, filename)
                digest = sha256(image).hexdigest()
                if digest != raw_candidate.get("image_sha256"):
                    raise ValueError(f"production image candidate hash mismatch: {item.item_id}")
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
                        f"subject_name:{entry.get('subject_name')}",
                        f"engine_id:{raw_candidate.get('engine_id')}",
                        f"detector_gate_passed:{raw_candidate.get('detector_gate', {}).get('passed')}",
                        f"model:{model}",
                    ),
                )
                materialized.append(ProductionImageCandidate(
                    item_id=item.item_id,
                    candidate_index=index,
                    artifact=artifact,
                    report=raw_candidate,
                ))
            candidates[item.item_id] = tuple(materialized)
        return ProductionImageBatchOutput(candidates, plans, report)

    def _stream_progress(self, seen: set[str]) -> None:
        try:
            logs = self.worker.logs(self.kernel_slug)
        except Exception:
            return
        for line in logs.splitlines():
            marker = "AI_AGENT_PROGRESS "
            if marker not in line:
                continue
            payload = line[line.index(marker):].strip()
            if payload in seen:
                continue
            seen.add(payload)
            print(payload, flush=True)

    @staticmethod
    def _parse_report(data: bytes) -> dict:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("production image report is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("production image report must be an object")
        return payload

    def _build_worker_source(self, items: tuple[ProductionImageItem, ...]) -> str:
        config = {
            "profile_engines": self.profile_engines,
            "profile_positive_constraints": self.profile_positive_constraints,
            "profile_negative_constraints": self.profile_negative_constraints,
            "profile_dimensions": {key: list(value) for key, value in self.profile_dimensions.items()},
            "planner_model": self.planner_model,
            "detector_model": self.detector_model,
            "detector_box_threshold": self.detector_box_threshold,
            "detector_text_threshold": self.detector_text_threshold,
            "items": [
                {
                    "item_id": item.item_id,
                    "command": item.command,
                    "seed": item.seed,
                    "fallback_profile": item.fallback_profile,
                    "hard_gates": [
                        {
                            "label": gate.label,
                            "query": gate.query or gate.label,
                            "expected_count": gate.expected_count,
                        }
                        for gate in item.hard_gates
                    ],
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


def progress(step, **extra):
    payload = {"step": step, "elapsed_seconds": round(time.perf_counter() - STARTED, 3), **extra}
    print("AI_AGENT_PROGRESS " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


STARTED = time.perf_counter()
progress("worker_boot")

try:
    import torch
    from PIL import Image
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EDMDPMSolverMultistepScheduler
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
    )
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "diffusers<1", "transformers>=4.57,<5", "accelerate<2", "safetensors", "Pillow<13",
    ])
    import torch
    from PIL import Image
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EDMDPMSolverMultistepScheduler
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForZeroShotObjectDetection

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available")
gpu_name = torch.cuda.get_device_name(0)
progress("dependencies_ready", gpu=gpu_name)


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
    return not value or value in {
        "english image description", "english image prompt", "english prompt",
        "image description", "subject name", "exact english subject name",
    } or "<" in value or ">" in value


def words(text):
    return [token for token in "".join(ch.casefold() if ch.isalnum() else " " for ch in (text or "")).split() if len(token) >= 3]


def subject_matches_prompt(subject_name, prompt):
    subject_tokens = [token for token in words(subject_name) if token not in {"the", "one", "main"}]
    prompt_tokens = set(words(prompt))
    return bool(subject_tokens) and any(token in prompt_tokens for token in subject_tokens)


def parse_compiled(raw):
    raw = (raw or "").strip().strip(chr(96)).strip()
    result = {"prompt": "", "subject_class": "", "style_class": "", "subject_name": ""}
    aliases = {
        "PROMPT": "prompt", "PROMT": "prompt", "PROMP": "prompt",
        "SUBJECTCLASS": "subject_class", "SUBJECT": "subject_class",
        "STYLECLASS": "style_class", "STYLE": "style_class",
        "SUBJECTNAME": "subject_name", "SPECIES": "subject_name", "ENTITY": "subject_name",
    }
    cleaned = raw.replace("：", ":")
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= first < last:
        try:
            payload = json.loads(cleaned[first:last + 1])
        except Exception:
            payload = None
        if isinstance(payload, dict):
            result["prompt"] = str(payload.get("PROMPT") or payload.get("prompt") or "").strip()
            result["subject_class"] = normalize_subject(str(payload.get("SUBJECT_CLASS") or payload.get("subject_class") or ""))
            result["style_class"] = normalize_style(str(payload.get("STYLE_CLASS") or payload.get("style_class") or ""))
            result["subject_name"] = str(payload.get("SUBJECT_NAME") or payload.get("subject_name") or payload.get("species") or "").strip()
            return result
    for line in cleaned.splitlines():
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


progress("planner_loading", model=CONFIG["planner_model"])
planner_load_started = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(CONFIG["planner_model"])
planner_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["planner_model"], torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)
planner_model.eval()
planner_ready = time.perf_counter()
planner = {}
for item_index, item in enumerate(CONFIG["items"]):
    progress("planner_item", item_id=item["item_id"], item_index=item_index + 1, item_total=len(CONFIG["items"]))
    instruction = (
        "MEDIA_COMMAND_COMPILE\n"
        "Translate and preserve the user's exact named subject/species. Never generalize a specific species into a broader or different animal. "
        "Understand USER_COMMAND semantically without a fixed species list. Preserve subject count and requested visual style. "
        "SUBJECT_CLASS must be animal, human, or general. STYLE_CLASS must be photo, 3d, mascot, illustration, or general. "
        "SUBJECT_NAME must be the exact English common name of the requested main subject/species and that same name must appear in PROMPT. "
        "Do not invent anatomy counts or appendages. Return exactly four labeled lines and no commentary:\n"
        "SUBJECT_CLASS: animal|human|general\n"
        "STYLE_CLASS: photo|3d|mascot|illustration|general\n"
        "SUBJECT_NAME: exact English common subject/species name\n"
        "PROMPT: complete English image description containing SUBJECT_NAME\n"
        "USER_COMMAND: " + item["command"]
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(planner_model.device)
    with torch.inference_mode():
        generated = planner_model.generate(**inputs, max_new_tokens=220, do_sample=False, use_cache=True, repetition_penalty=1.04)
    raw = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    parsed = parse_compiled(raw)
    valid = (
        len(parsed["prompt"]) >= 20
        and not looks_vietnamese(parsed["prompt"])
        and not placeholder(parsed["prompt"])
        and not placeholder(parsed["subject_name"])
        and subject_matches_prompt(parsed["subject_name"], parsed["prompt"])
        and parsed["subject_class"] in {"animal", "human", "general"}
        and parsed["style_class"] in {"photo", "3d", "mascot", "illustration", "general"}
    )
    if not valid:
        raise RuntimeError("semantic planner failed exact-subject contract for item " + item["item_id"] + ": " + raw[:700])
    planner[item["item_id"]] = {"raw": raw, "parsed": parsed}
    del generated, inputs

del planner_model, tokenizer
gc.collect(); torch.cuda.empty_cache()
progress("planner_complete", model_load_seconds=round(planner_ready - planner_load_started, 3))

compiled = []
for item in CONFIG["items"]:
    parsed = planner[item["item_id"]]["parsed"]
    profile = choose_profile(parsed["subject_class"], parsed["style_class"], item["fallback_profile"])
    engine = CONFIG["profile_engines"][profile]
    width, height = CONFIG["profile_dimensions"][profile]
    positive = CONFIG["profile_positive_constraints"][profile]
    negative = CONFIG["profile_negative_constraints"][profile]
    prompt = parsed["prompt"].rstrip(" .") + ". Quality requirements: " + positive.rstrip(" .") + "."
    if profile == "mascot_premium":
        prompt += (
            " Species-anatomy guard: keep only anatomy naturally belonging to the named species; never duplicate horns, antlers, tusks, ears, limbs, tail, wings or fins. "
            "Cute stylization may change proportions but must not create extra body parts. Keep the character fully inside a clean unbranded frame."
        )
        negative = negative.rstrip(" ,") + ", extra horns, duplicate horns, extra antlers, extra tusks, extra ears, extra legs, extra tails, duplicated appendages, pseudo-logo, signature, watermark"
    else:
        negative = negative.rstrip(" ,") + ", ear tag, livestock tag, identification tag, numbered tag, text on animal, artificial HDR, overprocessed photo"
    compiled.append({
        **item,
        "prompt": prompt,
        "negative_prompt": negative,
        "subject_class": parsed["subject_class"],
        "style_class": parsed["style_class"],
        "subject_name": parsed["subject_name"],
        "semantic_profile": profile,
        "planner_raw": planner[item["item_id"]]["raw"],
        "width": int(width), "height": int(height), "engine": engine,
    })

engine_groups = {}
for item in compiled:
    engine_groups.setdefault(item["engine"]["engine_id"], []).append(item)

item_reports = {}
engine_timings = {}
for engine_id, engine_items in engine_groups.items():
    engine = engine_items[0]["engine"]
    progress("engine_loading", engine_id=engine_id, model=engine["model"], item_ids=[x["item_id"] for x in engine_items])
    load_started = time.perf_counter()
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
    ready = time.perf_counter()
    progress("engine_ready", engine_id=engine_id, model_load_seconds=round(ready - load_started, 3))

    for item in engine_items:
        candidates = []
        for candidate_index in range(int(engine["candidate_count"])):
            progress("candidate_generating", item_id=item["item_id"], engine_id=engine_id, candidate=candidate_index + 1, candidate_total=int(engine["candidate_count"]))
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
                prompt=item["prompt"], negative_prompt=item["negative_prompt"] or None,
                width=render_width, height=render_height,
                num_inference_steps=int(engine["steps"]), guidance_scale=float(engine["guidance"]), generator=generator,
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
                "candidate_index": candidate_index, "filename": filename, "seed": seed,
                "image_sha256": sha256(data).hexdigest(), "engine_id": engine_id,
                "model": engine["model"], "scheduler": engine["scheduler"],
                "render_width": render_width, "render_height": render_height,
                "safe_border_crop": safe_crop, "generation_seconds": round(time.perf_counter() - gen_started, 3),
                "detector_gate": {"passed": True, "checks": []},
            })
            progress("candidate_generated", item_id=item["item_id"], candidate=candidate_index + 1, generation_seconds=candidates[-1]["generation_seconds"])
        item_reports[item["item_id"]] = {
            "user_command": item["command"], "prompt": item["prompt"], "negative_prompt": item["negative_prompt"],
            "subject_class": item["subject_class"], "style_class": item["style_class"], "subject_name": item["subject_name"],
            "semantic_profile": item["semantic_profile"], "planner_raw": item["planner_raw"],
            "width": item["width"], "height": item["height"], "engine_id": engine_id,
            "hard_gates": item["hard_gates"], "candidates": candidates,
        }
    engine_timings[engine_id] = {
        "model": engine["model"], "model_load_seconds": round(ready - load_started, 3),
        "total_seconds": round(time.perf_counter() - load_started, 3),
    }
    del pipe
gc.collect(); torch.cuda.empty_cache()
progress("generation_complete", engine_ids=list(engine_groups))

# Optional detector stage. Counts are supplied only by trusted test/task contracts;
# the planner never invents expected anatomy counts.
gated_items = [item for item in compiled if item["hard_gates"]]
detector_timing = None
if gated_items:
    progress("detector_loading", model=CONFIG["detector_model"], item_ids=[x["item_id"] for x in gated_items])
    detector_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(CONFIG["detector_model"])
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(CONFIG["detector_model"]).to("cuda")
    detector.eval()
    detector_ready = time.perf_counter()
    progress("detector_ready", model_load_seconds=round(detector_ready - detector_started, 3))

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(1e-6, area_a + area_b - inter)

    def dedupe(boxes, scores, threshold=0.35):
        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        keep = []
        for idx in order:
            if all(iou(boxes[idx], boxes[chosen]) < threshold for chosen in keep):
                keep.append(idx)
        return keep

    for item in gated_items:
        for candidate in item_reports[item["item_id"]]["candidates"]:
            progress("detector_candidate", item_id=item["item_id"], candidate=candidate["candidate_index"] + 1)
            image = Image.open(Path("/kaggle/working") / candidate["filename"]).convert("RGB")
            checks = []
            passed = True
            for gate in item["hard_gates"]:
                query = str(gate.get("query") or gate["label"]).strip().rstrip(".") + "."
                inputs = processor(images=image, text=query, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    outputs = detector(**inputs)
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs.input_ids,
                    box_threshold=float(CONFIG["detector_box_threshold"]),
                    text_threshold=float(CONFIG["detector_text_threshold"]),
                    target_sizes=[image.size[::-1]],
                )[0]
                boxes = result["boxes"].detach().cpu().tolist()
                scores = result["scores"].detach().cpu().tolist()
                keep = dedupe(boxes, scores)
                count = len(keep)
                expected = int(gate["expected_count"])
                ok = count == expected
                passed = passed and ok
                checks.append({
                    "label": gate["label"], "query": query, "expected_count": expected,
                    "detected_count": count, "passed": ok,
                    "scores": [round(float(scores[idx]), 4) for idx in keep],
                    "boxes": [[round(float(v), 1) for v in boxes[idx]] for idx in keep],
                })
                del inputs, outputs
            candidate["detector_gate"] = {"passed": passed, "checks": checks}
            progress("detector_result", item_id=item["item_id"], candidate=candidate["candidate_index"] + 1, passed=passed, checks=checks)
    detector_timing = {
        "model_load_seconds": round(detector_ready - detector_started, 3),
        "total_seconds": round(time.perf_counter() - detector_started, 3),
    }
    del detector, processor
gc.collect(); torch.cuda.empty_cache()
progress("detector_complete", enabled=bool(gated_items))

report = {
    "gpu_name": gpu_name,
    "planner_model": CONFIG["planner_model"],
    "engine_timings": engine_timings,
    "detector_model": CONFIG["detector_model"] if gated_items else None,
    "detector_timing": detector_timing,
    "worker_total_seconds": round(time.perf_counter() - STARTED, 3),
    "items": item_reports,
}
Path("/kaggle/working/production_image_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
progress("worker_complete", worker_total_seconds=report["worker_total_seconds"])
print("AI_AGENT_PRODUCTION_IMAGE_BATCH_OK", flush=True)
'''
        return textwrap.dedent(template).replace("__CONFIG_JSON__", repr(config_json)).strip() + "\n"
