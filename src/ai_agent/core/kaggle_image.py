"""On-demand text-to-image generation on a self-controlled Kaggle GPU worker."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import textwrap
import time

from .image_model import ImageArtifact, ImageGenerationRequest
from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker


@dataclass
class KaggleImageProvider:
    """Run an open-weight image model on Kaggle GPU and return verified bytes."""

    worker: KaggleGpuWorker
    model: str = "playgroundai/playground-v2.5-1024px-aesthetic"
    model_variant: str | None = None
    kernel_slug: str = "ai-agent-image-worker"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    inference_steps: int = 30
    guidance_scale: float = 3.0
    scheduler: str = "edm_dpm"
    prompt_alignment_threshold: float = 0.22
    visual_quality_margin_threshold: float = 0.015
    enforce_visual_quality_margin: bool = True
    enable_clip_precheck: bool = True
    enable_cpu_offload: bool = True
    enable_inline_planner: bool = False
    inline_planner_fast_model: str = "Qwen/Qwen3-0.6B"
    inline_planner_final_model: str = "Qwen/Qwen3-1.7B"
    inline_planner_raw_command: str = ""
    inline_planner_positive_constraints: str = ""
    enable_inline_vlm: bool = False
    inline_vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    inline_vlm_profile: str = ""
    inline_vlm_rubric: tuple[str, ...] = ()
    inline_vlm_expected_subject_count: int | None = 1
    inline_vlm_min_score: float = 9.0
    inline_vlm_attention_backend: str = "sdpa"
    quality_good_text: str = (
        "premium production-ready image, coherent geometry, crisp detail, professional composition and lighting, "
        "clean materials and textures, no obvious AI artifacts"
    )
    quality_bad_texts: tuple[str, ...] = (
        "bad AI image with warped geometry, duplicated subjects or objects and broken perspective",
        "low quality blurry noisy unfinished image with melted details and obvious AI artifacts",
    )
    provider: str = "kaggle-gpu-local-model"
    last_report: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if self.scheduler not in {"default", "edm_dpm", "dpm_karras"}:
            raise ValueError("scheduler must be default, edm_dpm or dpm_karras")
        if not self.quality_good_text.strip() or not self.quality_bad_texts:
            raise ValueError("visual quality critic text must be configured")
        if any(not item.strip() for item in self.quality_bad_texts):
            raise ValueError("visual quality critic bad texts must be non-empty")
        if not -1.0 <= self.prompt_alignment_threshold <= 1.0:
            raise ValueError("prompt_alignment_threshold must be in [-1, 1]")
        if not -2.0 <= self.visual_quality_margin_threshold <= 2.0:
            raise ValueError("visual_quality_margin_threshold must be in [-2, 2]")
        if self.enable_inline_planner:
            if not self.inline_planner_fast_model.strip() or not self.inline_planner_final_model.strip():
                raise ValueError("inline planner models are required")
            if not self.inline_planner_raw_command.strip():
                raise ValueError("inline_planner_raw_command is required")
            if not self.inline_planner_positive_constraints.strip():
                raise ValueError("inline_planner_positive_constraints is required")
        if self.enable_inline_vlm:
            if not self.inline_vlm_model.strip():
                raise ValueError("inline_vlm_model is required")
            if not self.inline_vlm_profile.strip():
                raise ValueError("inline_vlm_profile is required")
            if not self.inline_vlm_rubric or any(not item.strip() for item in self.inline_vlm_rubric):
                raise ValueError("inline_vlm_rubric must contain non-empty criteria")
            if self.inline_vlm_expected_subject_count is not None and self.inline_vlm_expected_subject_count <= 0:
                raise ValueError("inline_vlm_expected_subject_count must be positive")
            if not 0 <= self.inline_vlm_min_score <= 10:
                raise ValueError("inline_vlm_min_score must be in [0, 10]")
            if self.inline_vlm_attention_backend not in {"sdpa", "eager"}:
                raise ValueError("inline_vlm_attention_backend must be sdpa or eager")

    def generate(self, request: ImageGenerationRequest) -> ImageArtifact:
        assert_core_invariants()
        if request.width % 8 or request.height % 8:
            raise ValueError("image dimensions must be divisible by 8")

        source = self._build_worker_source(request)
        kernel_title = self.kernel_slug.replace("-", " ").title()
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=kernel_title,
            source=source,
            enable_internet=True,
            is_private=True,
        )

        for _ in range(self.max_poll_attempts):
            status = self.worker.status(self.kernel_slug)
            if status.terminal:
                if not status.successful:
                    raise RuntimeError(
                        f"Kaggle image worker failed: {status.status} {status.failure_message}"
                    )
                break
            if self.poll_interval:
                time.sleep(self.poll_interval)
        else:
            raise TimeoutError("Timed out waiting for Kaggle image worker")

        image = self.worker.download_output_file(self.kernel_slug, "generated.png")
        report_bytes = self.worker.download_output_file(self.kernel_slug, "image_report.json")
        try:
            report = json.loads(report_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Kaggle image report is invalid JSON") from exc
        if not isinstance(report, dict):
            raise ValueError("Kaggle image report must be a JSON object")
        self.last_report = report

        digest = sha256(image).hexdigest()
        if report.get("image_sha256") != digest:
            raise ValueError("Downloaded image hash does not match Kaggle report")
        if int(report.get("width", 0)) != request.width or int(report.get("height", 0)) != request.height:
            raise ValueError("Generated image dimensions do not match the request")
        if report.get("model") != self.model:
            raise ValueError("Kaggle report model does not match configured model")
        if not str(report.get("gpu_name", "")).strip():
            raise ValueError("Kaggle image report does not contain GPU evidence")
        prompt_alignment_score: float | None = None
        visual_quality_margin: float | None = None
        visual_defect_score: float | None = None
        if self.enable_clip_precheck:
            try:
                prompt_alignment_score = float(report.get("prompt_alignment_score"))
            except (TypeError, ValueError):
                prompt_alignment_score = -1.0
            if prompt_alignment_score < self.prompt_alignment_threshold:
                raise ValueError(
                    "prompt alignment below threshold: "
                    f"{prompt_alignment_score:.4f} < {self.prompt_alignment_threshold:.4f}"
                )
            try:
                visual_quality_margin = float(report.get("visual_quality_margin"))
            except (TypeError, ValueError):
                visual_quality_margin = -2.0
            try:
                visual_defect_score = float(report.get("visual_defect_score"))
            except (TypeError, ValueError):
                visual_defect_score = 0.0
            if (
                self.enforce_visual_quality_margin
                and visual_quality_margin < self.visual_quality_margin_threshold
            ):
                raise ValueError(
                    "visual quality margin below threshold: "
                    f"{visual_quality_margin:.4f} < {self.visual_quality_margin_threshold:.4f}"
                )

        return ImageArtifact(
            data=image,
            mime_type="image/png",
            provider=self.provider,
            model=self.model,
            evidence=(
                f"kaggle_kernel:{submission.ref}",
                f"image_sha256:{digest}",
                f"dimensions:{request.width}x{request.height}",
                f"seed:{report.get('seed')}",
                *((f"prompt_alignment_score:{prompt_alignment_score:.6f}",) if prompt_alignment_score is not None else ()),
                *((f"visual_quality_margin:{visual_quality_margin:.6f}",) if visual_quality_margin is not None else ()),
                *((f"visual_defect_score:{visual_defect_score:.6f}",) if visual_defect_score is not None else ()),
                f"clip_precheck:{self.enable_clip_precheck}",
                f"cpu_offload:{self.enable_cpu_offload}",
                f"inline_planner:{self.enable_inline_planner}",
                *((f"inline_planner_model:{report.get('inline_planner', {}).get('model')}",) if self.enable_inline_planner else ()),
                *((f"inline_planner_fallback:{bool(report.get('inline_planner', {}).get('fallback_used'))}",) if self.enable_inline_planner else ()),
                f"inline_vlm:{self.enable_inline_vlm}",
                f"hf_xet_high_performance:{bool(report.get('hf_xet_high_performance'))}",
                *((f"inline_vlm_model:{self.inline_vlm_model}",) if self.enable_inline_vlm else ()),
                f"visual_quality_margin_enforced:{self.enforce_visual_quality_margin and self.enable_clip_precheck}",
                f"gpu:{report.get('gpu_name')}",
                f"model:{self.model}",
                f"model_variant:{self.model_variant or 'default'}",
            ),
        )

    def _build_worker_source(self, request: ImageGenerationRequest) -> str:
        seed = 0 if request.seed is None else request.seed
        config = {
            "model": self.model,
            "model_variant": self.model_variant,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "seed": seed,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "scheduler": self.scheduler,
            "quality_good_text": self.quality_good_text,
            "quality_bad_texts": list(self.quality_bad_texts),
            "prompt_alignment_threshold": self.prompt_alignment_threshold,
            "visual_quality_margin_threshold": self.visual_quality_margin_threshold,
            "enable_clip_precheck": self.enable_clip_precheck,
            "enable_cpu_offload": self.enable_cpu_offload,
            "enable_inline_planner": self.enable_inline_planner,
            "inline_planner_fast_model": self.inline_planner_fast_model,
            "inline_planner_final_model": self.inline_planner_final_model,
            "inline_planner_raw_command": self.inline_planner_raw_command,
            "inline_planner_positive_constraints": self.inline_planner_positive_constraints,
            "enable_inline_vlm": self.enable_inline_vlm,
            "inline_vlm_model": self.inline_vlm_model,
            "inline_vlm_profile": self.inline_vlm_profile,
            "inline_vlm_rubric": list(self.inline_vlm_rubric),
            "inline_vlm_expected_subject_count": self.inline_vlm_expected_subject_count,
            "inline_vlm_min_score": self.inline_vlm_min_score,
            "inline_vlm_attention_backend": self.inline_vlm_attention_backend,
        }
        config_json = json.dumps(config, ensure_ascii=False)
        return textwrap.dedent(
            f"""
            from __future__ import annotations

            from hashlib import sha256
            import gc
            import json
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            CONFIG = json.loads({config_json!r})
            os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

            try:
                import torch
                import torch.nn.functional as F
                from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EDMDPMSolverMultistepScheduler
                from transformers import (
                    AutoModelForCausalLM,
                    AutoProcessor,
                    AutoTokenizer,
                    CLIPImageProcessor,
                    CLIPTokenizer,
                    CLIPVisionModelWithProjection,
                    CLIPTextModelWithProjection,
                    Qwen3VLForConditionalGeneration,
                )
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "diffusers<1", "transformers>=4.57,<5", "accelerate<2", "safetensors",
                ])
                import torch
                import torch.nn.functional as F
                from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EDMDPMSolverMultistepScheduler
                from transformers import (
                    AutoModelForCausalLM,
                    AutoProcessor,
                    AutoTokenizer,
                    CLIPImageProcessor,
                    CLIPTokenizer,
                    CLIPVisionModelWithProjection,
                    CLIPTextModelWithProjection,
                    Qwen3VLForConditionalGeneration,
                )

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            worker_started = time.perf_counter()
            gpu_name = torch.cuda.get_device_name(0)
            generator = torch.Generator(device="cpu").manual_seed(int(CONFIG["seed"]))

            inline_planner = None
            if CONFIG["enable_inline_planner"]:
                planner_started = time.perf_counter()

                def parse_compiled_prompt(raw):
                    raw = (raw or "").strip().strip(chr(96)).strip()
                    if not raw:
                        return ""
                    cleaned = raw
                    if cleaned.casefold().startswith("json"):
                        cleaned = cleaned[4:].lstrip("\\n :")
                    first_brace = cleaned.find("{{")
                    last_brace = cleaned.rfind("}}")
                    candidates = [cleaned]
                    if 0 <= first_brace < last_brace:
                        candidates.insert(0, cleaned[first_brace:last_brace + 1])
                    for candidate in candidates:
                        try:
                            payload = json.loads(candidate)
                        except Exception:
                            continue
                        if isinstance(payload, dict):
                            for key in ("PROMPT", "prompt", "image_prompt", "description"):
                                value = payload.get(key)
                                if isinstance(value, str) and value.strip():
                                    return value.strip()
                    normalized = cleaned.replace("：", ":")
                    for line in normalized.splitlines():
                        line = line.strip().lstrip("-*# ").strip()
                        if ":" not in line:
                            continue
                        key, value = line.split(":", 1)
                        key = "".join(ch for ch in key.strip().upper() if ch.isalpha())
                        if key in {{"PROMPT", "PROMT", "PROMP"}} and value.strip():
                            return value.strip().strip(chr(96)).strip()
                    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
                    if len(lines) == 1:
                        line = lines[0]
                        if ":" in line and line.casefold().startswith(("here is", "here's", "image prompt", "visual prompt")):
                            line = line.split(":", 1)[1].strip()
                        return line
                    return ""

                def looks_vietnamese(text):
                    chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
                    return any(ch in chars for ch in text.casefold())

                def compile_prompt(model_name):
                    load_started = time.perf_counter()
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    planner_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                    planner_model.eval()
                    model_ready = time.perf_counter()
                    instruction = (
                        "MEDIA_COMMAND_COMPILE\\n"
                        "Translate the USER_COMMAND into ONE concise ENGLISH still-image description. "
                        "Preserve subject count, species/person/object, action, location, camera/framing, lighting, "
                        "and visual style. Do not invent alternatives. Return exactly one line: "
                        "PROMPT: <English image description>\\nUSER_COMMAND: "
                        + CONFIG["inline_planner_raw_command"]
                    )
                    messages = [{{"role": "user", "content": instruction}}]
                    template_kwargs = {{
                        "tokenize": False,
                        "add_generation_prompt": True,
                        "enable_thinking": False,
                    }}
                    rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
                    inputs = tokenizer([rendered], return_tensors="pt").to(planner_model.device)
                    generate_started = time.perf_counter()
                    with torch.inference_mode():
                        generated = planner_model.generate(
                            **inputs,
                            max_new_tokens=120,
                            do_sample=False,
                            use_cache=True,
                            repetition_penalty=1.05,
                            no_repeat_ngram_size=3,
                        )
                    new_tokens = generated[:, inputs.input_ids.shape[1]:]
                    raw_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                    generate_finished = time.perf_counter()
                    compiled = parse_compiled_prompt(raw_text)
                    valid = (
                        len(compiled) >= 20
                        and not looks_vietnamese(compiled)
                        and "USER_COMMAND" not in compiled
                    )
                    timing = {{
                        "model_load_seconds": round(model_ready - load_started, 3),
                        "generate_seconds": round(generate_finished - generate_started, 3),
                        "total_seconds": round(generate_finished - load_started, 3),
                    }}
                    del generated, inputs, planner_model, tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    return compiled if valid else "", raw_text, timing

                compiled_prompt, planner_raw, planner_timing = compile_prompt(
                    CONFIG["inline_planner_fast_model"]
                )
                planner_model_used = CONFIG["inline_planner_fast_model"]
                planner_fallback_used = False
                if not compiled_prompt:
                    planner_fallback_used = True
                    compiled_prompt, planner_raw, planner_timing = compile_prompt(
                        CONFIG["inline_planner_final_model"]
                    )
                    planner_model_used = CONFIG["inline_planner_final_model"]
                if not compiled_prompt:
                    raise RuntimeError("inline media planner did not produce a usable English prompt")
                CONFIG["prompt"] = (
                    compiled_prompt.rstrip(" .")
                    + ". Quality requirements: "
                    + CONFIG["inline_planner_positive_constraints"].strip().rstrip(" .")
                    + "."
                )
                inline_planner = {{
                    "model": planner_model_used,
                    "fallback_used": planner_fallback_used,
                    "raw_output": planner_raw,
                    "compiled_prompt": compiled_prompt,
                    "final_prompt": CONFIG["prompt"],
                    "timings": planner_timing,
                    "total_stage_seconds": round(time.perf_counter() - planner_started, 3),
                }}

            image_model_load_started = time.perf_counter()

            pipe_kwargs = {{"torch_dtype": torch.float16}}
            if CONFIG.get("model_variant"):
                pipe_kwargs["variant"] = CONFIG["model_variant"]
            pipe = AutoPipelineForText2Image.from_pretrained(
                CONFIG["model"],
                **pipe_kwargs,
            )
            if CONFIG["scheduler"] == "edm_dpm":
                pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            elif CONFIG["scheduler"] == "dpm_karras":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
            if CONFIG["enable_cpu_offload"]:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to("cuda")
            pipe.enable_vae_slicing()
            image_model_ready = time.perf_counter()

            clip_image_processor = None
            clip_tokenizer = None
            clip_vision = None
            clip_text = None
            quality_good = None
            quality_bad = ()
            if CONFIG["enable_clip_precheck"]:
                clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
                clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                clip_vision = CLIPVisionModelWithProjection.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).eval()
                clip_text = CLIPTextModelWithProjection.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).eval()

                def image_embedding(image):
                    values = clip_image_processor(images=image, return_tensors="pt").pixel_values
                    with torch.no_grad():
                        vector = clip_vision(pixel_values=values).image_embeds[0].float()
                    return F.normalize(vector, dim=0)

                def text_embedding(text):
                    values = clip_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        vector = clip_text(**values).text_embeds[0].float()
                    return F.normalize(vector, dim=0)

                quality_good = text_embedding(CONFIG["quality_good_text"])
                quality_bad = tuple(
                    text_embedding(text) for text in CONFIG["quality_bad_texts"]
                )

            result = pipe(
                prompt=CONFIG["prompt"],
                negative_prompt=CONFIG["negative_prompt"] or None,
                width=int(CONFIG["width"]),
                height=int(CONFIG["height"]),
                num_inference_steps=int(CONFIG["inference_steps"]),
                guidance_scale=float(CONFIG["guidance_scale"]),
                generator=generator,
            )
            image = result.images[0]
            prompt_alignment_score = None
            visual_quality_score = None
            visual_defect_score = None
            visual_quality_margin = None
            if CONFIG["enable_clip_precheck"]:
                generated_embedding = image_embedding(image)
                prompt_alignment_score = float(
                    torch.dot(text_embedding(CONFIG["prompt"]), generated_embedding).item()
                )
                visual_quality_score = float(torch.dot(quality_good, generated_embedding).item())
                visual_defect_score = max(
                    float(torch.dot(bad, generated_embedding).item()) for bad in quality_bad
                )
                visual_quality_margin = visual_quality_score - visual_defect_score

            output = Path("/kaggle/working/generated.png")
            image.save(output, format="PNG")
            data = output.read_bytes()
            digest = sha256(data).hexdigest()
            image_ready = time.perf_counter()

            inline_vlm = None
            if CONFIG["enable_inline_vlm"]:
                # Release the diffusion stack before loading the small VLM in the same GPU session.
                try:
                    del result
                except Exception:
                    pass
                try:
                    del pipe
                except Exception:
                    pass
                clip_vision = None
                clip_text = None
                gc.collect()
                torch.cuda.empty_cache()

                qa_model_load_started = time.perf_counter()
                qa_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    CONFIG["inline_vlm_model"],
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation=CONFIG["inline_vlm_attention_backend"],
                )
                qa_processor = AutoProcessor.from_pretrained(CONFIG["inline_vlm_model"])
                qa_model.eval()
                qa_model_ready = time.perf_counter()

                review_image = image.copy().convert("RGB")
                review_image.thumbnail((1024, 1024))
                review_width, review_height = review_image.size
                views = [
                    review_image,
                    review_image.crop((0, 0, review_width, max(1, review_height // 2))),
                    review_image.crop((0, review_height // 2, review_width, review_height)),
                ]
                rubric = "\\n".join("- " + item for item in CONFIG["inline_vlm_rubric"])
                instruction = (
                    "You are the strict final visual-quality inspector for a production image pipeline. "
                    "Judge only what is actually visible. PROFILE: " + CONFIG["inline_vlm_profile"]
                    + "; ORIGINAL PROMPT: " + CONFIG["prompt"]
                    + "; EXPECTED MAIN SUBJECT COUNT: " + str(CONFIG["inline_vlm_expected_subject_count"])
                    + "; QUALITY RUBRIC: " + rubric + ". "
                    "You receive full frame, upper crop and lower crop of the SAME image. "
                    "Inspect subject count, anatomy/geometry, limbs/paws/hands, face/eyes, object connections, "
                    "materials, lighting, perspective, texture continuity, background coherence and AI artifacts. "
                    "For animal photos, natural camera realism and correct separated legs/paws/joints/ground contact are hard requirements; "
                    "reject fused, missing, extra or ambiguous limbs, impossible gait, fake HDR, oversharpening, plastic fur or synthetic polish. "
                    "Score 0-10 for q=overall quality, m=prompt match, s=structure/anatomy, d=detail, a=aesthetic, "
                    "c=composition and b=benchmark match. Set p=true ONLY if every score is at least "
                    + str(CONFIG["inline_vlm_min_score"])
                    + ", subject count is correct, and x/f/u are empty. If uncertain about a critical region, p=false. "
                    "Return ONLY compact JSON: "
                    + '{{"p":true,"q":9,"m":9,"s":9,"d":9,"a":9,"c":9,"b":9,"n":1,"x":[],"f":[],"u":[],"i":[]}}'
                )
                content = [{{"type": "image", "image": view}} for view in views]
                content.append({{"type": "text", "text": instruction}})
                messages = [{{"role": "user", "content": content}}]
                qa_inputs = qa_processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(qa_model.device)
                qa_review_started = time.perf_counter()
                with torch.inference_mode():
                    qa_generated = qa_model.generate(
                        **qa_inputs,
                        max_new_tokens=180,
                        do_sample=False,
                        use_cache=True,
                        repetition_penalty=1.03,
                    )
                qa_generated_trimmed = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(qa_inputs.input_ids, qa_generated)
                ]
                qa_review_text = qa_processor.batch_decode(
                    qa_generated_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                if not qa_review_text:
                    raise RuntimeError("inline vision-quality model returned empty review")
                qa_finished = time.perf_counter()
                inline_vlm = {{
                    "model": CONFIG["inline_vlm_model"],
                    "attention_backend": CONFIG["inline_vlm_attention_backend"],
                    "review_text": qa_review_text,
                    "timings": {{
                        "model_load_seconds": round(qa_model_ready - qa_model_load_started, 3),
                        "review_seconds": round(qa_finished - qa_review_started, 3),
                        "total_seconds": round(qa_finished - qa_model_load_started, 3),
                    }},
                }}

            report = {{
                "image_sha256": digest,
                "width": image.width,
                "height": image.height,
                "seed": int(CONFIG["seed"]),
                "model": CONFIG["model"],
                "model_variant": CONFIG.get("model_variant"),
                "gpu_name": gpu_name,
                "prompt_sha256": sha256(CONFIG["prompt"].encode("utf-8")).hexdigest(),
                "prompt_alignment_score": prompt_alignment_score,
                "visual_quality_score": visual_quality_score,
                "visual_defect_score": visual_defect_score,
                "visual_quality_margin": visual_quality_margin,
                "clip_precheck": bool(CONFIG["enable_clip_precheck"]),
                "inline_planner": inline_planner,
                "compiled_prompt": CONFIG["prompt"],
                "inline_vlm": inline_vlm,
                "hf_xet_high_performance": os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1",
                "timings": {{
                    "image_model_load_seconds": round(image_model_ready - image_model_load_started, 3),
                    "image_generation_seconds": round(image_ready - image_model_ready, 3),
                    "worker_total_seconds": round(time.perf_counter() - worker_started, 3),
                }},
            }}
            Path("/kaggle/working/image_report.json").write_text(
                json.dumps(report, indent=2) + "\\n",
                encoding="utf-8",
            )
            print("AI_AGENT_IMAGE_OK")
            print(json.dumps(report, indent=2))
            """
        ).strip() + "\n"
