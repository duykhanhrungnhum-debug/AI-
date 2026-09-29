"""On-demand text-to-image generation on a self-controlled Kaggle GPU worker."""
from __future__ import annotations

from dataclasses import dataclass
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
    quality_good_text: str = (
        "premium production-ready image, coherent geometry, crisp detail, professional composition and lighting, "
        "clean materials and textures, no obvious AI artifacts"
    )
    quality_bad_texts: tuple[str, ...] = (
        "bad AI image with warped geometry, duplicated subjects or objects and broken perspective",
        "low quality blurry noisy unfinished image with melted details and obvious AI artifacts",
    )
    provider: str = "kaggle-gpu-local-model"

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
                f"visual_quality_margin_enforced:{self.enforce_visual_quality_margin and self.enable_clip_precheck}",
                f"gpu:{report.get('gpu_name')}",
                f"model:{self.model}",
            ),
        )

    def _build_worker_source(self, request: ImageGenerationRequest) -> str:
        seed = 0 if request.seed is None else request.seed
        config = {
            "model": self.model,
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
        }
        config_json = json.dumps(config, ensure_ascii=False)
        return textwrap.dedent(
            f"""
            from __future__ import annotations

            from hashlib import sha256
            import json
            import subprocess
            import sys
            from pathlib import Path

            CONFIG = json.loads({config_json!r})

            try:
                import torch
                import torch.nn.functional as F
                from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EDMDPMSolverMultistepScheduler
                from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "diffusers<1", "transformers<5", "accelerate<2", "safetensors",
                ])
                import torch
                import torch.nn.functional as F
                from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EDMDPMSolverMultistepScheduler
                from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            generator = torch.Generator(device="cpu").manual_seed(int(CONFIG["seed"]))

            pipe = AutoPipelineForText2Image.from_pretrained(
                CONFIG["model"],
                torch_dtype=torch.float16,
            )
            if CONFIG["scheduler"] == "edm_dpm":
                pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            elif CONFIG["scheduler"] == "dpm_karras":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

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

            report = {{
                "image_sha256": digest,
                "width": image.width,
                "height": image.height,
                "seed": int(CONFIG["seed"]),
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "prompt_sha256": sha256(CONFIG["prompt"].encode("utf-8")).hexdigest(),
                "prompt_alignment_score": prompt_alignment_score,
                "visual_quality_score": visual_quality_score,
                "visual_defect_score": visual_defect_score,
                "visual_quality_margin": visual_quality_margin,
                "clip_precheck": bool(CONFIG["enable_clip_precheck"]),
            }}
            Path("/kaggle/working/image_report.json").write_text(
                json.dumps(report, indent=2) + "\\n",
                encoding="utf-8",
            )
            print("AI_AGENT_IMAGE_OK")
            print(json.dumps(report, indent=2))
            """
        ).strip() + "\n"
