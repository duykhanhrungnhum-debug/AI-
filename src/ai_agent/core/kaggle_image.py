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
    model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    kernel_slug: str = "ai-agent-image-worker"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    inference_steps: int = 20
    guidance_scale: float = 7.5
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
                from diffusers import StableDiffusionPipeline
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "diffusers<1", "transformers<5", "accelerate<2", "safetensors",
                ])
                import torch
                from diffusers import StableDiffusionPipeline

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            generator = torch.Generator(device="cuda").manual_seed(int(CONFIG["seed"]))

            pipe = StableDiffusionPipeline.from_pretrained(
                CONFIG["model"],
                torch_dtype=torch.float16,
            )
            pipe.enable_attention_slicing()
            pipe = pipe.to("cuda")

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
            }}
            Path("/kaggle/working/image_report.json").write_text(
                json.dumps(report, indent=2) + "\\n",
                encoding="utf-8",
            )
            print("AI_AGENT_IMAGE_OK")
            print(json.dumps(report, indent=2))
            """
        ).strip() + "\n"
