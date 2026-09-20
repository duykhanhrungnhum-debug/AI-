"""Open-model LLM execution on an on-demand Kaggle GPU worker."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import textwrap
import time

from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker
from .model import ModelResponse


@dataclass(frozen=True)
class KaggleModelBatchResult:
    responses: tuple[ModelResponse, ...]
    evidence: tuple[str, ...]


@dataclass
class KaggleModelProvider:
    """Run an open-weight instruction model on Kaggle GPU.

    The model is downloaded as weights and inference happens inside the Kaggle
    worker. No external hosted inference API is used.
    """

    worker: KaggleGpuWorker
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    kernel_slug: str = "ai-agent-llm-worker"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    max_new_tokens: int = 512
    temperature: float = 0.0
    provider: str = "kaggle-gpu-open-model"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")

    def generate(self, prompt: str) -> ModelResponse:
        result = self.generate_many([prompt])
        return result.responses[0]

    def generate_many(self, prompts: list[str] | tuple[str, ...]) -> KaggleModelBatchResult:
        assert_core_invariants()
        prompts = tuple(prompt.strip() for prompt in prompts)
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("prompts must contain non-empty text")

        source = self._build_worker_source(prompts)
        title = self.kernel_slug.replace("-", " ").title()
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=title,
            source=source,
            enable_internet=True,
            is_private=True,
        )
        self._wait()

        report_bytes = self.worker.download_output_file(self.kernel_slug, "responses.json")
        try:
            report = json.loads(report_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Kaggle model report is invalid JSON") from exc
        if not isinstance(report, dict):
            raise ValueError("Kaggle model report must be an object")
        if report.get("model") != self.model:
            raise ValueError("Kaggle model report has unexpected model")
        gpu_name = str(report.get("gpu_name", "")).strip()
        if not gpu_name:
            raise ValueError("Kaggle model report does not contain GPU evidence")

        items = report.get("responses")
        if not isinstance(items, list) or len(items) != len(prompts):
            raise ValueError("Kaggle model response count does not match request")

        responses: list[ModelResponse] = []
        evidence: list[str] = [
            f"kaggle_kernel:{submission.ref}",
            f"gpu:{gpu_name}",
            f"model:{self.model}",
            f"response_count:{len(items)}",
        ]
        for index, (prompt, item) in enumerate(zip(prompts, items, strict=True)):
            if not isinstance(item, dict):
                raise ValueError("Kaggle model response entry must be an object")
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Kaggle model response {index} is empty")

            expected_prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()
            actual_response_hash = sha256(text.encode("utf-8")).hexdigest()
            if item.get("prompt_sha256") != expected_prompt_hash:
                raise ValueError(f"Kaggle model prompt hash mismatch at index {index}")
            if item.get("response_sha256") != actual_response_hash:
                raise ValueError(f"Kaggle model response hash mismatch at index {index}")

            evidence.extend((
                f"prompt_{index}_sha256:{expected_prompt_hash}",
                f"response_{index}_sha256:{actual_response_hash}",
            ))
            responses.append(ModelResponse(text=text, provider=self.provider, model=self.model))

        return KaggleModelBatchResult(tuple(responses), tuple(evidence))

    def _wait(self) -> None:
        for _ in range(self.max_poll_attempts):
            status = self.worker.status(self.kernel_slug)
            if status.terminal:
                if not status.successful:
                    raise RuntimeError(
                        f"Kaggle model worker failed: {status.status} {status.failure_message}"
                    )
                return
            if self.poll_interval:
                time.sleep(self.poll_interval)
        raise TimeoutError("Timed out waiting for Kaggle model worker")

    def _build_worker_source(self, prompts: tuple[str, ...]) -> str:
        config = {
            "model": self.model,
            "prompts": prompts,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
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
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers<5", "accelerate<2", "safetensors", "sentencepiece",
                ])
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model"],
                torch_dtype=torch.float16,
                device_map="auto",
            )

            model.eval()

            responses = []
            for prompt in CONFIG["prompts"]:
                messages = [{{"role": "user", "content": prompt}}]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
                generate_kwargs = {{
                    "max_new_tokens": int(CONFIG["max_new_tokens"]),
                    "do_sample": float(CONFIG["temperature"]) > 0,
                    "repetition_penalty": 1.08,
                    "no_repeat_ngram_size": 3,
                }}
                if generate_kwargs["do_sample"]:
                    generate_kwargs["temperature"] = float(CONFIG["temperature"])

                generated = model.generate(**model_inputs, **generate_kwargs)
                new_tokens = generated[:, model_inputs.input_ids.shape[1]:]
                text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                if not text:
                    raise RuntimeError("model returned empty text")

                responses.append({{
                    "text": text,
                    "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                    "response_sha256": sha256(text.encode("utf-8")).hexdigest(),
                }})

            report = {{
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "responses": responses,
            }}
            Path("/kaggle/working/responses.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            print("AI_AGENT_LOCAL_LLM_OK")
            print(json.dumps({{
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "response_count": len(responses),
            }}, indent=2))
            """
        ).strip() + "\n"
