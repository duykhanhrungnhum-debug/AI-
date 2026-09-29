"""Multimodal final visual-quality gate for generated media."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import base64
import json
import textwrap
import time

from PIL import Image

from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker


@dataclass(frozen=True)
class VisionQualityRequest:
    item_id: str
    image: bytes
    prompt: str
    profile: str
    rubric: tuple[str, ...]
    expected_subject_count: int | None = 1

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id is required")
        if not self.image:
            raise ValueError("image must not be empty")
        if len(self.image) > 12_000_000:
            raise ValueError("image exceeds 12 MB")
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if not self.profile.strip():
            raise ValueError("profile is required")
        if not self.rubric or any(not item.strip() for item in self.rubric):
            raise ValueError("rubric must contain non-empty criteria")
        if self.expected_subject_count is not None and self.expected_subject_count <= 0:
            raise ValueError("expected_subject_count must be positive")


@dataclass(frozen=True)
class VisionQualityResult:
    item_id: str
    passed: bool
    quality_score: float
    prompt_match_score: float
    subject_count: int | None
    major_issues: tuple[str, ...]
    minor_issues: tuple[str, ...]
    review_text: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class BatchVisionQualityResult:
    items: tuple[VisionQualityResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item.passed for item in self.items)

    @property
    def failed_item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items if not item.passed)


@dataclass
class KaggleVisionQualityVerifier:
    """Use a small open VLM as the final 'eyes-on-image' quality critic."""

    worker: KaggleGpuWorker
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    kernel_slug: str = "ai-agent-vision-quality"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    min_quality_score: float = 8.0
    min_prompt_match_score: float = 8.0
    provider: str = "kaggle-gpu-vlm-quality"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if not 0 <= self.min_quality_score <= 10:
            raise ValueError("min_quality_score must be in [0, 10]")
        if not 0 <= self.min_prompt_match_score <= 10:
            raise ValueError("min_prompt_match_score must be in [0, 10]")

    def verify(self, request: VisionQualityRequest) -> VisionQualityResult:
        return self.verify_many((request,)).items[0]

    def verify_many(
        self,
        requests: tuple[VisionQualityRequest, ...] | list[VisionQualityRequest],
    ) -> BatchVisionQualityResult:
        assert_core_invariants()
        requests = tuple(requests)
        if not requests:
            raise ValueError("at least one visual-quality request is required")
        ids = [item.item_id for item in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("item_id values must be unique")

        review_images = {
            request.item_id: self._compact_review_image(request.image)
            for request in requests
        }
        source = self._build_worker_source(requests, review_images=review_images)
        if len(source.encode("utf-8")) >= 900_000:
            raise ValueError("VLM worker source remains too large after image compaction")
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=self.kernel_slug.replace("-", " ").title(),
            source=source,
            enable_internet=True,
            is_private=True,
        )
        self._wait()

        report_bytes = self.worker.download_output_file(self.kernel_slug, "vision_quality.json")
        try:
            report = json.loads(report_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("vision quality report is invalid JSON") from exc
        if not isinstance(report, dict):
            raise ValueError("vision quality report must be an object")
        if report.get("model") != self.model:
            raise ValueError("vision quality report model does not match configured model")
        gpu_name = str(report.get("gpu_name") or "").strip()
        if not gpu_name:
            raise ValueError("vision quality report does not contain GPU evidence")

        entries = report.get("items")
        if not isinstance(entries, list) or len(entries) != len(requests):
            raise ValueError("vision quality report item count mismatch")
        by_id = {
            str(entry.get("item_id")): entry
            for entry in entries
            if isinstance(entry, dict)
        }

        results: list[VisionQualityResult] = []
        for request in requests:
            entry = by_id.get(request.item_id)
            if entry is None:
                raise ValueError(f"vision quality report missing item: {request.item_id}")
            if entry.get("image_sha256") != sha256(request.image).hexdigest():
                raise ValueError(f"vision quality image hash mismatch: {request.item_id}")
            review_image = review_images[request.item_id]
            if entry.get("review_image_sha256") != sha256(review_image).hexdigest():
                raise ValueError(f"vision quality review image hash mismatch: {request.item_id}")

            parsed = self._parse_review(str(entry.get("review_text") or ""))
            quality_score = self._score(parsed.get("quality_score"))
            prompt_match_score = self._score(parsed.get("prompt_match_score"))
            major_issues = self._string_tuple(parsed.get("major_issues"))
            minor_issues = self._string_tuple(parsed.get("minor_issues"))
            try:
                subject_count = int(parsed.get("subject_count"))
            except (TypeError, ValueError):
                subject_count = None

            vlm_pass = parsed.get("pass") is True
            count_pass = (
                request.expected_subject_count is None
                or subject_count == request.expected_subject_count
            )
            passed = (
                vlm_pass
                and quality_score >= self.min_quality_score
                and prompt_match_score >= self.min_prompt_match_score
                and not major_issues
                and count_pass
            )

            issues = list(major_issues)
            if quality_score < self.min_quality_score:
                issues.append(
                    f"VLM quality score below threshold: {quality_score:.2f} < {self.min_quality_score:.2f}"
                )
            if prompt_match_score < self.min_prompt_match_score:
                issues.append(
                    "VLM prompt match below threshold: "
                    f"{prompt_match_score:.2f} < {self.min_prompt_match_score:.2f}"
                )
            if not count_pass:
                issues.append(
                    f"VLM subject count mismatch: expected {request.expected_subject_count}, got {subject_count}"
                )
            if not vlm_pass and not issues:
                issues.append("VLM rejected image")

            review_text = str(entry.get("review_text") or "")
            results.append(VisionQualityResult(
                item_id=request.item_id,
                passed=passed,
                quality_score=quality_score,
                prompt_match_score=prompt_match_score,
                subject_count=subject_count,
                major_issues=tuple(dict.fromkeys(issues)),
                minor_issues=minor_issues,
                review_text=review_text,
                evidence=(
                    f"kaggle_kernel:{submission.ref}",
                    f"gpu:{gpu_name}",
                    f"vlm_model:{self.model}",
                    f"image_sha256:{sha256(request.image).hexdigest()}",
                    f"review_image_sha256:{sha256(review_image).hexdigest()}",
                    f"review_image_bytes:{len(review_image)}",
                    f"quality_score:{quality_score:.2f}",
                    f"prompt_match_score:{prompt_match_score:.2f}",
                    f"subject_count:{subject_count}",
                    f"profile:{request.profile}",
                ),
            ))

        return BatchVisionQualityResult(tuple(results))

    def _wait(self) -> None:
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
                    raise RuntimeError(f"Kaggle vision-quality worker failed: {detail}")
                return
            if self.poll_interval:
                time.sleep(self.poll_interval)
        raise TimeoutError("Timed out waiting for Kaggle vision-quality worker")

    @staticmethod
    def _score(value: object) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return -1.0
        return min(10.0, max(0.0, score))

    @staticmethod
    def _string_tuple(value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(str(item).strip() for item in value if str(item).strip())

    @staticmethod
    def _parse_review(text: str) -> dict:
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3:
                raw = "\n".join(lines[1:-1]).strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].lstrip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("VLM review does not contain a JSON object")
        try:
            parsed = json.loads(raw[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("VLM review JSON is malformed") from exc
        if not isinstance(parsed, dict):
            raise ValueError("VLM review JSON must be an object")
        return parsed

    @staticmethod
    def _compact_review_image(image: bytes) -> bytes:
        """Encode a visually faithful JPEG review copy small enough for Kaggle source limits."""
        try:
            with Image.open(BytesIO(image)) as opened:
                frame = opened.convert("RGB")
                frame.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                output = BytesIO()
                frame.save(
                    output,
                    format="JPEG",
                    quality=85,
                    optimize=True,
                    progressive=True,
                )
                compact = output.getvalue()
        except Exception as exc:
            raise ValueError("image cannot be compacted for VLM review") from exc
        if not compact:
            raise ValueError("VLM review image is empty after compaction")
        return compact

    def _build_worker_source(
        self,
        requests: tuple[VisionQualityRequest, ...],
        *,
        review_images: dict[str, bytes] | None = None,
    ) -> str:
        review_images = review_images or {
            item.item_id: self._compact_review_image(item.image)
            for item in requests
        }
        config = {
            "model": self.model,
            "items": [
                {
                    "item_id": item.item_id,
                    "image_b64": base64.b64encode(review_images[item.item_id]).decode("ascii"),
                    "image_sha256": sha256(item.image).hexdigest(),
                    "review_image_sha256": sha256(review_images[item.item_id]).hexdigest(),
                    "prompt": item.prompt,
                    "profile": item.profile,
                    "rubric": list(item.rubric),
                    "expected_subject_count": item.expected_subject_count,
                }
                for item in requests
            ],
        }
        config_json = json.dumps(config, ensure_ascii=False)
        return textwrap.dedent(
            f"""
            from __future__ import annotations

            import base64
            from hashlib import sha256
            from io import BytesIO
            import json
            import subprocess
            import sys
            from pathlib import Path

            CONFIG = json.loads({config_json!r})

            try:
                import torch
                from PIL import Image
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                from qwen_vl_utils import process_vision_info
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers>=4.49,<5", "accelerate<2", "safetensors",
                    "Pillow", "qwen-vl-utils",
                ])
                import torch
                from PIL import Image
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                from qwen_vl_utils import process_vision_info

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                CONFIG["model"],
                torch_dtype=torch.float16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(CONFIG["model"])
            model.eval()

            reports = []
            for item in CONFIG["items"]:
                image_bytes = base64.b64decode(item["image_b64"])
                if sha256(image_bytes).hexdigest() != item["review_image_sha256"]:
                    raise RuntimeError("vision-quality review image hash mismatch")
                image = Image.open(BytesIO(image_bytes)).convert("RGB")

                rubric = "\n".join(f"- {{criterion}}" for criterion in item["rubric"])
                expected = item.get("expected_subject_count")
                instruction = (
                    f"You are the final visual-quality inspector for a production image pipeline. "
                    f"Be strict and judge what is actually visible, not what the prompt intended. "
                    f"PROFILE: {{item['profile']}}; ORIGINAL PROMPT: {{item['prompt']}}; "
                    f"EXPECTED MAIN SUBJECT COUNT: {{expected}}; QUALITY RUBRIC: {{rubric}}. "
                    "Inspect specifically for wrong subject count, duplicated people/characters/objects, "
                    "malformed or fused hands/fingers/limbs, broken anatomy, distorted face/eyes/mouth, "
                    "warped geometry, impossible object connections, bad perspective, unreadable accidental "
                    "pseudo-text, muddy or unfinished details, identity drift when visually evident, and prompt mismatch. "
                    "Minor stylistic preferences are not major defects. A clean anatomically plausible stylized mascot "
                    "may have its normal species limbs. Score production readiness from 0 to 10 and prompt match from 0 to 10. "
                    "Set pass=true ONLY if quality_score >= 8, prompt_match_score >= 8, subject count is correct, and "
                    "major_issues is empty. Return JSON only with exactly these keys: "
                    '{{"pass": true, "quality_score": 0.0, "prompt_match_score": 0.0, "subject_count": 1, '
                    '"major_issues": [], "minor_issues": [], "summary": ""}}'
                )
                messages = [{{
                    "role": "user",
                    "content": [
                        {{"type": "image", "image": image}},
                        {{"type": "text", "text": instruction}},
                    ],
                }}]
                rendered = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[rendered],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=420,
                        do_sample=False,
                        repetition_penalty=1.04,
                    )
                generated_trimmed = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(inputs.input_ids, generated)
                ]
                review_text = processor.batch_decode(
                    generated_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                if not review_text:
                    raise RuntimeError("vision-quality model returned empty review")
                reports.append({{
                    "item_id": item["item_id"],
                    "image_sha256": item["image_sha256"],
                    "review_image_sha256": item["review_image_sha256"],
                    "review_text": review_text,
                }})

            result = {{
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "items": reports,
            }}
            Path("/kaggle/working/vision_quality.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print("AI_AGENT_VISION_QUALITY_OK")
            print(json.dumps({{
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "item_count": len(reports),
            }}, indent=2))
            """
        ).strip() + "\n"
