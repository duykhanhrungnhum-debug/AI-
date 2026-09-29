"""Multimodal final visual-quality gate for generated media."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import base64
import json
import textwrap
import time

from PIL import Image

from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker
from .media_command import benchmark_manifest


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
    structure_score: float
    detail_score: float
    aesthetic_score: float
    composition_score: float
    benchmark_match_score: float
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
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    kernel_slug: str = "ai-agent-vision-quality"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    min_quality_score: float = 8.0
    min_prompt_match_score: float = 8.0
    attention_backend: str = "sdpa"
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
        if self.attention_backend not in {"sdpa", "eager"}:
            raise ValueError("attention_backend must be sdpa or eager")

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
        report_attention_backend = str(report.get("attention_backend") or "").strip()
        if report_attention_backend != self.attention_backend:
            raise ValueError("vision quality report attention backend mismatch")
        timings = report.get("timings") if isinstance(report.get("timings"), dict) else {}

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
            defect_parsed = self._parse_review(str(entry.get("defect_review_text") or ""))
            defect_issues = tuple(dict.fromkeys((
                *self._string_tuple(defect_parsed.get("critical_defects")),
                *self._string_tuple(defect_parsed.get("benchmark_failures")),
                *self._string_tuple(defect_parsed.get("uncertain_regions")),
            )))
            defect_reject = defect_parsed.get("reject") is True or bool(defect_issues)
            profile_config = benchmark_manifest().get("profiles", {}).get(request.profile, {})
            user_rejected = (
                sha256(request.image).hexdigest()
                in set(profile_config.get("known_rejected_sha256", ()))
            )
            quality_score = self._score(parsed.get("quality_score"))
            prompt_match_score = self._score(parsed.get("prompt_match_score"))
            structure_score = self._score(parsed.get("structure_score"))
            detail_score = self._score(parsed.get("detail_score"))
            aesthetic_score = self._score(parsed.get("aesthetic_score"))
            composition_score = self._score(parsed.get("composition_score"))
            benchmark_match_score = self._score(parsed.get("benchmark_match_score"))
            major_issues = tuple(dict.fromkeys((
                *self._string_tuple(parsed.get("major_issues")),
                *(f"defect hunter: {issue}" for issue in defect_issues),
            )))
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
            component_scores = (
                structure_score,
                detail_score,
                aesthetic_score,
                composition_score,
                benchmark_match_score,
            )
            passed = (
                vlm_pass
                and quality_score >= self.min_quality_score
                and prompt_match_score >= self.min_prompt_match_score
                and all(score >= self.min_quality_score for score in component_scores)
                and not defect_reject
                and not user_rejected
                and not major_issues
                and count_pass
            )

            issues = list(major_issues)
            if user_rejected:
                issues.append("image exactly matches a user-rejected quality example")
            if quality_score < self.min_quality_score:
                issues.append(
                    f"VLM quality score below threshold: {quality_score:.2f} < {self.min_quality_score:.2f}"
                )
            if prompt_match_score < self.min_prompt_match_score:
                issues.append(
                    "VLM prompt match below threshold: "
                    f"{prompt_match_score:.2f} < {self.min_prompt_match_score:.2f}"
                )
            for label, score in (
                ("structure", structure_score),
                ("detail", detail_score),
                ("aesthetic", aesthetic_score),
                ("composition", composition_score),
                ("benchmark_match", benchmark_match_score),
            ):
                if score < self.min_quality_score:
                    issues.append(
                        f"VLM {label} score below threshold: {score:.2f} < {self.min_quality_score:.2f}"
                    )
            if not count_pass:
                issues.append(
                    f"VLM subject count mismatch: expected {request.expected_subject_count}, got {subject_count}"
                )
            if not vlm_pass and not issues:
                issues.append("VLM rejected image")

            review_text = (
                "DEFECT_HUNTER:\n" + str(entry.get("defect_review_text") or "")
                + "\nSCORER:\n" + str(entry.get("review_text") or "")
            )
            results.append(VisionQualityResult(
                item_id=request.item_id,
                passed=passed,
                quality_score=quality_score,
                prompt_match_score=prompt_match_score,
                structure_score=structure_score,
                detail_score=detail_score,
                aesthetic_score=aesthetic_score,
                composition_score=composition_score,
                benchmark_match_score=benchmark_match_score,
                subject_count=subject_count,
                major_issues=tuple(dict.fromkeys(issues)),
                minor_issues=minor_issues,
                review_text=review_text,
                evidence=(
                    f"kaggle_kernel:{submission.ref}",
                    f"gpu:{gpu_name}",
                    f"vlm_model:{self.model}",
                    f"attention_backend:{self.attention_backend}",
                    f"qa_dependency_seconds:{float(timings.get('dependency_seconds', -1.0)):.3f}",
                    f"qa_model_load_seconds:{float(timings.get('model_load_seconds', -1.0)):.3f}",
                    f"qa_review_seconds:{float(timings.get('review_seconds', -1.0)):.3f}",
                    f"qa_worker_total_seconds:{float(timings.get('worker_total_seconds', -1.0)):.3f}",
                    f"image_sha256:{sha256(request.image).hexdigest()}",
                    f"review_image_sha256:{sha256(review_image).hexdigest()}",
                    f"review_image_bytes:{len(review_image)}",
                    f"quality_score:{quality_score:.2f}",
                    f"prompt_match_score:{prompt_match_score:.2f}",
                    f"structure_score:{structure_score:.2f}",
                    f"detail_score:{detail_score:.2f}",
                    f"aesthetic_score:{aesthetic_score:.2f}",
                    f"composition_score:{composition_score:.2f}",
                    f"benchmark_match_score:{benchmark_match_score:.2f}",
                    f"defect_hunter_reject:{defect_reject}",
                    f"defect_hunter_issue_count:{len(defect_issues)}",
                    f"user_rejected_feedback:{user_rejected}",
                    "review_passes:1",
                    "review_views:3",
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

        # Fast worker output may use compact keys to reduce generation latency.
        if "p" in parsed or "q" in parsed:
            parsed = {
                "pass": parsed.get("p") is True,
                "quality_score": parsed.get("q"),
                "prompt_match_score": parsed.get("m"),
                "structure_score": parsed.get("s"),
                "detail_score": parsed.get("d"),
                "aesthetic_score": parsed.get("a"),
                "composition_score": parsed.get("c"),
                "benchmark_match_score": parsed.get("b"),
                "subject_count": parsed.get("n"),
                "major_issues": [],
                "critical_defects": parsed.get("x") or [],
                "benchmark_failures": parsed.get("f") or [],
                "uncertain_regions": parsed.get("u") or [],
                "minor_issues": parsed.get("i") or [],
                "summary": "",
            }
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
            "attention_backend": self.attention_backend,
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
        lines = [
            "from __future__ import annotations",
            "",
            "import base64",
            "from hashlib import sha256",
            "from io import BytesIO",
            "import json",
            "import subprocess",
            "import sys",
            "import time",
            "from pathlib import Path",
            "",
            f"CONFIG = json.loads({config_json!r})",
            "worker_started = time.perf_counter()",
            "",
            "try:",
            "    import bitsandbytes",
            "    import torch",
            "    from PIL import Image",
            "    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration",
            "except ImportError:",
            "    subprocess.check_call([",
            '        sys.executable, "-m", "pip", "install", "--quiet",',
            '        "transformers>=4.57,<5", "accelerate<2", "safetensors",',
            '        "bitsandbytes>=0.46.1", "Pillow",',
            "    ])",
            "    import bitsandbytes",
            "    import torch",
            "    from PIL import Image",
            "    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration",
            "",
            "if not torch.cuda.is_available():",
            '    raise RuntimeError("CUDA GPU is not available")',
            "",
            "dependencies_ready = time.perf_counter()",
            "gpu_name = torch.cuda.get_device_name(0)",
            "quantization = BitsAndBytesConfig(",
            "    load_in_4bit=True,",
            '    bnb_4bit_quant_type="nf4",',
            "    bnb_4bit_use_double_quant=True,",
            "    bnb_4bit_compute_dtype=torch.float16,",
            ")",
            "model_load_started = time.perf_counter()",
            "model = Qwen3VLForConditionalGeneration.from_pretrained(",
            '    CONFIG["model"],',
            "    quantization_config=quantization,",
            '    device_map="auto",',
            "    low_cpu_mem_usage=True,",
            '    attn_implementation=CONFIG["attention_backend"],',
            ")",
            'processor = AutoProcessor.from_pretrained(CONFIG["model"])',
            "model.eval()",
            "model_ready = time.perf_counter()",
            "",
            "reports = []",
            "review_seconds = 0.0",
            'for item in CONFIG["items"]:',
            '    image_bytes = base64.b64decode(item["image_b64"])',
            '    if sha256(image_bytes).hexdigest() != item["review_image_sha256"]:',
            '        raise RuntimeError("vision-quality review image hash mismatch")',
            '    image = Image.open(BytesIO(image_bytes)).convert("RGB")',
            '    width, height = image.size',
            '    views = [',
            '        image,',
            '        image.crop((0, 0, width, max(1, height // 2))),',
            '        image.crop((0, height // 2, width, height)),',
            '    ]',
            "",
            '    rubric = "\\n".join(f"- {criterion}" for criterion in item["rubric"])',
            '    expected = item.get("expected_subject_count")',
            "    instruction = (",
            '        f"You are the final visual-quality inspector for a production image pipeline. "',
            '        f"Be strict and judge what is actually visible, not what the prompt intended. "',
            '        f"PROFILE: {item[\'profile\']}; ORIGINAL PROMPT: {item[\'prompt\']}; "',
            '        f"EXPECTED MAIN SUBJECT COUNT: {expected}; QUALITY RUBRIC: {rubric}. "',
            '        "You receive three views of the SAME generated image: full frame, upper crop, lower crop. "',
            '        "Inspect specifically for wrong subject count and duplicated people/characters/objects before scoring local quality. "',
            '        "Use the crops as forensic zooms, never count them as extra subjects. Inspect malformed or fused hands/fingers/limbs, visible anatomy/appendages, face/eyes/mouth, "',
            '        "object connections, silhouette, materials, lighting, perspective, texture continuity and accidental pseudo-text. "',
            '        "For human_photo_premium, reject any fused/missing/extra fingers, unnatural wrists/elbows/shoulders, distorted face, "',
            '        "plastic or melted clothing/skin, warped architecture or props, or obvious AI artifacts. "',
            '        "For animal_photo_premium, camera naturalism is a hard requirement: reject HDR-like tone mapping, excessive sharpening or microcontrast, "',
            '        "oversaturated color, airbrushed or uniformly perfect fur, artificial glow, fake bokeh, advertising-style retouch, synthetic polish, or any image that looks Photoshopped rather than naturally camera-captured. Natural camera softness and small imperfections are desirable. "',
            '        "For mascot_premium, reject crude or generic 3D, malformed/duplicated claws or legs, melted shell/body, inconsistent eyes, "',
            '        "weak expression, muddy materials, flat lighting, dirty silhouette/cutout edges, or anything below polished studio asset quality. "',
            '        "Treat QUALITY RUBRIC as the user-approved benchmark target, not optional advice. Generic technical beauty is insufficient if benchmark shape/material/style differs. "',
            '        "Score 0-10: quality_score overall production readiness; prompt_match_score; structure_score anatomy/geometry; "',
            '        "detail_score local materials/textures/edges; aesthetic_score polish/lighting/expression; composition_score framing/background; "',
            '        "benchmark_match_score how closely the output reaches the specific user benchmark visual language described in QUALITY RUBRIC. "',
            '        "Set pass=true ONLY if every score including benchmark_match_score is >= 8, subject count is correct, and all defect/benchmark/uncertainty lists are empty. "',
            '        "Act as scorer and defect hunter in this single pass. Put blocking visible defects in critical_defects, benchmark misses in benchmark_failures, and any critical region you cannot verify in uncertain_regions. "',
            '        "If a critical region is uncertain, pass MUST be false. Never average a local defect away with a high overall score. "',
            '        "Return ONLY one compact JSON object, with no prose before or after it. Use short issue phrases. "',
            '        "Keys: p=pass, q=quality, m=prompt_match, s=structure, d=detail, a=aesthetic, c=composition, b=benchmark_match, "',
            '        "n=subject_count, x=critical_defects, f=benchmark_failures, u=uncertain_regions, i=minor_issues. "',
            '        \'Example schema: {"p":true,"q":9,"m":9,"s":9,"d":9,"a":9,"c":9,"b":9,"n":1,"x":[],"f":[],"u":[],"i":[]}\'',
            "    )",
            "    def run_review(text_instruction):",
            "        content = [{\"type\": \"image\", \"image\": view} for view in views]",
            '        content.append({"type": "text", "text": text_instruction})',
            '        messages = [{"role": "user", "content": content}]',
            "        inputs = processor.apply_chat_template(",
            "            messages,",
            "            tokenize=True,",
            "            add_generation_prompt=True,",
            "            return_dict=True,",
            '            return_tensors="pt",',
            "        ).to(model.device)",
            "        with torch.inference_mode():",
            "            generated = model.generate(",
            "                **inputs,",
            "                max_new_tokens=200,",
            "                do_sample=False,",
            "                use_cache=True,",
            "                repetition_penalty=1.03,",
            "            )",
            "        generated_trimmed = [",
            "            output_ids[len(input_ids):]",
            "            for input_ids, output_ids in zip(inputs.input_ids, generated)",
            "        ]",
            "        text = processor.batch_decode(",
            "            generated_trimmed,",
            "            skip_special_tokens=True,",
            "            clean_up_tokenization_spaces=False,",
            "        )[0].strip()",
            "        if not text:",
            '            raise RuntimeError("vision-quality model returned empty review")',
            "        return text",
            "",
            "    review_started = time.perf_counter()",
            "    review_text = run_review(instruction)",
            "    review_seconds += time.perf_counter() - review_started",
            "    defect_review_text = review_text",
            "    reports.append({",
            '        "item_id": item["item_id"],',
            '        "image_sha256": item["image_sha256"],',
            '        "review_image_sha256": item["review_image_sha256"],',
            '        "defect_review_text": defect_review_text,',
            '        "review_text": review_text,',
            "    })",
            "",
            "worker_finished = time.perf_counter()",
            "result = {",
            '    "model": CONFIG["model"],',
            '    "gpu_name": gpu_name,',
            '    "attention_backend": CONFIG["attention_backend"],',
            '    "items": reports,',
            '    "timings": {',
            '        "dependency_seconds": round(dependencies_ready - worker_started, 3),',
            '        "model_load_seconds": round(model_ready - model_load_started, 3),',
            '        "review_seconds": round(review_seconds, 3),',
            '        "worker_total_seconds": round(worker_finished - worker_started, 3),',
            "    },",
            "}",
            'Path("/kaggle/working/vision_quality.json").write_text(',
            '    json.dumps(result, ensure_ascii=False, indent=2) + "\\n",',
            '    encoding="utf-8",',
            ")",
            'print("AI_AGENT_VISION_QUALITY_OK")',
            "print(json.dumps({",
            '    "model": CONFIG["model"],',
            '    "gpu_name": gpu_name,',
            '    "item_count": len(reports),',
            "}, indent=2))",
        ]
        return "\n".join(lines) + "\n"


@dataclass
class HybridVisionQualityVerifier:
    """Use a smaller Qwen3-VL fast gate and fall back to 8B only when needed."""

    worker: KaggleGpuWorker
    fast_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    final_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    kernel_slug: str = "ai-agent-hybrid-vision-quality"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    min_quality_score: float = 9.0
    min_prompt_match_score: float = 9.0
    clear_pass_score: float = 9.2
    attention_backend: str = "sdpa"
    provider: str = "kaggle-gpu-hybrid-vlm-quality"

    def __post_init__(self) -> None:
        if not self.fast_model.strip() or not self.final_model.strip():
            raise ValueError("fast_model and final_model are required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if not 0 <= self.min_quality_score <= 10:
            raise ValueError("min_quality_score must be in [0, 10]")
        if not 0 <= self.min_prompt_match_score <= 10:
            raise ValueError("min_prompt_match_score must be in [0, 10]")
        if not max(self.min_quality_score, self.min_prompt_match_score) <= self.clear_pass_score <= 10:
            raise ValueError("clear_pass_score must be >= configured minimum scores and <= 10")
        if self.attention_backend not in {"sdpa", "eager"}:
            raise ValueError("attention_backend must be sdpa or eager")

    def _make_verifier(self, *, model: str, suffix: str) -> KaggleVisionQualityVerifier:
        return KaggleVisionQualityVerifier(
            worker=self.worker,
            model=model,
            kernel_slug=f"{self.kernel_slug}-{suffix}",
            poll_interval=self.poll_interval,
            max_poll_attempts=self.max_poll_attempts,
            min_quality_score=self.min_quality_score,
            min_prompt_match_score=self.min_prompt_match_score,
            attention_backend=self.attention_backend,
        )

    def _clear_fast_pass(self, result: VisionQualityResult) -> bool:
        scores = (
            result.quality_score,
            result.prompt_match_score,
            result.structure_score,
            result.detail_score,
            result.aesthetic_score,
            result.composition_score,
            result.benchmark_match_score,
        )
        return (
            result.passed
            and not result.major_issues
            and min(scores) >= self.clear_pass_score
        )

    def verify(self, request: VisionQualityRequest) -> VisionQualityResult:
        return self.verify_many((request,)).items[0]

    def verify_many(
        self,
        requests: tuple[VisionQualityRequest, ...] | list[VisionQualityRequest],
    ) -> BatchVisionQualityResult:
        requests = tuple(requests)
        if not requests:
            raise ValueError("at least one visual-quality request is required")

        fast_batch = self._make_verifier(model=self.fast_model, suffix="fast").verify_many(requests)
        fallback_requests = tuple(
            request
            for request, fast_result in zip(requests, fast_batch.items, strict=True)
            if not self._clear_fast_pass(fast_result)
        )

        final_by_id: dict[str, VisionQualityResult] = {}
        if fallback_requests:
            final_batch = self._make_verifier(model=self.final_model, suffix="final").verify_many(
                fallback_requests
            )
            final_by_id = {item.item_id: item for item in final_batch.items}

        results: list[VisionQualityResult] = []
        for request, fast_result in zip(requests, fast_batch.items, strict=True):
            if self._clear_fast_pass(fast_result):
                results.append(replace(
                    fast_result,
                    evidence=fast_result.evidence + (
                        "hybrid_stage:fast_clear_pass",
                        f"hybrid_fast_model:{self.fast_model}",
                        f"hybrid_final_model:{self.final_model}",
                        f"hybrid_clear_pass_score:{self.clear_pass_score:.2f}",
                    ),
                ))
                continue

            final_result = final_by_id[request.item_id]
            results.append(replace(
                final_result,
                evidence=final_result.evidence + (
                    "hybrid_stage:final_fallback",
                    f"hybrid_fast_model:{self.fast_model}",
                    f"hybrid_final_model:{self.final_model}",
                    f"hybrid_fast_quality_score:{fast_result.quality_score:.2f}",
                    f"hybrid_fast_structure_score:{fast_result.structure_score:.2f}",
                    f"hybrid_clear_pass_score:{self.clear_pass_score:.2f}",
                ),
            ))

        return BatchVisionQualityResult(tuple(results))
