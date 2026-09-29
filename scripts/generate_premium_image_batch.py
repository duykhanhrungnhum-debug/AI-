#!/usr/bin/env python3
"""Generate multiple premium images while loading command/VLM models only once per batch."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import perf_counter

from ai_agent.core.image_model import ImageGenerationRequest
from ai_agent.core.kaggle_image import KaggleImageProvider
from ai_agent.core.kaggle_model import KaggleModelProvider
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_command import MEDIA_COMMAND_BRAIN_MODEL, MEDIA_COMMAND_FAST_MODEL, MediaCommandPlanner, benchmark_manifest
from ai_agent.core.vision_quality import HybridVisionQualityVerifier, VisionQualityRequest, result_from_inline_review


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return (cleaned or "item")[:32]


def main() -> int:
    raw = os.environ.get("MEDIA_COMMANDS_JSON", "").strip()
    if not raw:
        raise ValueError("MEDIA_COMMANDS_JSON is required")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("MEDIA_COMMANDS_JSON must be a non-empty JSON list")

    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        if isinstance(entry, str):
            item_id, command = f"item-{index + 1}", entry.strip()
        elif isinstance(entry, dict):
            item_id = str(entry.get("id") or f"item-{index + 1}").strip()
            command = str(entry.get("command") or "").strip()
        else:
            raise ValueError("each batch entry must be a string or object")
        if not item_id or not command:
            raise ValueError("each batch entry requires a non-empty id and command")
        if item_id in seen:
            raise ValueError(f"duplicate batch id: {item_id}")
        seen.add(item_id)
        items.append((item_id, command))

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token or not username:
        raise RuntimeError("KAGGLE_API_TOKEN and KAGGLE_USERNAME are required")

    output = Path(os.environ.get("MEDIA_OUTPUT_DIR", "premium-image-batch-output"))
    output.mkdir(parents=True, exist_ok=True)

    worker = KaggleGpuWorker(
        api_token=token,
        username=username,
        timeout=120,
        submission_retry_attempts=5,
        submission_retry_delay_seconds=30,
    )
    fast_model_name = os.environ.get("MEDIA_COMMAND_FAST_MODEL", MEDIA_COMMAND_FAST_MODEL)
    final_model_name = os.environ.get("MEDIA_COMMAND_MODEL", MEDIA_COMMAND_BRAIN_MODEL)
    use_inline_planner = os.environ.get("MEDIA_INLINE_PLANNER", "1") != "0"
    timings: dict[str, float | dict[str, float]] = {
        "image_generation": {},
        "image_worker": {},
        "inline_planner_worker": {},
        "inline_vlm_worker": {},
    }
    planning_started = perf_counter()
    planner_model_used = "inline-per-item" if use_inline_planner else fast_model_name
    planner_fallback_used = False
    if use_inline_planner:
        plans = MediaCommandPlanner().plan_many(
            tuple(command for _, command in items)
        )
    else:
        fast_language_model = KaggleModelProvider(
            worker=worker,
            model=fast_model_name,
            kernel_slug="ai-agent-media-command-fast",
            poll_interval=3,
            max_poll_attempts=120,
            max_new_tokens=140,
            temperature=0.0,
            enable_thinking=False,
        )
        try:
            plans = MediaCommandPlanner(fast_language_model).plan_many(
                tuple(command for _, command in items)
            )
        except Exception as fast_exc:
            planner_fallback_used = True
            final_language_model = KaggleModelProvider(
                worker=worker,
                model=final_model_name,
                kernel_slug="ai-agent-media-command-final",
                poll_interval=3,
                max_poll_attempts=120,
                max_new_tokens=180,
                temperature=0.0,
                enable_thinking=False,
            )
            try:
                plans = MediaCommandPlanner(final_language_model).plan_many(
                    tuple(command for _, command in items)
                )
                planner_model_used = final_model_name
            except Exception as final_exc:
                (output / "planning-failure.json").write_text(
                    json.dumps({
                        "fast_error": str(fast_exc),
                        "final_error": str(final_exc),
                    }, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise
    timings["planning"] = round(perf_counter() - planning_started, 3)
    planner_models_used: dict[str, str] = {}

    generated: dict[str, tuple[object, object, dict]] = {}
    inline_reviews: dict[str, tuple[VisionQualityRequest, object]] = {}
    statuses: dict[str, dict] = {}

    for (item_id, command), plan in zip(items, plans, strict=True):
        item_dir = output / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        if plan.mode != "image":
            statuses[item_id] = {"verified": False, "stage": "planning", "error": "resolved to video"}
            continue

        config = plan.model_config
        seed = int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")
        plan_data = {
            "command": command,
            "mode": plan.mode,
            "profile": plan.profile.name,
            "model": config.model,
            "inference_steps": config.inference_steps,
            "guidance_scale": config.guidance_scale,
            "scheduler": config.scheduler,
            "compiler_output": plan.compiler_output,
            "prompt": plan.prompt,
            "negative_prompt": plan.negative_prompt,
            "width": plan.profile.width,
            "height": plan.profile.height,
            "seed": seed,
        }
        (item_dir / "plan.json").write_text(
            json.dumps(plan_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        provider = KaggleImageProvider(
            worker=worker,
            model=config.model,
            model_variant="fp16" if "RealVisXL" in config.model else None,
            kernel_slug=f"ai-agent-image-{_slug(item_id)}",
            poll_interval=3,
            max_poll_attempts=180,
            inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
            scheduler=config.scheduler,
            prompt_alignment_threshold=float(os.environ.get("MEDIA_PROMPT_ALIGNMENT_THRESHOLD", "0.22")),
            visual_quality_margin_threshold=float(os.environ.get("MEDIA_VISUAL_QUALITY_MARGIN_THRESHOLD", "0.015")),
            enforce_visual_quality_margin=False,
            enable_clip_precheck=os.environ.get("MEDIA_CLIP_PRECHECK", "0") == "1",
            enable_cpu_offload=os.environ.get("MEDIA_CPU_OFFLOAD", "0") == "1",
            enable_inline_planner=use_inline_planner,
            inline_planner_fast_model=fast_model_name,
            inline_planner_final_model=final_model_name,
            inline_planner_raw_command=command if use_inline_planner else "",
            inline_planner_positive_constraints=plan.profile.positive_constraints if use_inline_planner else "",
            quality_good_text=config.quality_good_text,
            quality_bad_texts=config.quality_bad_texts,
            enable_inline_vlm=True,
            inline_vlm_model="Qwen/Qwen3-VL-2B-Instruct",
            inline_vlm_profile=plan.profile.name,
            inline_vlm_rubric=tuple(benchmark_manifest()["profiles"][plan.profile.name]["must_pass"]),
            inline_vlm_expected_subject_count=1,
            inline_vlm_min_score=9.0,
        )
        request = ImageGenerationRequest(
            prompt=str(plan_data.get("prompt") or plan.prompt),
            negative_prompt=plan.negative_prompt,
            width=plan.profile.width,
            height=plan.profile.height,
            seed=seed,
        )
        generation_started = perf_counter()
        try:
            artifact = provider.generate(request)
        except Exception as exc:
            timings["image_generation"][item_id] = round(perf_counter() - generation_started, 3)
            statuses[item_id] = {"verified": False, "stage": "image_generation", "error": str(exc)}
            try:
                candidate = worker.download_output_file(provider.kernel_slug, "generated.png")
                (item_dir / "candidate.png").write_bytes(candidate)
            except Exception:
                pass
            continue
        timings["image_generation"][item_id] = round(perf_counter() - generation_started, 3)
        report_timings = provider.last_report.get("timings")
        if isinstance(report_timings, dict):
            timings["image_worker"][item_id] = report_timings
        inline_planner_report = provider.last_report.get("inline_planner")
        if isinstance(inline_planner_report, dict):
            planner_models_used[item_id] = str(inline_planner_report.get("model") or "")
            if isinstance(inline_planner_report.get("timings"), dict):
                timings["inline_planner_worker"][item_id] = {
                    **inline_planner_report["timings"],
                    "total_stage_seconds": inline_planner_report.get("total_stage_seconds"),
                    "fallback_used": bool(inline_planner_report.get("fallback_used")),
                }
            actual_prompt = str(provider.last_report.get("compiled_prompt") or "").strip()
            if actual_prompt:
                plan_data["compiler_output"] = str(inline_planner_report.get("raw_output") or "")
                plan_data["prompt"] = actual_prompt
                (item_dir / "plan.json").write_text(
                    json.dumps(plan_data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        inline_report = provider.last_report.get("inline_vlm")
        if isinstance(inline_report, dict) and isinstance(inline_report.get("timings"), dict):
            timings["inline_vlm_worker"][item_id] = inline_report["timings"]

        (item_dir / "candidate.png").write_bytes(artifact.data)
        benchmark = benchmark_manifest()["profiles"][plan.profile.name]
        digest = sha256(artifact.data).hexdigest()
        if digest in set(benchmark.get("known_rejected_sha256", ())):
            statuses[item_id] = {
                "verified": False,
                "stage": "user_benchmark",
                "error": "exact image was previously rejected by the user",
            }
            continue

        generated[item_id] = (plan, artifact, plan_data)
        inline_payload = provider.last_report.get("inline_vlm")
        if not isinstance(inline_payload, dict):
            statuses[item_id] = {
                "verified": False,
                "stage": "inline_vlm_quality",
                "error": "inline VLM report missing",
            }
            continue
        inline_text = str(inline_payload.get("review_text") or "").strip()
        inline_model = str(inline_payload.get("model") or "").strip()
        if not inline_text or inline_model != "Qwen/Qwen3-VL-2B-Instruct":
            statuses[item_id] = {
                "verified": False,
                "stage": "inline_vlm_quality",
                "error": "inline VLM report invalid",
            }
            continue
        review_request = VisionQualityRequest(
            item_id=item_id,
            image=artifact.data,
            prompt=str(plan_data.get("prompt") or plan.prompt),
            profile=plan.profile.name,
            rubric=tuple(benchmark["must_pass"]),
            expected_subject_count=1,
        )
        inline_evidence = (
            f"kaggle_kernel:{provider.kernel_slug}",
            f"gpu:{provider.last_report.get('gpu_name')}",
            "hybrid_fast_model:Qwen/Qwen3-VL-2B-Instruct",
            "hybrid_final_model:Qwen/Qwen3-VL-8B-Instruct",
            "hybrid_clear_pass_score:9.00",
            "hybrid_fast_reject_score:8.00",
        )
        inline_reviews[item_id] = (
            review_request,
            result_from_inline_review(
                review_request,
                inline_text,
                model=inline_model,
                evidence=inline_evidence,
                min_quality_score=9.0,
                min_prompt_match_score=9.0,
            ),
        )

    reviews = {}
    fallback_requests = []
    verifier = HybridVisionQualityVerifier(
        worker=worker,
        kernel_slug="ai-agent-premium-image-vlm-batch",
        poll_interval=3,
        max_poll_attempts=120,
        min_quality_score=9.0,
        min_prompt_match_score=9.0,
        clear_pass_score=9.0,
        fast_reject_score=8.0,
    )
    for item_id, (request, inline_result) in inline_reviews.items():
        if verifier._clear_fast_pass(inline_result):
            reviews[item_id] = inline_result
            continue
        if verifier._obvious_fast_reject(request, inline_result):
            reviews[item_id] = inline_result
            continue
        fallback_requests.append(request)

    if fallback_requests:
        vlm_started = perf_counter()
        final_verifier = verifier._make_verifier(
            model=verifier.final_model,
            suffix="final",
        )
        final_batch = final_verifier.verify_many(tuple(fallback_requests))
        timings["vlm_quality_fallback"] = round(perf_counter() - vlm_started, 3)
        for item in final_batch.items:
            reviews[item.item_id] = item

    for item_id, (plan, artifact, plan_data) in generated.items():
        item_dir = output / item_id
        review = reviews.get(item_id)
        if review is None:
            if item_id not in statuses:
                statuses[item_id] = {
                    "verified": False,
                    "stage": "vlm_quality",
                    "error": "visual quality result missing",
                }
            continue
        review_payload = {
            "passed": review.passed,
            "quality_score": review.quality_score,
            "prompt_match_score": review.prompt_match_score,
            "structure_score": review.structure_score,
            "detail_score": review.detail_score,
            "aesthetic_score": review.aesthetic_score,
            "composition_score": review.composition_score,
            "benchmark_match_score": review.benchmark_match_score,
            "subject_count": review.subject_count,
            "major_issues": list(review.major_issues),
            "minor_issues": list(review.minor_issues),
            "review_text": review.review_text,
            "evidence": list(review.evidence),
        }
        (item_dir / "vlm-quality.json").write_text(
            json.dumps(review_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if review.passed:
            (item_dir / "image.png").write_bytes(artifact.data)
            statuses[item_id] = {
                "verified": True,
                "stage": "complete",
                "image_sha256": sha256(artifact.data).hexdigest(),
            }
        else:
            statuses[item_id] = {
                "verified": False,
                "stage": "vlm_quality",
                "error": "; ".join(review.major_issues or ("quality gate failed",)),
            }

    summary = {
        "batch_size": len(items),
        "verified_count": sum(1 for item in statuses.values() if item.get("verified")),
        "failed_count": sum(1 for item in statuses.values() if not item.get("verified")),
        "brain_model": planner_model_used,
        "planner_fast_model": fast_model_name,
        "planner_final_model": final_model_name,
        "planner_fallback_used": planner_fallback_used,
        "inline_planner": use_inline_planner,
        "planner_models_used": planner_models_used,
        "clip_precheck": os.environ.get("MEDIA_CLIP_PRECHECK", "0") == "1",
        "cpu_offload": os.environ.get("MEDIA_CPU_OFFLOAD", "0") == "1",
        "timing_seconds": timings,
        "statuses": statuses,
    }
    (output / "batch-status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PREMIUM_IMAGE_BATCH_DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
