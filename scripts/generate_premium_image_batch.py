#!/usr/bin/env python3
"""Generate multiple premium images in one semantically routed Kaggle GPU worker."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from ai_agent.core.kaggle_image_batch import (
    KaggleSemanticImageBatchProvider,
    SemanticImageBatchItem,
)
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_command import (
    ANIMAL_PHOTO_PREMIUM,
    GENERAL_PREMIUM,
    HUMAN_PHOTO_PREMIUM,
    MASCOT_PREMIUM,
    MEDIA_COMMAND_BRAIN_MODEL,
    MEDIA_COMMAND_FAST_MODEL,
    MediaCommandPlanner,
    benchmark_manifest,
    image_model_config,
)
from ai_agent.core.vision_quality import (
    HybridVisionQualityVerifier,
    VisionQualityRequest,
    result_from_inline_review,
)


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

    # This planner call is deterministic/local and only supplies a conservative fallback
    # profile. The real subject/style routing happens semantically inside the GPU worker.
    plans = MediaCommandPlanner().plan_many(tuple(command for _, command in items))
    for (item_id, _), plan in zip(items, plans, strict=True):
        if plan.mode != "image":
            raise ValueError(f"{item_id} resolved to video; use the video pipeline")

    profiles = (
        HUMAN_PHOTO_PREMIUM,
        ANIMAL_PHOTO_PREMIUM,
        MASCOT_PREMIUM,
        GENERAL_PREMIUM,
    )
    manifest_profiles = benchmark_manifest()["profiles"]
    profile_positive = {profile.name: profile.positive_constraints for profile in profiles}
    profile_negative = {profile.name: profile.negative_constraints for profile in profiles}
    profile_rubrics = {
        profile.name: tuple(manifest_profiles[profile.name]["must_pass"])
        for profile in profiles
    }
    profile_dimensions = {
        profile.name: (profile.width, profile.height)
        for profile in profiles
    }

    worker = KaggleGpuWorker(
        api_token=token,
        username=username,
        timeout=120,
        submission_retry_attempts=5,
        submission_retry_delay_seconds=30,
    )
    config = image_model_config(GENERAL_PREMIUM)
    provider = KaggleSemanticImageBatchProvider(
        worker=worker,
        model=config.model,
        model_variant="fp16" if "RealVisXL" in config.model else None,
        kernel_slug="ai-agent-premium-image-semantic-batch",
        poll_interval=3,
        max_poll_attempts=int(os.environ.get("MEDIA_IMAGE_MAX_POLL_ATTEMPTS", "600")),
        inference_steps=config.inference_steps,
        guidance_scale=config.guidance_scale,
        planner_fast_model=os.environ.get("MEDIA_COMMAND_FAST_MODEL", MEDIA_COMMAND_FAST_MODEL),
        planner_final_model=os.environ.get("MEDIA_COMMAND_MODEL", MEDIA_COMMAND_BRAIN_MODEL),
        vlm_model="Qwen/Qwen3-VL-2B-Instruct",
        vlm_min_score=9.0,
        profile_positive_constraints=profile_positive,
        profile_negative_constraints=profile_negative,
        profile_rubrics=profile_rubrics,
        profile_dimensions=profile_dimensions,
    )

    batch_items = tuple(
        SemanticImageBatchItem(
            item_id=item_id,
            command=command,
            seed=int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big"),
            fallback_profile=plan.profile.name,
        )
        for (item_id, command), plan in zip(items, plans, strict=True)
    )

    started = perf_counter()
    batch = provider.generate_batch(batch_items)
    elapsed = round(perf_counter() - started, 3)

    reviews: dict[str, object] = {}
    requests: dict[str, VisionQualityRequest] = {}
    statuses: dict[str, dict] = {}
    plan_data_by_id: dict[str, dict] = {}

    for (item_id, command), plan, batch_item in zip(items, plans, batch_items, strict=True):
        item_dir = output / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        artifact = batch.artifacts[item_id]
        report = batch.reports[item_id]
        profile = str(report["semantic_profile"])
        benchmark = manifest_profiles[profile]
        prompt = str(report["prompt"])
        plan_data = {
            "command": command,
            "mode": "image",
            "profile": profile,
            "subject_class": str(report.get("subject_class") or ""),
            "style_class": str(report.get("style_class") or ""),
            "model": config.model,
            "model_variant": provider.model_variant,
            "inference_steps": config.inference_steps,
            "guidance_scale": config.guidance_scale,
            "scheduler": config.scheduler,
            "compiler_output": str(report.get("planner_raw") or ""),
            "planner_model": str(report.get("planner_model") or ""),
            "prompt": prompt,
            "negative_prompt": str(report.get("negative_prompt") or ""),
            "width": int(report["width"]),
            "height": int(report["height"]),
            "seed": batch_item.seed,
        }
        plan_data_by_id[item_id] = plan_data
        (item_dir / "plan.json").write_text(
            json.dumps(plan_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (item_dir / "candidate.png").write_bytes(artifact.data)

        digest = sha256(artifact.data).hexdigest()
        if digest in set(benchmark.get("known_rejected_sha256", ())):
            statuses[item_id] = {
                "verified": False,
                "stage": "user_benchmark",
                "error": "exact image was previously rejected by the user",
            }
            continue

        review_text = str(report.get("review_text") or "").strip()
        if not review_text:
            statuses[item_id] = {
                "verified": False,
                "stage": "inline_vlm_quality",
                "error": "inline VLM review missing",
            }
            continue

        request = VisionQualityRequest(
            item_id=item_id,
            image=artifact.data,
            prompt=prompt,
            profile=profile,
            rubric=tuple(benchmark["must_pass"]),
            expected_subject_count=1,
        )
        requests[item_id] = request
        inline_result = result_from_inline_review(
            request,
            review_text,
            model=provider.vlm_model,
            evidence=(
                f"kaggle_kernel:{provider.kernel_slug}",
                f"gpu:{batch.batch_report.get('gpu_name')}",
                f"semantic_profile:{profile}",
                f"subject_class:{report.get('subject_class')}",
                f"style_class:{report.get('style_class')}",
                f"planner_model:{report.get('planner_model')}",
                "batch_single_worker:True",
                "hybrid_fast_model:Qwen/Qwen3-VL-2B-Instruct",
                "hybrid_final_model:Qwen/Qwen3-VL-8B-Instruct",
                "hybrid_clear_pass_score:9.00",
                "hybrid_fast_reject_score:8.00",
            ),
            min_quality_score=9.0,
            min_prompt_match_score=9.0,
        )
        reviews[item_id] = inline_result

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
    fallback_requests = []
    for item_id, review in tuple(reviews.items()):
        request = requests[item_id]
        if verifier._clear_fast_pass(review) or verifier._obvious_fast_reject(request, review):
            continue
        fallback_requests.append(request)

    fallback_elapsed = 0.0
    if fallback_requests:
        fallback_started = perf_counter()
        final_batch = verifier._make_verifier(
            model=verifier.final_model,
            suffix="final",
        ).verify_many(tuple(fallback_requests))
        fallback_elapsed = round(perf_counter() - fallback_started, 3)
        for item in final_batch.items:
            reviews[item.item_id] = item

    for item_id, review in reviews.items():
        item_dir = output / item_id
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
            artifact = batch.artifacts[item_id]
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

    for item_id, _ in items:
        if item_id not in statuses:
            statuses[item_id] = {
                "verified": False,
                "stage": "vlm_quality",
                "error": "visual quality result missing",
            }

    summary = {
        "batch_size": len(items),
        "verified_count": sum(1 for item in statuses.values() if item.get("verified")),
        "failed_count": sum(1 for item in statuses.values() if not item.get("verified")),
        "one_image_model": config.model,
        "one_gpu_worker": True,
        "semantic_router": True,
        "elapsed_seconds": elapsed,
        "fallback_8b_seconds": fallback_elapsed,
        "worker_report": {
            "planner_fast_timing": batch.batch_report.get("planner_fast_timing"),
            "planner_final_timing": batch.batch_report.get("planner_final_timing"),
            "image_model_load_seconds": batch.batch_report.get("image_model_load_seconds"),
            "qa_model_load_seconds": batch.batch_report.get("qa_model_load_seconds"),
            "worker_total_seconds": batch.batch_report.get("worker_total_seconds"),
        },
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
