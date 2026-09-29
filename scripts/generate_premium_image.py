#!/usr/bin/env python3
"""Generate a premium image from one natural-language command."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

from ai_agent.core.image_model import ImageGenerationRequest
from ai_agent.core.kaggle_image import KaggleImageProvider
from ai_agent.core.kaggle_model import KaggleModelProvider
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_command import MEDIA_COMMAND_BRAIN_MODEL, MEDIA_COMMAND_FAST_MODEL, MediaCommandPlanner, benchmark_manifest
from ai_agent.core.vision_quality import HybridVisionQualityVerifier, VisionQualityRequest, result_from_inline_review


def main() -> int:
    command = os.environ.get("MEDIA_COMMAND", "").strip()
    if not command:
        raise ValueError("MEDIA_COMMAND is required")

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token or not username:
        raise RuntimeError("KAGGLE_API_TOKEN and KAGGLE_USERNAME are required")

    output = Path(os.environ.get("MEDIA_OUTPUT_DIR", "premium-image-output"))
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
    if use_inline_planner:
        plan = MediaCommandPlanner().plan(command)
    else:
        fast_language_model = KaggleModelProvider(
            worker=worker,
            model=fast_model_name,
            kernel_slug="ai-agent-media-command-fast",
            poll_interval=15,
            max_poll_attempts=120,
            max_new_tokens=140,
            temperature=0.0,
            enable_thinking=False,
        )
        try:
            plan = MediaCommandPlanner(fast_language_model).plan(command)
        except Exception:
            final_language_model = KaggleModelProvider(
                worker=worker,
                model=final_model_name,
                kernel_slug="ai-agent-media-command-final",
                poll_interval=15,
                max_poll_attempts=120,
                max_new_tokens=180,
                temperature=0.0,
                enable_thinking=False,
            )
            plan = MediaCommandPlanner(final_language_model).plan(command)
    if plan.mode != "image":
        raise ValueError("MEDIA_COMMAND resolved to video; use the video pipeline")

    config = plan.model_config
    provider = KaggleImageProvider(
        worker=worker,
        model=config.model,
        kernel_slug="ai-agent-premium-image-command",
        poll_interval=15,
        max_poll_attempts=180,
        inference_steps=config.inference_steps,
        guidance_scale=config.guidance_scale,
        scheduler=config.scheduler,
        prompt_alignment_threshold=float(os.environ.get("MEDIA_PROMPT_ALIGNMENT_THRESHOLD", "0.22")),
        visual_quality_margin_threshold=float(os.environ.get("MEDIA_VISUAL_QUALITY_MARGIN_THRESHOLD", "0.015")),
        enforce_visual_quality_margin=False,
        quality_good_text=config.quality_good_text,
        quality_bad_texts=config.quality_bad_texts,
        enable_clip_precheck=os.environ.get("MEDIA_CLIP_PRECHECK", "0") == "1",
        enable_cpu_offload=os.environ.get("MEDIA_CPU_OFFLOAD", "0") == "1",
        enable_inline_planner=use_inline_planner,
        inline_planner_fast_model=fast_model_name,
        inline_planner_final_model=final_model_name,
        inline_planner_raw_command=command if use_inline_planner else "",
        inline_planner_positive_constraints=plan.profile.positive_constraints if use_inline_planner else "",
        enable_inline_vlm=True,
        inline_vlm_model="Qwen/Qwen3-VL-2B-Instruct",
        inline_vlm_profile=plan.profile.name,
        inline_vlm_rubric=tuple(benchmark_manifest()["profiles"][plan.profile.name]["must_pass"]),
        inline_vlm_expected_subject_count=1,
        inline_vlm_min_score=9.0,
    )
    seed = int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")
    request = ImageGenerationRequest(
        prompt=str(plan_data.get("prompt") or plan.prompt),
        negative_prompt=plan.negative_prompt,
        width=plan.profile.width,
        height=plan.profile.height,
        seed=seed,
    )

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
    (output / "plan.json").write_text(
        json.dumps(plan_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        artifact = provider.generate(request)
    except Exception as exc:
        failure = {"verified": False, "error": str(exc), **plan_data}
        (output / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # Preserve the rejected candidate without another GPU render.
        try:
            (output / "candidate.png").write_bytes(
                worker.download_output_file(provider.kernel_slug, "generated.png")
            )
            (output / "image_report.json").write_bytes(
                worker.download_output_file(provider.kernel_slug, "image_report.json")
            )
        except Exception as collect_exc:
            (output / "candidate-collect-error.txt").write_text(
                repr(collect_exc) + "\n", encoding="utf-8"
            )
        raise

    inline_planner_report = provider.last_report.get("inline_planner")
    if isinstance(inline_planner_report, dict):
        actual_prompt = str(provider.last_report.get("compiled_prompt") or "").strip()
        if actual_prompt:
            plan_data["compiler_output"] = str(inline_planner_report.get("raw_output") or "")
            plan_data["prompt"] = actual_prompt
            (output / "plan.json").write_text(
                json.dumps(plan_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    candidate_path = output / "candidate.png"
    candidate_path.write_bytes(artifact.data)

    benchmark = benchmark_manifest()["profiles"][plan.profile.name]
    candidate_digest = sha256(artifact.data).hexdigest()
    if candidate_digest in set(benchmark.get("known_rejected_sha256", ())):
        (output / "user-benchmark-rejection.json").write_text(
            json.dumps({
                "verified": False,
                "stage": "user_benchmark",
                "image_sha256": candidate_digest,
                "reason": "exact image was previously rejected by the user",
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("generated image matches a user-rejected benchmark example")

    rubric = tuple(benchmark["must_pass"])
    request_for_review = VisionQualityRequest(
        item_id="premium-image",
        image=artifact.data,
        prompt=plan.prompt,
        profile=plan.profile.name,
        rubric=rubric,
        expected_subject_count=1,
    )
    vlm = HybridVisionQualityVerifier(
        worker=worker,
        kernel_slug="ai-agent-premium-image-vlm",
        poll_interval=15,
        max_poll_attempts=120,
        min_quality_score=9.0,
        min_prompt_match_score=9.0,
        clear_pass_score=9.0,
        fast_reject_score=8.0,
    )
    try:
        inline_payload = provider.last_report.get("inline_vlm")
        if not isinstance(inline_payload, dict):
            raise RuntimeError("inline VLM report missing")
        inline_text = str(inline_payload.get("review_text") or "").strip()
        inline_model = str(inline_payload.get("model") or "").strip()
        if not inline_text or inline_model != "Qwen/Qwen3-VL-2B-Instruct":
            raise RuntimeError("inline VLM report invalid")
        visual_review = result_from_inline_review(
            request_for_review,
            inline_text,
            model=inline_model,
            evidence=(
                f"kaggle_kernel:{provider.kernel_slug}",
                f"gpu:{provider.last_report.get('gpu_name')}",
                "hybrid_fast_model:Qwen/Qwen3-VL-2B-Instruct",
                "hybrid_final_model:Qwen/Qwen3-VL-8B-Instruct",
                "hybrid_clear_pass_score:9.00",
                "hybrid_fast_reject_score:8.00",
            ),
            min_quality_score=9.0,
            min_prompt_match_score=9.0,
        )
        if (
            not vlm._clear_fast_pass(visual_review)
            and not vlm._obvious_fast_reject(request_for_review, visual_review)
        ):
            visual_review = vlm._make_verifier(
                model=vlm.final_model,
                suffix="final",
            ).verify(request_for_review)
    except Exception as exc:
        (output / "vlm-failure.json").write_text(
            json.dumps({
                "verified": False,
                "error": str(exc),
                "stage": "vlm_quality",
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise

    visual_review_payload = {
        "passed": visual_review.passed,
        "quality_score": visual_review.quality_score,
        "prompt_match_score": visual_review.prompt_match_score,
        "subject_count": visual_review.subject_count,
        "major_issues": list(visual_review.major_issues),
        "minor_issues": list(visual_review.minor_issues),
        "review_text": visual_review.review_text,
        "evidence": list(visual_review.evidence),
    }
    (output / "vlm-quality.json").write_text(
        json.dumps(visual_review_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not visual_review.passed:
        raise RuntimeError(
            "VLM visual quality rejected image: "
            + "; ".join(visual_review.major_issues or ("quality gate failed",))
        )

    image_path = output / "image.png"
    image_path.write_bytes(artifact.data)
    manifest = {
        "verified": True,
        "vlm_quality": visual_review_payload,
        **plan_data,
        "provider": artifact.provider,
        "evidence": list(artifact.evidence),
        "image_path": str(image_path),
        "image_sha256": sha256(artifact.data).hexdigest(),
        "size_bytes": len(artifact.data),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PREMIUM_IMAGE_COMMAND_OK")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
