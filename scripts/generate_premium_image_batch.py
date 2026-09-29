#!/usr/bin/env python3
"""Generate multiple premium images while loading command/VLM models only once per batch."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re

from ai_agent.core.image_model import ImageGenerationRequest
from ai_agent.core.kaggle_image import KaggleImageProvider
from ai_agent.core.kaggle_model import KaggleModelProvider
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_command import MEDIA_COMMAND_BRAIN_MODEL, MediaCommandPlanner, benchmark_manifest
from ai_agent.core.vision_quality import KaggleVisionQualityVerifier, VisionQualityRequest


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
    language_model = KaggleModelProvider(
        worker=worker,
        model=os.environ.get("MEDIA_COMMAND_MODEL", MEDIA_COMMAND_BRAIN_MODEL),
        kernel_slug="ai-agent-media-command-batch",
        poll_interval=15,
        max_poll_attempts=120,
        max_new_tokens=180,
        temperature=0.0,
        enable_thinking=False,
    )
    planner = MediaCommandPlanner(language_model)
    try:
        plans = planner.plan_many(tuple(command for _, command in items))
    except Exception as exc:
        (output / "planning-failure.json").write_text(
            json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            raw_report = worker.download_output_file(
                language_model.kernel_slug,
                "responses.json",
            )
            (output / "compiler-responses.json").write_bytes(raw_report)
        except Exception:
            pass
        raise

    generated: dict[str, tuple[object, object, dict]] = {}
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
            kernel_slug=f"ai-agent-image-{_slug(item_id)}",
            poll_interval=15,
            max_poll_attempts=180,
            inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
            scheduler=config.scheduler,
            prompt_alignment_threshold=float(os.environ.get("MEDIA_PROMPT_ALIGNMENT_THRESHOLD", "0.22")),
            visual_quality_margin_threshold=float(os.environ.get("MEDIA_VISUAL_QUALITY_MARGIN_THRESHOLD", "0.015")),
            enforce_visual_quality_margin=False,
            enable_clip_precheck=os.environ.get("MEDIA_CLIP_PRECHECK", "0") == "1",
            quality_good_text=config.quality_good_text,
            quality_bad_texts=config.quality_bad_texts,
        )
        request = ImageGenerationRequest(
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            width=plan.profile.width,
            height=plan.profile.height,
            seed=seed,
        )
        try:
            artifact = provider.generate(request)
        except Exception as exc:
            statuses[item_id] = {"verified": False, "stage": "image_generation", "error": str(exc)}
            try:
                candidate = worker.download_output_file(provider.kernel_slug, "generated.png")
                (item_dir / "candidate.png").write_bytes(candidate)
            except Exception:
                pass
            continue

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

    review_requests = tuple(
        VisionQualityRequest(
            item_id=item_id,
            image=artifact.data,
            prompt=plan.prompt,
            profile=plan.profile.name,
            rubric=tuple(benchmark_manifest()["profiles"][plan.profile.name]["must_pass"]),
            expected_subject_count=1,
        )
        for item_id, (plan, artifact, _) in generated.items()
    )

    reviews = {}
    if review_requests:
        verifier = KaggleVisionQualityVerifier(
            worker=worker,
            kernel_slug="ai-agent-premium-image-vlm-batch",
            poll_interval=15,
            max_poll_attempts=120,
            min_quality_score=8.0,
            min_prompt_match_score=8.0,
        )
        batch_review = verifier.verify_many(review_requests)
        reviews = {item.item_id: item for item in batch_review.items}

    for item_id, (plan, artifact, plan_data) in generated.items():
        item_dir = output / item_id
        review = reviews[item_id]
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
        "brain_model": os.environ.get("MEDIA_COMMAND_MODEL", MEDIA_COMMAND_BRAIN_MODEL),
        "clip_precheck": os.environ.get("MEDIA_CLIP_PRECHECK", "0") == "1",
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
