#!/usr/bin/env python3
"""Production image batch: semantic routing -> style engine -> independent 8B QA."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from ai_agent.core.kaggle_style_image_batch import (
    KaggleStyleRoutedImageBatchProvider,
    StyleRoutedCandidate,
    StyleRoutedImageItem,
)
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_command import (
    ANIMAL_PHOTO_PREMIUM,
    GENERAL_PREMIUM,
    HUMAN_PHOTO_PREMIUM,
    MASCOT_IMAGE_MODEL,
    MASCOT_PREMIUM,
    MEDIA_COMMAND_BRAIN_MODEL,
    MEDIA_COMMAND_FAST_MODEL,
    MediaCommandPlanner,
    UNIFIED_ANIMAL_STYLE_IMAGE_MODEL,
    benchmark_manifest,
)
from ai_agent.core.vision_quality import (
    BatchVisionQualityResult,
    KaggleVisionQualityVerifier,
    VisionQualityRequest,
    VisionQualityResult,
)


class FailClosedVisionVerifier(KaggleVisionQualityVerifier):
    """Malformed model output is a QA failure, never a pipeline crash or pass."""

    @staticmethod
    def _parse_review(text: str) -> dict:
        try:
            return KaggleVisionQualityVerifier._parse_review(text)
        except ValueError:
            return {
                "pass": False,
                "quality_score": 0,
                "prompt_match_score": 0,
                "structure_score": 0,
                "detail_score": 0,
                "aesthetic_score": 0,
                "composition_score": 0,
                "benchmark_match_score": 0,
                "subject_count": None,
                "major_issues": [],
                "critical_defects": ["malformed non-JSON VLM review"],
                "benchmark_failures": [],
                "uncertain_regions": ["QA response could not be verified"],
                "minor_issues": [],
                "summary": "fail closed on malformed QA response",
            }


def _load_commands() -> list[tuple[str, str]]:
    raw = os.environ.get("MEDIA_COMMANDS_JSON", "").strip()
    if not raw:
        raise ValueError("MEDIA_COMMANDS_JSON is required")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("MEDIA_COMMANDS_JSON must be a non-empty list")
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        if isinstance(entry, str):
            item_id = f"item-{index + 1}"
            command = entry.strip()
        elif isinstance(entry, dict):
            item_id = str(entry.get("id") or f"item-{index + 1}").strip()
            command = str(entry.get("command") or "").strip()
        else:
            raise ValueError("each MEDIA_COMMANDS_JSON entry must be text or an object")
        if not item_id or not command:
            raise ValueError("each image item requires id and command")
        if item_id in seen:
            raise ValueError(f"duplicate image item id: {item_id}")
        seen.add(item_id)
        items.append((item_id, command))
    return items


def _profile_maps() -> tuple[dict, dict, dict, dict, dict]:
    profiles = (
        HUMAN_PHOTO_PREMIUM,
        ANIMAL_PHOTO_PREMIUM,
        MASCOT_PREMIUM,
        GENERAL_PREMIUM,
    )
    positive = {profile.name: profile.positive_constraints for profile in profiles}
    negative = {profile.name: profile.negative_constraints for profile in profiles}
    dimensions = {profile.name: (profile.width, profile.height) for profile in profiles}
    manifest = benchmark_manifest()["profiles"]
    rubrics = {
        profile.name: tuple(manifest[profile.name]["must_pass"])
        for profile in profiles
    }

    photo = UNIFIED_ANIMAL_STYLE_IMAGE_MODEL
    mascot = MASCOT_IMAGE_MODEL
    engines = {
        HUMAN_PHOTO_PREMIUM.name: {
            "engine_id": "photoreal",
            "model": photo.model,
            "variant": "fp16",
            "steps": photo.inference_steps,
            "guidance": photo.guidance_scale,
            "scheduler": photo.scheduler,
            "candidate_count": 1,
        },
        ANIMAL_PHOTO_PREMIUM.name: {
            "engine_id": "photoreal",
            "model": photo.model,
            "variant": "fp16",
            "steps": photo.inference_steps,
            "guidance": photo.guidance_scale,
            "scheduler": photo.scheduler,
            "candidate_count": 1,
        },
        GENERAL_PREMIUM.name: {
            "engine_id": "photoreal",
            "model": photo.model,
            "variant": "fp16",
            "steps": photo.inference_steps,
            "guidance": photo.guidance_scale,
            "scheduler": photo.scheduler,
            "candidate_count": 1,
        },
        MASCOT_PREMIUM.name: {
            "engine_id": "mascot3d",
            "model": mascot.model,
            "variant": "fp16",
            "steps": mascot.inference_steps,
            "guidance": mascot.guidance_scale,
            "scheduler": mascot.scheduler,
            "candidate_count": 3,
        },
    }
    return positive, negative, dimensions, rubrics, engines


def _qa_rubric(profile: str, base: tuple[str, ...]) -> tuple[str, ...]:
    anatomy = (
        "independently identify the named species/entity from the prompt and inspect its visible anatomy; reject duplicated, extra, missing or incompatible horn-like, antler-like, tusk-like, ear-like, leg/hoof/paw-like, tail-like, wing-like or fin-like appendages; cute stylization never excuses extra body parts",
        "inspect head silhouette and all four quadrant crops specifically for extra protrusions or duplicated appendages before granting a structure score above 8",
        "no visible text, pseudo-text, logo, watermark, signature, badge, emblem, app/UI mark or branding anywhere in the frame",
    )
    if profile in {MASCOT_PREMIUM.name, ANIMAL_PHOTO_PREMIUM.name}:
        return (*base, *anatomy)
    return (*base, anatomy[-1])


def _review_to_json(result: VisionQualityResult) -> dict:
    return {
        "passed": result.passed,
        "quality_score": result.quality_score,
        "prompt_match_score": result.prompt_match_score,
        "structure_score": result.structure_score,
        "detail_score": result.detail_score,
        "aesthetic_score": result.aesthetic_score,
        "composition_score": result.composition_score,
        "benchmark_match_score": result.benchmark_match_score,
        "subject_count": result.subject_count,
        "major_issues": list(result.major_issues),
        "minor_issues": list(result.minor_issues),
        "review_text": result.review_text,
        "evidence": list(result.evidence),
    }


def _score_floor(result: VisionQualityResult) -> float:
    return min(
        result.quality_score,
        result.prompt_match_score,
        result.structure_score,
        result.detail_score,
        result.aesthetic_score,
        result.composition_score,
        result.benchmark_match_score,
    )


def _run_qa(
    *,
    worker: KaggleGpuWorker,
    candidates: dict[str, tuple[StyleRoutedCandidate, ...]],
    plans: dict[str, dict],
    rubrics: dict[str, tuple[str, ...]],
    suffix: str,
) -> tuple[BatchVisionQualityResult, dict[str, StyleRoutedCandidate]]:
    requests: list[VisionQualityRequest] = []
    candidate_by_qa_id: dict[str, StyleRoutedCandidate] = {}
    for item_id, item_candidates in candidates.items():
        plan = plans[item_id]
        profile = str(plan["semantic_profile"])
        for candidate in item_candidates:
            qa_id = f"{item_id}__c{candidate.candidate_index}"
            candidate_by_qa_id[qa_id] = candidate
            requests.append(VisionQualityRequest(
                item_id=qa_id,
                image=candidate.artifact.data,
                prompt=str(plan["prompt"]),
                profile=profile,
                rubric=_qa_rubric(profile, rubrics[profile]),
                expected_subject_count=1,
            ))

    verifier = FailClosedVisionVerifier(
        worker=worker,
        model="Qwen/Qwen3-VL-8B-Instruct",
        kernel_slug=f"ai-agent-production-image-8b-{suffix}",
        poll_interval=3,
        max_poll_attempts=240,
        min_quality_score=9.0,
        min_prompt_match_score=9.0,
    )
    batch = verifier.verify_many(tuple(requests))
    return batch, candidate_by_qa_id


def _select(
    qa: BatchVisionQualityResult,
    candidate_by_qa_id: dict[str, StyleRoutedCandidate],
) -> tuple[dict[str, StyleRoutedCandidate], dict[str, list[tuple[StyleRoutedCandidate, VisionQualityResult]]]]:
    grouped: dict[str, list[tuple[StyleRoutedCandidate, VisionQualityResult]]] = {}
    for result in qa.items:
        candidate = candidate_by_qa_id[result.item_id]
        grouped.setdefault(candidate.item_id, []).append((candidate, result))

    selected: dict[str, StyleRoutedCandidate] = {}
    for item_id, entries in grouped.items():
        passing = [entry for entry in entries if entry[1].passed]
        if not passing:
            continue
        candidate, _ = max(
            passing,
            key=lambda entry: (_score_floor(entry[1]), -entry[0].candidate_index),
        )
        selected[item_id] = candidate
    return selected, grouped


def main() -> int:
    items = _load_commands()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token or not username:
        raise RuntimeError("KAGGLE_API_TOKEN and KAGGLE_USERNAME are required")
    output = Path(os.environ.get("MEDIA_OUTPUT_DIR", "production-image-batch-output"))
    output.mkdir(parents=True, exist_ok=True)

    positive, negative, dimensions, rubrics, engines = _profile_maps()
    fallback_plans = MediaCommandPlanner().plan_many(tuple(command for _, command in items))
    batch_items = tuple(
        StyleRoutedImageItem(
            item_id=item_id,
            command=command,
            seed=int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big"),
            fallback_profile=plan.profile.name,
        )
        for (item_id, command), plan in zip(items, fallback_plans, strict=True)
    )

    worker = KaggleGpuWorker(
        api_token=token,
        username=username,
        timeout=120,
        submission_retry_attempts=5,
        submission_retry_delay_seconds=30,
    )
    provider = KaggleStyleRoutedImageBatchProvider(
        worker=worker,
        profile_engines=engines,
        profile_positive_constraints=positive,
        profile_negative_constraints=negative,
        profile_dimensions=dimensions,
        kernel_slug="ai-agent-production-style-image",
        poll_interval=3,
        max_poll_attempts=int(os.environ.get("MEDIA_IMAGE_MAX_POLL_ATTEMPTS", "600")),
        planner_fast_model=os.environ.get("MEDIA_COMMAND_FAST_MODEL", MEDIA_COMMAND_FAST_MODEL),
        planner_final_model=os.environ.get("MEDIA_COMMAND_MODEL", MEDIA_COMMAND_BRAIN_MODEL),
    )

    started = perf_counter()
    generated = provider.generate_batch(batch_items)
    qa, qa_candidates = _run_qa(
        worker=worker,
        candidates=generated.candidates,
        plans=generated.plans,
        rubrics=rubrics,
        suffix="initial",
    )
    selected, grouped = _select(qa, qa_candidates)

    retry_ids = [item.item_id for item in batch_items if item.item_id not in selected]
    retry_generated = None
    retry_qa = None
    retry_grouped: dict[str, list[tuple[StyleRoutedCandidate, VisionQualityResult]]] = {}
    if retry_ids:
        retry_items = []
        command_by_id = dict(items)
        item_by_id = {item.item_id: item for item in batch_items}
        for item_id in retry_ids:
            entries = grouped.get(item_id, [])
            best_issues: tuple[str, ...] = ()
            if entries:
                _, best_review = max(entries, key=lambda entry: _score_floor(entry[1]))
                best_issues = best_review.major_issues[:4]
            correction = "; ".join(best_issues) or "strict QA rejection"
            source = item_by_id[item_id]
            retry_items.append(StyleRoutedImageItem(
                item_id=item_id,
                command=(
                    command_by_id[item_id]
                    + " Correction retry: fix the previous visible defects without changing the requested subject or style. "
                    + "Use clean species-correct anatomy with no duplicated or extra appendages and no branding. Previous QA: "
                    + correction
                ),
                seed=source.seed + 1000,
                fallback_profile=source.fallback_profile,
            ))
        retry_generated = provider.generate_batch(tuple(retry_items))
        retry_qa, retry_candidates = _run_qa(
            worker=worker,
            candidates=retry_generated.candidates,
            plans=retry_generated.plans,
            rubrics=rubrics,
            suffix="retry",
        )
        retry_selected, retry_grouped = _select(retry_qa, retry_candidates)
        selected.update(retry_selected)

    elapsed = round(perf_counter() - started, 3)
    statuses: dict[str, dict] = {}
    for item_id, command in items:
        item_dir = output / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        source_plan = (
            retry_generated.plans[item_id]
            if retry_generated is not None and item_id in retry_generated.plans
            else generated.plans[item_id]
        )
        (item_dir / "plan.json").write_text(
            json.dumps(source_plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        all_entries = [*grouped.get(item_id, []), *retry_grouped.get(item_id, [])]
        qa_dump = []
        for candidate, review in all_entries:
            candidate_name = f"candidate-{len(qa_dump)}.png"
            (item_dir / candidate_name).write_bytes(candidate.artifact.data)
            qa_dump.append({
                "candidate_file": candidate_name,
                "candidate_index": candidate.candidate_index,
                "model": candidate.artifact.model,
                "image_sha256": sha256(candidate.artifact.data).hexdigest(),
                "review": _review_to_json(review),
            })
        (item_dir / "qa-candidates.json").write_text(
            json.dumps(qa_dump, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        final = selected.get(item_id)
        if final is None:
            issues = []
            for _, review in all_entries:
                issues.extend(review.major_issues)
            statuses[item_id] = {
                "verified": False,
                "stage": "8b_visual_quality",
                "error": "; ".join(dict.fromkeys(issues)) or "no candidate passed strict 8B QA",
            }
            continue
        final_bytes = final.artifact.data
        (item_dir / "image.png").write_bytes(final_bytes)
        statuses[item_id] = {
            "verified": True,
            "stage": "complete",
            "model": final.artifact.model,
            "image_sha256": sha256(final_bytes).hexdigest(),
        }

    summary = {
        "batch_size": len(items),
        "verified_count": sum(1 for value in statuses.values() if value.get("verified")),
        "failed_count": sum(1 for value in statuses.values() if not value.get("verified")),
        "semantic_router": True,
        "style_specialized_engines": True,
        "single_generation_worker_per_round": True,
        "engines": {
            key: {"engine_id": value["engine_id"], "model": value["model"], "scheduler": value["scheduler"]}
            for key, value in engines.items()
        },
        "retry_count": len(retry_ids),
        "retry_ids": retry_ids,
        "elapsed_seconds": elapsed,
        "initial_worker_report": generated.batch_report,
        "retry_worker_report": retry_generated.batch_report if retry_generated else None,
        "statuses": statuses,
    }
    (output / "batch-status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("STYLE_ROUTED_IMAGE_BATCH_DONE")
    print(json.dumps({
        "verified_count": summary["verified_count"],
        "failed_count": summary["failed_count"],
        "retry_ids": retry_ids,
        "elapsed_seconds": elapsed,
        "statuses": statuses,
    }, ensure_ascii=False, indent=2))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
