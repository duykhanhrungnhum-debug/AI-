#!/usr/bin/env python3
"""Simple production image skill.

Natural-language request -> semantic planner -> style engine -> strict 8B QA ->
one targeted retry for failed items only.

No detector model, no species keyword table, no planner-generated anatomy counts.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from ai_agent.core.kaggle_production_image_batch import (
    KaggleProductionImageBatchProvider,
    ProductionImageCandidate,
    ProductionImageItem,
)
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_command import (
    ANIMAL_IMAGE_MODEL,
    ANIMAL_PHOTO_PREMIUM,
    GENERAL_PREMIUM,
    HUMAN_PHOTO_PREMIUM,
    MASCOT_IMAGE_MODEL,
    MASCOT_PREMIUM,
    MEDIA_COMMAND_BRAIN_MODEL,
    MediaCommandPlanner,
    benchmark_manifest,
)
from ai_agent.core.vision_quality import (
    BatchVisionQualityResult,
    KaggleVisionQualityVerifier,
    VisionQualityRequest,
    VisionQualityResult,
)

QUALITY_TARGET = 9.0


class FailClosedVisionVerifier(KaggleVisionQualityVerifier):
    """Malformed QA output is a rejection, never a crash or false PASS."""

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
                "major_issues": ["QA returned malformed output"],
                "critical_defects": ["malformed non-JSON VLM review"],
                "benchmark_failures": [],
                "uncertain_regions": [],
                "minor_issues": [],
                "summary": "fail closed on malformed QA response",
            }


def load_requests() -> list[dict[str, str]]:
    raw = os.environ.get("MEDIA_COMMANDS_JSON", "").strip()
    if not raw:
        raise ValueError("MEDIA_COMMANDS_JSON is required")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("MEDIA_COMMANDS_JSON must be a non-empty list")

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        if isinstance(entry, str):
            entry = {"id": f"item-{index + 1}", "command": entry}
        if not isinstance(entry, dict):
            raise ValueError("each image request must be text or an object")
        item_id = str(entry.get("id") or f"item-{index + 1}").strip()
        command = str(entry.get("command") or "").strip()
        if not item_id or not command or item_id in seen:
            raise ValueError("image requests require unique id and non-empty command")
        seen.add(item_id)
        result.append({"id": item_id, "command": command})
    return result


def build_maps():
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

    photo = ANIMAL_IMAGE_MODEL
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
            "candidate_count": 2,
        },
    }
    return positive, negative, dimensions, rubrics, engines


def qa_rubric(profile: str, base: tuple[str, ...]) -> tuple[str, ...]:
    shared = (
        "match the ORIGINAL USER REQUEST exactly, including the requested subject/species, count, action and visual style",
        "reject a generalized, mistranslated or different subject even if the image is attractive",
        "reject obvious malformed anatomy, duplicated body parts, extra appendages, broken limbs or impossible geometry",
        "no visible text, pseudo-text, watermark, signature, logo, badge or UI mark unless the user explicitly requested it",
        "score honestly; 9/10 means production-ready with no obvious visible defect",
    )
    return (*base, *shared)


def qa_prompt(command: str, plan: dict) -> str:
    return (
        "ORIGINAL USER REQUEST (authoritative): " + command + "\n"
        "ENGLISH SUBJECT: " + str(plan.get("subject_english") or "") + "\n"
        "COMPILED PROMPT (secondary): " + str(plan.get("prompt") or "") + "\n"
        "Judge the image primarily against the original request."
    )


def score_floor(result: VisionQualityResult) -> float:
    return min(
        result.quality_score,
        result.prompt_match_score,
        result.structure_score,
        result.detail_score,
        result.aesthetic_score,
        result.composition_score,
        result.benchmark_match_score,
    )


def run_qa(worker, candidates, plans, commands, rubrics, suffix):
    requests: list[VisionQualityRequest] = []
    by_id: dict[str, ProductionImageCandidate] = {}
    for item_id, item_candidates in candidates.items():
        plan = plans[item_id]
        profile = str(plan["semantic_profile"])
        for candidate in item_candidates:
            qa_id = f"{item_id}__c{candidate.candidate_index}"
            by_id[qa_id] = candidate
            requests.append(VisionQualityRequest(
                item_id=qa_id,
                image=candidate.artifact.data,
                prompt=qa_prompt(commands[item_id], plan),
                profile=profile,
                rubric=qa_rubric(profile, rubrics[profile]),
                expected_subject_count=1,
            ))

    if not requests:
        return BatchVisionQualityResult(tuple()), by_id

    print("AI_AGENT_PROGRESS " + json.dumps({
        "step": "qa_start", "round": suffix, "candidate_count": len(requests)
    }, separators=(",", ":")), flush=True)

    verifier = FailClosedVisionVerifier(
        worker=worker,
        model="Qwen/Qwen3-VL-8B-Instruct",
        kernel_slug=f"ai-agent-simple-image-qa-{suffix}",
        poll_interval=3,
        max_poll_attempts=300,
        min_quality_score=QUALITY_TARGET,
        min_prompt_match_score=QUALITY_TARGET,
    )
    result = verifier.verify_many(tuple(requests))
    print("AI_AGENT_PROGRESS " + json.dumps({
        "step": "qa_complete", "round": suffix, "candidate_count": len(requests)
    }, separators=(",", ":")), flush=True)
    return result, by_id


def group_reviews(qa: BatchVisionQualityResult, by_id):
    grouped: dict[str, list[tuple[ProductionImageCandidate, VisionQualityResult]]] = {}
    for review in qa.items:
        candidate = by_id[review.item_id]
        grouped.setdefault(candidate.item_id, []).append((candidate, review))
    return grouped


def select_verified(grouped):
    selected: dict[str, ProductionImageCandidate] = {}
    for item_id, entries in grouped.items():
        passing = [
            entry for entry in entries
            if entry[1].passed and score_floor(entry[1]) >= QUALITY_TARGET
        ]
        if passing:
            selected[item_id] = max(
                passing,
                key=lambda entry: (score_floor(entry[1]), -entry[0].candidate_index),
            )[0]
    return selected


def review_json(review: VisionQualityResult) -> dict:
    return {
        "passed": review.passed,
        "score_floor": score_floor(review),
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
    }


def main() -> int:
    requests = load_requests()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token or not username:
        raise RuntimeError("KAGGLE_API_TOKEN and KAGGLE_USERNAME are required")

    output = Path(os.environ.get("MEDIA_OUTPUT_DIR", "simple-image-output"))
    output.mkdir(parents=True, exist_ok=True)
    positive, negative, dimensions, rubrics, engines = build_maps()
    commands = {item["id"]: item["command"] for item in requests}

    fallback = MediaCommandPlanner().plan_many(tuple(item["command"] for item in requests))
    items = tuple(
        ProductionImageItem(
            item_id=item["id"],
            command=item["command"],
            seed=int.from_bytes(sha256(item["command"].encode("utf-8")).digest()[:4], "big"),
            fallback_profile=plan.profile.name,
            hard_gates=(),
        )
        for item, plan in zip(requests, fallback, strict=True)
    )

    worker = KaggleGpuWorker(
        api_token=token,
        username=username,
        timeout=120,
        submission_retry_attempts=5,
        submission_retry_delay_seconds=30,
    )
    provider = KaggleProductionImageBatchProvider(
        worker=worker,
        profile_engines=engines,
        profile_positive_constraints=positive,
        profile_negative_constraints=negative,
        profile_dimensions=dimensions,
        kernel_slug="ai-agent-simple-image-standard",
        poll_interval=3,
        max_poll_attempts=int(os.environ.get("MEDIA_IMAGE_MAX_POLL_ATTEMPTS", "700")),
        planner_model=os.environ.get("MEDIA_COMMAND_MODEL", MEDIA_COMMAND_BRAIN_MODEL),
    )

    started = perf_counter()
    generated = provider.generate_batch(items)
    qa, qa_by_id = run_qa(worker, generated.candidates, generated.plans, commands, rubrics, "initial")
    grouped = group_reviews(qa, qa_by_id)
    selected = select_verified(grouped)

    retry_ids = [item.item_id for item in items if item.item_id not in selected]
    retry_generated = None
    retry_grouped = {}
    if retry_ids:
        by_item = {item.item_id: item for item in items}
        retry_items = []
        for item_id in retry_ids:
            reviews = grouped.get(item_id, [])
            issues: list[str] = []
            if reviews:
                _, best = max(reviews, key=lambda entry: score_floor(entry[1]))
                issues.extend(best.major_issues[:4])
            original = by_item[item_id]
            retry_items.append(ProductionImageItem(
                item_id=item_id,
                command=(
                    original.command
                    + " Correction retry: preserve the exact requested subject and style. Fix these visible issues only: "
                    + ("; ".join(issues) or "quality below 9/10")
                    + ". Produce a clean production-ready image."
                ),
                seed=original.seed + 1000,
                fallback_profile=original.fallback_profile,
                hard_gates=(),
            ))
        print("AI_AGENT_PROGRESS " + json.dumps({
            "step": "retry_start", "item_ids": retry_ids
        }, separators=(",", ":")), flush=True)
        retry_generated = provider.generate_batch(tuple(retry_items))
        retry_qa, retry_by_id = run_qa(
            worker, retry_generated.candidates, retry_generated.plans,
            commands, rubrics, "retry"
        )
        retry_grouped = group_reviews(retry_qa, retry_by_id)
        selected.update(select_verified(retry_grouped))

    statuses = {}
    for item in items:
        item_dir = output / item.item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        plan = (
            retry_generated.plans[item.item_id]
            if retry_generated and item.item_id in retry_generated.plans
            else generated.plans[item.item_id]
        )
        (item_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        all_reviews = [*grouped.get(item.item_id, []), *retry_grouped.get(item.item_id, [])]
        qa_dump = []
        for index, (candidate, review) in enumerate(all_reviews):
            filename = f"candidate-{index}.png"
            (item_dir / filename).write_bytes(candidate.artifact.data)
            qa_dump.append({
                "candidate_file": filename,
                "model": candidate.artifact.model,
                "image_sha256": sha256(candidate.artifact.data).hexdigest(),
                "qa": review_json(review),
            })
        (item_dir / "qa.json").write_text(
            json.dumps(qa_dump, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        final = selected.get(item.item_id)
        if final is None:
            best_score = max((score_floor(review) for _, review in all_reviews), default=0)
            statuses[item.item_id] = {
                "verified": False,
                "stage": "8b_qa",
                "best_score_floor": best_score,
                "target": QUALITY_TARGET,
            }
            continue
        final_bytes = final.artifact.data
        (item_dir / "image.png").write_bytes(final_bytes)
        final_review = next(
            review for candidate, review in all_reviews
            if candidate.artifact.data == final_bytes
        )
        statuses[item.item_id] = {
            "verified": True,
            "stage": "complete",
            "model": final.artifact.model,
            "score_floor": score_floor(final_review),
            "target": QUALITY_TARGET,
            "image_sha256": sha256(final_bytes).hexdigest(),
        }

    elapsed = round(perf_counter() - started, 3)
    summary = {
        "pipeline": "natural_request -> semantic_router -> style_engine -> 8b_qa -> one_targeted_retry",
        "quality_target": QUALITY_TARGET,
        "detector_enabled": False,
        "batch_size": len(items),
        "verified_count": sum(1 for status in statuses.values() if status["verified"]),
        "failed_count": sum(1 for status in statuses.values() if not status["verified"]),
        "retry_ids": retry_ids,
        "elapsed_seconds": elapsed,
        "statuses": statuses,
    }
    (output / "batch-status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("AI_AGENT_SIMPLE_IMAGE_DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
