#!/usr/bin/env python3
"""Standard production image skill.

Semantic router -> style-specialized generator -> optional detector hard gate -> strict 8B QA
-> one targeted retry for failed items only.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from ai_agent.core.kaggle_production_image_batch import (
    KaggleProductionImageBatchProvider,
    ObjectCountGate,
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


class FailClosedVisionVerifier(KaggleVisionQualityVerifier):
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


def _load_items() -> list[dict]:
    raw = os.environ.get("MEDIA_COMMANDS_JSON", "").strip()
    if not raw:
        raise ValueError("MEDIA_COMMANDS_JSON is required")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("MEDIA_COMMANDS_JSON must be a non-empty list")
    results: list[dict] = []
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
        gates: list[ObjectCountGate] = []
        raw_gates = entry.get("hard_gates") or []
        if not isinstance(raw_gates, list):
            raise ValueError("hard_gates must be a list")
        for gate in raw_gates:
            if not isinstance(gate, dict):
                raise ValueError("hard_gates entries must be objects")
            gates.append(ObjectCountGate(
                label=str(gate.get("label") or "").strip(),
                query=str(gate.get("query") or gate.get("label") or "").strip(),
                expected_count=int(gate.get("expected_count")),
            ))
        results.append({"id": item_id, "command": command, "hard_gates": tuple(gates)})
    return results


def _maps():
    profiles = (HUMAN_PHOTO_PREMIUM, ANIMAL_PHOTO_PREMIUM, MASCOT_PREMIUM, GENERAL_PREMIUM)
    positive = {profile.name: profile.positive_constraints for profile in profiles}
    negative = {profile.name: profile.negative_constraints for profile in profiles}
    dimensions = {profile.name: (profile.width, profile.height) for profile in profiles}
    manifest = benchmark_manifest()["profiles"]
    rubrics = {profile.name: tuple(manifest[profile.name]["must_pass"]) for profile in profiles}

    photo = ANIMAL_IMAGE_MODEL
    mascot = MASCOT_IMAGE_MODEL
    engines = {
        HUMAN_PHOTO_PREMIUM.name: {
            "engine_id": "photoreal", "model": photo.model, "variant": "fp16",
            "steps": photo.inference_steps, "guidance": photo.guidance_scale,
            "scheduler": photo.scheduler, "candidate_count": 2,
        },
        ANIMAL_PHOTO_PREMIUM.name: {
            "engine_id": "photoreal", "model": photo.model, "variant": "fp16",
            "steps": photo.inference_steps, "guidance": photo.guidance_scale,
            "scheduler": photo.scheduler, "candidate_count": 2,
        },
        GENERAL_PREMIUM.name: {
            "engine_id": "photoreal", "model": photo.model, "variant": "fp16",
            "steps": photo.inference_steps, "guidance": photo.guidance_scale,
            "scheduler": photo.scheduler, "candidate_count": 2,
        },
        MASCOT_PREMIUM.name: {
            "engine_id": "mascot3d", "model": mascot.model, "variant": "fp16",
            "steps": mascot.inference_steps, "guidance": mascot.guidance_scale,
            "scheduler": mascot.scheduler, "candidate_count": 3,
        },
    }
    return positive, negative, dimensions, rubrics, engines


def _qa_rubric(profile: str, base: tuple[str, ...]) -> tuple[str, ...]:
    shared = (
        "the image must match the ORIGINAL USER REQUEST, not merely the translated generation prompt; reject a generalized or different species/entity",
        "no visible text, pseudo-text, logo, watermark, signature, badge, emblem, UI/app mark or branding anywhere",
    )
    anatomy = (
        "independently inspect species/entity anatomy and reject duplicated, extra, missing or incompatible horn-like, antler-like, tusk-like, ear-like, leg/hoof/paw-like, tail-like, wing-like or fin-like appendages",
        "cute stylization may change proportions but never excuses extra body parts or wrong species identity",
    )
    return (*base, *shared, *(anatomy if profile in {ANIMAL_PHOTO_PREMIUM.name, MASCOT_PREMIUM.name} else ()))


def _qa_prompt(command: str, plan: dict) -> str:
    return (
        "ORIGINAL USER REQUEST (authoritative; may be Vietnamese): " + command + "\n"
        "COMPILED GENERATION PROMPT (secondary): " + str(plan.get("prompt") or "") + "\n"
        "PLANNER SUBJECT NAME: " + str(plan.get("subject_name") or "") + "\n"
        "Reject if the visible image follows a different or generalized subject/species than the original request."
    )


def _run_qa(worker, candidates, plans, commands, rubrics, suffix):
    requests: list[VisionQualityRequest] = []
    by_qa_id: dict[str, ProductionImageCandidate] = {}
    detector_rejects: dict[str, list[str]] = {}
    for item_id, item_candidates in candidates.items():
        plan = plans[item_id]
        profile = str(plan["semantic_profile"])
        for candidate in item_candidates:
            gate = candidate.report.get("detector_gate") or {"passed": True, "checks": []}
            if gate.get("passed") is not True:
                issues = []
                for check in gate.get("checks") or []:
                    issues.append(
                        f"detector {check.get('label')}: expected {check.get('expected_count')}, detected {check.get('detected_count')}"
                    )
                detector_rejects.setdefault(item_id, []).extend(issues or ["detector hard gate failed"])
                continue
            qa_id = f"{item_id}__c{candidate.candidate_index}"
            by_qa_id[qa_id] = candidate
            requests.append(VisionQualityRequest(
                item_id=qa_id,
                image=candidate.artifact.data,
                prompt=_qa_prompt(commands[item_id], plan),
                profile=profile,
                rubric=_qa_rubric(profile, rubrics[profile]),
                expected_subject_count=1,
            ))
    if not requests:
        return BatchVisionQualityResult(tuple()), by_qa_id, detector_rejects
    print("AI_AGENT_PROGRESS " + json.dumps({"step": "qa_start", "suffix": suffix, "candidate_count": len(requests)}, separators=(",", ":")), flush=True)
    verifier = FailClosedVisionVerifier(
        worker=worker,
        model="Qwen/Qwen3-VL-8B-Instruct",
        kernel_slug=f"ai-agent-production-image-8b-{suffix}",
        poll_interval=3,
        max_poll_attempts=300,
        min_quality_score=9.0,
        min_prompt_match_score=9.0,
    )
    result = verifier.verify_many(tuple(requests))
    print("AI_AGENT_PROGRESS " + json.dumps({"step": "qa_complete", "suffix": suffix, "candidate_count": len(requests)}, separators=(",", ":")), flush=True)
    return result, by_qa_id, detector_rejects


def _score_floor(result: VisionQualityResult) -> float:
    return min(
        result.quality_score, result.prompt_match_score, result.structure_score,
        result.detail_score, result.aesthetic_score, result.composition_score,
        result.benchmark_match_score,
    )


def _group(qa, by_qa_id):
    grouped: dict[str, list[tuple[ProductionImageCandidate, VisionQualityResult]]] = {}
    for result in qa.items:
        candidate = by_qa_id[result.item_id]
        grouped.setdefault(candidate.item_id, []).append((candidate, result))
    return grouped


def _select(grouped):
    selected: dict[str, ProductionImageCandidate] = {}
    for item_id, entries in grouped.items():
        passing = [entry for entry in entries if entry[1].passed]
        if passing:
            selected[item_id] = max(passing, key=lambda entry: (_score_floor(entry[1]), -entry[0].candidate_index))[0]
    return selected


def _review_json(result):
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


def main() -> int:
    raw_items = _load_items()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token or not username:
        raise RuntimeError("KAGGLE_API_TOKEN and KAGGLE_USERNAME are required")
    out = Path(os.environ.get("MEDIA_OUTPUT_DIR", "production-image-output"))
    out.mkdir(parents=True, exist_ok=True)

    positive, negative, dimensions, rubrics, engines = _maps()
    commands = {item["id"]: item["command"] for item in raw_items}
    fallback_plans = MediaCommandPlanner().plan_many(tuple(item["command"] for item in raw_items))
    production_items = tuple(
        ProductionImageItem(
            item_id=item["id"], command=item["command"],
            seed=int.from_bytes(sha256(item["command"].encode("utf-8")).digest()[:4], "big"),
            fallback_profile=plan.profile.name, hard_gates=item["hard_gates"],
        )
        for item, plan in zip(raw_items, fallback_plans, strict=True)
    )

    worker = KaggleGpuWorker(
        api_token=token, username=username, timeout=120,
        submission_retry_attempts=5, submission_retry_delay_seconds=30,
    )
    provider = KaggleProductionImageBatchProvider(
        worker=worker,
        profile_engines=engines,
        profile_positive_constraints=positive,
        profile_negative_constraints=negative,
        profile_dimensions=dimensions,
        kernel_slug="ai-agent-production-image-standard",
        poll_interval=3,
        max_poll_attempts=int(os.environ.get("MEDIA_IMAGE_MAX_POLL_ATTEMPTS", "700")),
        planner_model=os.environ.get("MEDIA_COMMAND_MODEL", MEDIA_COMMAND_BRAIN_MODEL),
    )

    started = perf_counter()
    generated = provider.generate_batch(production_items)
    qa, qa_candidates, detector_rejects = _run_qa(
        worker, generated.candidates, generated.plans, commands, rubrics, "initial"
    )
    grouped = _group(qa, qa_candidates)
    selected = _select(grouped)

    retry_ids = [item.item_id for item in production_items if item.item_id not in selected]
    retry_generated = None
    retry_grouped = {}
    retry_detector_rejects = {}
    if retry_ids:
        retry_items = []
        by_id = {item.item_id: item for item in production_items}
        for item_id in retry_ids:
            issues = list(detector_rejects.get(item_id, []))
            entries = grouped.get(item_id, [])
            if entries:
                _, best = max(entries, key=lambda entry: _score_floor(entry[1]))
                issues.extend(best.major_issues[:5])
            source = by_id[item_id]
            retry_items.append(ProductionImageItem(
                item_id=item_id,
                command=(
                    source.command
                    + " Correction retry: preserve the exact original species/entity and style. Fix only these observed failures: "
                    + ("; ".join(dict.fromkeys(issues)) or "strict QA rejection")
                    + ". No text, tags, logos or branding. Do not generalize the subject."
                ),
                seed=source.seed + 1000,
                fallback_profile=source.fallback_profile,
                hard_gates=source.hard_gates,
            ))
        print("AI_AGENT_PROGRESS " + json.dumps({"step": "retry_start", "item_ids": retry_ids}, separators=(",", ":")), flush=True)
        retry_generated = provider.generate_batch(tuple(retry_items))
        retry_qa, retry_candidates, retry_detector_rejects = _run_qa(
            worker, retry_generated.candidates, retry_generated.plans, commands, rubrics, "retry"
        )
        retry_grouped = _group(retry_qa, retry_candidates)
        selected.update(_select(retry_grouped))

    statuses = {}
    for item in production_items:
        item_dir = out / item.item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        plan = retry_generated.plans.get(item.item_id) if retry_generated and item.item_id in retry_generated.plans else generated.plans[item.item_id]
        (item_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        all_entries = [*grouped.get(item.item_id, []), *retry_grouped.get(item.item_id, [])]
        dump = []
        for index, (candidate, review) in enumerate(all_entries):
            filename = f"candidate-{index}.png"
            (item_dir / filename).write_bytes(candidate.artifact.data)
            dump.append({
                "candidate_file": filename,
                "candidate_index": candidate.candidate_index,
                "model": candidate.artifact.model,
                "detector_gate": candidate.report.get("detector_gate"),
                "image_sha256": sha256(candidate.artifact.data).hexdigest(),
                "review": _review_json(review),
            })
        (item_dir / "qa-candidates.json").write_text(json.dumps(dump, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        final = selected.get(item.item_id)
        if final is None:
            issues = [*detector_rejects.get(item.item_id, []), *retry_detector_rejects.get(item.item_id, [])]
            for _, review in all_entries:
                issues.extend(review.major_issues)
            statuses[item.item_id] = {
                "verified": False,
                "stage": "detector_or_8b_quality",
                "error": "; ".join(dict.fromkeys(issues)) or "no candidate passed hard gates and 8B QA",
            }
            continue
        (item_dir / "image.png").write_bytes(final.artifact.data)
        statuses[item.item_id] = {
            "verified": True,
            "stage": "complete",
            "model": final.artifact.model,
            "detector_gate": final.report.get("detector_gate"),
            "image_sha256": sha256(final.artifact.data).hexdigest(),
        }

    elapsed = round(perf_counter() - started, 3)
    summary = {
        "batch_size": len(production_items),
        "verified_count": sum(1 for value in statuses.values() if value.get("verified")),
        "failed_count": sum(1 for value in statuses.values() if not value.get("verified")),
        "semantic_router": True,
        "planner_model": MEDIA_COMMAND_BRAIN_MODEL,
        "style_specialized_engines": True,
        "structured_progress": True,
        "detector_hard_gate": True,
        "qa_uses_original_user_request": True,
        "retry_count": len(retry_ids),
        "retry_ids": retry_ids,
        "elapsed_seconds": elapsed,
        "initial_worker_report": generated.batch_report,
        "retry_worker_report": retry_generated.batch_report if retry_generated else None,
        "statuses": statuses,
    }
    (out / "batch-status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PRODUCTION_IMAGE_BATCH_DONE")
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
