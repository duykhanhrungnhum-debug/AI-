"""Isolated runtime canary for AIKA warm image execution.

Enabled only when AIKA_WARM_CANARY_SELFTEST=true. This is never part of the
normal user request path and exists only to prove warm reuse + multi-image batch
on a real deployed canary service.
"""
from __future__ import annotations

import json
import os
import time
import traceback

from .chat_session import CHAT_BROKER


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _wait_job(job_id: str, *, timeout_seconds: float) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        state = CHAT_BROKER.get_job(job_id)
        if state["status"] == "done":
            return state
        if state["status"] == "error":
            raise RuntimeError(state.get("error") or state.get("worker_error") or "job failed")
        time.sleep(2)
    raise TimeoutError(f"job {job_id} exceeded {timeout_seconds:.0f}s")


def _run_image(label: str, message: str, *, timeout_seconds: float) -> dict:
    started = time.perf_counter()
    job = CHAT_BROKER.create_job(message, message=message)
    state = _wait_job(job.job_id, timeout_seconds=timeout_seconds)
    return {
        "label": label,
        "job_id": job.job_id,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "provider": state.get("provider"),
        "model": state.get("model"),
        "image_count": state.get("image_count"),
        "generation_seconds": state.get("elapsed_seconds"),
        "worker_state": state.get("worker_state"),
        "warm_image_enabled": state.get("warm_image_enabled"),
    }


def run_warm_canary_selftest() -> None:
    if not _flag("AIKA_WARM_CANARY_SELFTEST"):
        return

    # Let the HTTP server enter serve_forever before Kaggle starts calling back.
    time.sleep(3)
    try:
        if not CHAT_BROKER._warm_image.enabled:
            raise RuntimeError("AIKA_IMAGE_WARM_WORKER is not enabled on canary")

        before = CHAT_BROKER._warm_image.snapshot()
        first = _run_image(
            "first",
            "AIKA tạo một ảnh chân thực quả táo đỏ trên bàn gỗ, nền sạch.",
            timeout_seconds=720,
        )
        after_first = CHAT_BROKER._warm_image.snapshot()

        second = _run_image(
            "second",
            "AIKA tạo một ảnh chân thực quả cam tươi trên bàn gỗ, nền sạch.",
            timeout_seconds=240,
        )
        after_second = CHAT_BROKER._warm_image.snapshot()

        batch = _run_image(
            "batch",
            "AIKA tạo hai ảnh riêng biệt để kiểm tra: ảnh thứ nhất là một bông sen hồng chân thực trên mặt nước; "
            "ảnh thứ hai là một bông hướng dương chân thực ngoài đồng. Đây là hai file ảnh riêng, không ghép chung.",
            timeout_seconds=300,
        )
        after_batch = CHAT_BROKER._warm_image.snapshot()

        for result in (first, second, batch):
            if result["provider"] != "kaggle-image-warm":
                raise RuntimeError(f"{result['label']} used {result['provider']!r}, expected kaggle-image-warm")
            if result["warm_image_enabled"] is not True:
                raise RuntimeError(f"{result['label']} did not report warm_image_enabled=true")

        if int(first.get("image_count") or 0) != 1:
            raise RuntimeError(f"first expected 1 image, got {first.get('image_count')}")
        if int(second.get("image_count") or 0) != 1:
            raise RuntimeError(f"second expected 1 image, got {second.get('image_count')}")
        if int(batch.get("image_count") or 0) != 2:
            raise RuntimeError(f"batch expected 2 images, got {batch.get('image_count')}")

        if after_first.launch_count <= before.launch_count:
            raise RuntimeError("first job did not launch the warm worker")
        if after_second.launch_count != after_first.launch_count:
            raise RuntimeError(
                f"second job relaunched warm worker: {after_first.launch_count} -> {after_second.launch_count}"
            )
        if after_batch.launch_count != after_first.launch_count:
            raise RuntimeError(
                f"batch job relaunched warm worker: {after_first.launch_count} -> {after_batch.launch_count}"
            )
        if float(second["wall_seconds"]) >= 150:
            raise RuntimeError(f"second warm job took too long: {second['wall_seconds']}s")

        report = {
            "status": "success",
            "launch_count_before": before.launch_count,
            "launch_count_after_first": after_first.launch_count,
            "launch_count_after_second": after_second.launch_count,
            "launch_count_after_batch": after_batch.launch_count,
            "first": first,
            "second": second,
            "batch": batch,
        }
        print("AIKA_WARM_CANARY_SUCCESS " + json.dumps(report, ensure_ascii=False, separators=(",", ":")), flush=True)
    except Exception as exc:
        report = {
            "status": "failure",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print("AIKA_WARM_CANARY_FAILURE " + json.dumps(report, ensure_ascii=False, separators=(",", ":")), flush=True)
        traceback.print_exc()
