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
from .core.kaggle_worker import KaggleGpuWorker


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _emit(marker: str, payload: dict) -> None:
    print(marker + " " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _diagnostic_worker() -> KaggleGpuWorker:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token or not username:
        raise RuntimeError("Kaggle canary diagnostics require configured credentials")
    return KaggleGpuWorker(
        api_token=token,
        username=username,
        timeout=60,
        submission_retry_attempts=1,
        submission_retry_delay_seconds=0,
    )


def _kernel_status(worker: KaggleGpuWorker) -> dict:
    try:
        status = worker.status("ai-agent-image-warm")
        return {
            "status": status.status,
            "terminal": status.terminal,
            "successful": status.successful,
            "failure_message": status.failure_message[-1000:],
        }
    except Exception as exc:
        return {"status_error": f"{type(exc).__name__}: {exc}"[:1200]}


def _kernel_log_tail(worker: KaggleGpuWorker, limit: int = 5000) -> str:
    try:
        return worker.logs("ai-agent-image-warm")[-limit:]
    except Exception as exc:
        return f"LOG_READ_FAILED {type(exc).__name__}: {exc}"[:1200]


def _launch_only_diagnostic(worker: KaggleGpuWorker, *, timeout_seconds: float = 420) -> None:
    """Verify only worker submission/heartbeat; never create an image job."""
    before = CHAT_BROKER._warm_image.snapshot()
    started = CHAT_BROKER._warm_image.ensure_started()
    _emit("AIKA_WARM_LAUNCH_DIAG_BEGIN", {
        "ensure_started": started,
        "before_state": before.state,
        "before_stage": before.stage,
        "before_launch_count": before.launch_count,
        "kernel": _kernel_status(worker),
    })
    if not started:
        raise RuntimeError("warm manager refused to start")

    deadline = time.time() + timeout_seconds
    next_report = 0.0
    while time.time() < deadline:
        snapshot = CHAT_BROKER._warm_image.snapshot()
        error = CHAT_BROKER._warm_image.last_error()
        if error:
            raise RuntimeError("warm launch failed: " + error)
        if snapshot.state in {"ready", "idle", "busy"} and snapshot.last_seen:
            _emit("AIKA_WARM_LAUNCH_DIAG_SUCCESS", {
                "state": snapshot.state,
                "stage": snapshot.stage,
                "launch_count": snapshot.launch_count,
                "last_seen": snapshot.last_seen,
                "session_id": snapshot.session_id,
            })
            return
        if time.time() >= next_report:
            _emit("AIKA_WARM_LAUNCH_DIAG_WAIT", {
                "state": snapshot.state,
                "stage": snapshot.stage,
                "launch_count": snapshot.launch_count,
                "last_seen": snapshot.last_seen,
                "kernel": _kernel_status(worker),
            })
            next_report = time.time() + 15
        time.sleep(2)

    raise TimeoutError(
        "warm launch produced no ready heartbeat; stage="
        + CHAT_BROKER._warm_image.snapshot().stage
        + "; kernel=" + json.dumps(_kernel_status(worker), ensure_ascii=False)
        + "; log_tail=" + _kernel_log_tail(worker)
    )


def _wait_existing_kernel(worker: KaggleGpuWorker, *, timeout_seconds: float = 240) -> None:
    status = _kernel_status(worker)
    if status.get("terminal") is not False:
        _emit("AIKA_WARM_CANARY_KERNEL_PRECHECK", status)
        return

    _emit("AIKA_WARM_CANARY_EXISTING_KERNEL", status)
    started = time.time()
    next_report = started
    while time.time() - started < timeout_seconds:
        snapshot = CHAT_BROKER._warm_image.snapshot()
        if snapshot.state in {"ready", "idle", "busy"} and snapshot.last_seen:
            _emit("AIKA_WARM_CANARY_EXISTING_REUSED", {
                "state": snapshot.state,
                "stage": snapshot.stage,
                "launch_count": snapshot.launch_count,
                "session_id": snapshot.session_id,
            })
            return
        status = _kernel_status(worker)
        if status.get("terminal") is True:
            _emit("AIKA_WARM_CANARY_EXISTING_TERMINAL", status)
            return
        if time.time() >= next_report:
            _emit("AIKA_WARM_CANARY_EXISTING_WAIT", {
                "manager_state": snapshot.state,
                "manager_stage": snapshot.stage,
                "last_seen": snapshot.last_seen,
                "launch_count": snapshot.launch_count,
                "manager_error": CHAT_BROKER._warm_image.last_error(),
                "kernel": status,
            })
            next_report = time.time() + 30
        time.sleep(5)

    raise RuntimeError(
        "existing ai-agent-image-warm kernel stayed non-terminal without becoming ready; log_tail="
        + _kernel_log_tail(worker)
    )


def _wait_job(job_id: str, *, timeout_seconds: float, worker: KaggleGpuWorker, label: str) -> dict:
    deadline = time.time() + timeout_seconds
    next_report = time.time() + 30
    while time.time() < deadline:
        state = CHAT_BROKER.get_job(job_id)
        if state["status"] == "done":
            return state
        if state["status"] == "error":
            raise RuntimeError(state.get("error") or state.get("worker_error") or "job failed")
        if time.time() >= next_report:
            snapshot = CHAT_BROKER._warm_image.snapshot()
            _emit("AIKA_WARM_CANARY_JOB_WAIT", {
                "label": label,
                "job_status": state.get("status"),
                "worker_state": state.get("worker_state"),
                "worker_error": state.get("worker_error"),
                "manager_state": snapshot.state,
                "manager_stage": snapshot.stage,
                "manager_error": CHAT_BROKER._warm_image.last_error(),
                "launch_count": snapshot.launch_count,
                "session_id": snapshot.session_id,
                "current_job_id": snapshot.current_job_id,
                "lease_remaining": round(max(0.0, snapshot.job_lease_expires_at - time.time()), 1),
                "last_seen_age": round(time.time() - snapshot.last_seen, 1) if snapshot.last_seen else None,
                "kernel": _kernel_status(worker),
            })
            next_report = time.time() + 30
        time.sleep(2)
    raise TimeoutError(
        f"job {job_id} exceeded {timeout_seconds:.0f}s; stage={CHAT_BROKER._warm_image.snapshot().stage}; "
        f"kernel={_kernel_status(worker)}; log_tail={_kernel_log_tail(worker)}"
    )


def _run_image(label: str, message: str, *, timeout_seconds: float, worker: KaggleGpuWorker) -> dict:
    started = time.perf_counter()
    job = CHAT_BROKER.create_job(message, message=message)
    _emit("AIKA_WARM_CANARY_JOB_CREATED", {"label": label, "job_id": job.job_id})
    state = _wait_job(job.job_id, timeout_seconds=timeout_seconds, worker=worker, label=label)
    result = {
        "label": label,
        "job_id": job.job_id,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "provider": state.get("provider"),
        "model": state.get("model"),
        "image_count": state.get("image_count"),
        "generation_seconds": state.get("elapsed_seconds"),
        "worker_state": state.get("worker_state"),
        "worker_session_id": state.get("worker_session_id"),
        "warm_image_enabled": state.get("warm_image_enabled"),
    }
    _emit("AIKA_WARM_CANARY_JOB_DONE", result)
    return result


def _require_warm(result: dict) -> None:
    if result["provider"] != "kaggle-image-warm":
        raise RuntimeError(
            f"{result['label']} used {result['provider']!r}, expected kaggle-image-warm; stop canary before more GPU work"
        )
    if result["warm_image_enabled"] is not True:
        raise RuntimeError(f"{result['label']} did not report warm_image_enabled=true")


def run_warm_canary_selftest() -> None:
    if not _flag("AIKA_WARM_CANARY_SELFTEST"):
        return

    time.sleep(3)
    try:
        if not CHAT_BROKER._warm_image.enabled:
            raise RuntimeError("AIKA_IMAGE_WARM_WORKER is not enabled on canary")

        worker = _diagnostic_worker()
        _emit("AIKA_WARM_CANARY_BEGIN", {
            "warm_enabled": True,
            "diagnostic_only": _flag("AIKA_WARM_CANARY_DIAGNOSTIC_ONLY"),
            "kernel": _kernel_status(worker),
        })

        if _flag("AIKA_WARM_CANARY_DIAGNOSTIC_ONLY"):
            _launch_only_diagnostic(worker)
            return

        _wait_existing_kernel(worker)
        before = CHAT_BROKER._warm_image.snapshot()
        first = _run_image(
            "first",
            "AIKA tạo một ảnh chân thực quả táo đỏ trên bàn gỗ, nền sạch.",
            timeout_seconds=720,
            worker=worker,
        )
        _require_warm(first)
        after_first = CHAT_BROKER._warm_image.snapshot()

        second = _run_image(
            "second",
            "AIKA tạo một ảnh chân thực quả cam tươi trên bàn gỗ, nền sạch.",
            timeout_seconds=240,
            worker=worker,
        )
        _require_warm(second)
        after_second = CHAT_BROKER._warm_image.snapshot()

        if after_second.launch_count != after_first.launch_count:
            raise RuntimeError(
                f"second job relaunched warm worker: {after_first.launch_count} -> {after_second.launch_count}"
            )
        if second.get("worker_session_id") != first.get("worker_session_id"):
            raise RuntimeError("second job did not reuse the first worker session")
        if float(second["wall_seconds"]) >= 150:
            raise RuntimeError(f"second warm job took too long: {second['wall_seconds']}s")

        batch = _run_image(
            "batch",
            "AIKA tạo hai ảnh riêng biệt để kiểm tra: ảnh thứ nhất là một bông sen hồng chân thực trên mặt nước; "
            "ảnh thứ hai là một bông hướng dương chân thực ngoài đồng. Đây là hai file ảnh riêng, không ghép chung.",
            timeout_seconds=300,
            worker=worker,
        )
        _require_warm(batch)
        after_batch = CHAT_BROKER._warm_image.snapshot()

        if int(first.get("image_count") or 0) != 1:
            raise RuntimeError(f"first expected 1 image, got {first.get('image_count')}")
        if int(second.get("image_count") or 0) != 1:
            raise RuntimeError(f"second expected 1 image, got {second.get('image_count')}")
        if int(batch.get("image_count") or 0) != 2:
            raise RuntimeError(f"batch expected 2 images, got {batch.get('image_count')}")

        if after_first.launch_count <= before.launch_count:
            raise RuntimeError("first job did not launch the warm worker")
        if after_batch.launch_count != after_first.launch_count:
            raise RuntimeError(
                f"batch job relaunched warm worker: {after_first.launch_count} -> {after_batch.launch_count}"
            )
        if batch.get("worker_session_id") != first.get("worker_session_id"):
            raise RuntimeError("batch job did not reuse the first worker session")

        report = {
            "status": "success",
            "session_id": first.get("worker_session_id"),
            "launch_count_before": before.launch_count,
            "launch_count_after_first": after_first.launch_count,
            "launch_count_after_second": after_second.launch_count,
            "launch_count_after_batch": after_batch.launch_count,
            "first": first,
            "second": second,
            "batch": batch,
        }
        _emit("AIKA_WARM_CANARY_SUCCESS", report)
    except Exception as exc:
        report = {
            "status": "failure",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _emit("AIKA_WARM_CANARY_FAILURE", report)
        traceback.print_exc()
