"""Fail-fast warm-image soak validation for the isolated canary service."""
from __future__ import annotations

import json
import os
import time
import traceback

from .canary_selftest import _diagnostic_worker, _require_warm, _run_image
from .chat_session import CHAT_BROKER


def _emit(marker: str, payload: dict) -> None:
    print(marker + " " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def run_warm_soak() -> None:
    try:
        requested = int(os.environ.get("AIKA_WARM_SOAK_COUNT", "0"))
    except ValueError:
        requested = 0
    if requested <= 0:
        return
    count = max(1, min(requested, 20))

    time.sleep(3)
    try:
        if not CHAT_BROKER._warm_image.enabled:
            raise RuntimeError("AIKA_IMAGE_WARM_WORKER is not enabled on soak canary")

        worker = _diagnostic_worker()
        first_session = ""
        first_launch_count = 0
        wall_times: list[float] = []
        generation_times: list[float] = []

        _emit("AIKA_WARM_SOAK_BEGIN", {
            "count": count,
            "kernel_slug": CHAT_BROKER._warm_image.kernel_slug,
            "cold_fallback_allowed": False,
        })

        subjects = [
            "quả táo đỏ", "quả cam tươi", "quả lê xanh", "bông sen hồng",
            "bông hướng dương", "chiếc cốc gốm trắng", "quyển sách bìa xanh",
            "chiếc đồng hồ để bàn", "cây xương rồng nhỏ", "chiếc đèn bàn",
        ]
        for index in range(1, count + 1):
            subject = subjects[(index - 1) % len(subjects)]
            result = _run_image(
                f"soak-{index:02d}",
                f"AIKA tạo một ảnh chân thực {subject} trên nền sạch, ánh sáng tự nhiên.",
                timeout_seconds=720 if index == 1 else 180,
                worker=worker,
            )
            _require_warm(result)
            if int(result.get("image_count") or 0) != 1:
                raise RuntimeError(
                    f"soak-{index:02d} expected 1 image, got {result.get('image_count')}"
                )

            snapshot = CHAT_BROKER._warm_image.snapshot()
            session = str(result.get("worker_session_id") or "")
            if index == 1:
                first_session = session
                first_launch_count = snapshot.launch_count
                if not first_session or first_launch_count != 1:
                    raise RuntimeError(
                        f"invalid first soak session={first_session!r} launch_count={first_launch_count}"
                    )
            else:
                if session != first_session:
                    raise RuntimeError(
                        f"soak-{index:02d} changed session: {first_session} -> {session}"
                    )
                if snapshot.launch_count != first_launch_count:
                    raise RuntimeError(
                        f"soak-{index:02d} relaunched worker: {first_launch_count} -> {snapshot.launch_count}"
                    )
                if float(result["wall_seconds"]) >= 150:
                    raise RuntimeError(
                        f"soak-{index:02d} took too long: {result['wall_seconds']}s"
                    )

            wall_times.append(float(result["wall_seconds"]))
            generation_times.append(float(result.get("generation_seconds") or 0.0))
            _emit("AIKA_WARM_SOAK_PROGRESS", {
                "completed": index,
                "count": count,
                "session_id": first_session,
                "launch_count": snapshot.launch_count,
                "wall_seconds": result["wall_seconds"],
                "generation_seconds": result.get("generation_seconds"),
            })

        _emit("AIKA_WARM_SOAK_SUCCESS", {
            "count": count,
            "session_id": first_session,
            "launch_count": first_launch_count,
            "avg_warm_wall_seconds": round(sum(wall_times[1:]) / max(1, len(wall_times) - 1), 3),
            "max_warm_wall_seconds": round(max(wall_times[1:] or wall_times), 3),
            "avg_generation_seconds": round(sum(generation_times) / len(generation_times), 3),
        })
    except Exception as exc:
        _emit("AIKA_WARM_SOAK_FAILURE", {
            "error": f"{type(exc).__name__}: {exc}",
        })
        traceback.print_exc()
