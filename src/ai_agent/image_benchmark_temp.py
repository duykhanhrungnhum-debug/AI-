"""Temporary two-image benchmark for AIKA production Image V2.

This module is intentionally temporary and is removed after the benchmark run.
"""
from __future__ import annotations

import threading
import time

from ai_agent.executors.image import ImageExecutor


_LOCK = threading.Lock()
_STATE: dict = {"status": "idle", "images": []}

COMMAND = (
    "Tạo hai ảnh riêng: "
    "ảnh thứ nhất là một con cua và một con ếch thật, phong cách ảnh chụp chân thực, "
    "cả hai con cùng xuất hiện rõ ràng trong cùng một khung hình; "
    "ảnh thứ hai là một con cua và một con ếch hoạt hình 3D, "
    "cả hai con cùng xuất hiện rõ ràng trong cùng một khung hình."
)


def snapshot() -> dict:
    with _LOCK:
        public = {key: value for key, value in _STATE.items() if key != "images"}
        public["image_count"] = len(_STATE.get("images", []))
        return public


def get_image(index: int) -> bytes:
    with _LOCK:
        images = _STATE.get("images", [])
        return images[index]


def run() -> None:
    started = time.perf_counter()
    with _LOCK:
        _STATE.clear()
        _STATE.update({"status": "running", "started_monotonic": started, "images": []})
    try:
        results = ImageExecutor(kernel_slug="ai-agent-image-v2").execute_command_cold(
            COMMAND,
            width=1024,
            height=1024,
            max_images=2,
        )
        if len(results) != 2:
            raise RuntimeError(f"expected 2 images, got {len(results)}")

        evidence = []
        images = []
        for index, result in enumerate(results, 1):
            data = result.data
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError(f"image {index} is not PNG")
            if len(data) < 20_000:
                raise RuntimeError(f"image {index} is unexpectedly small: {len(data)} bytes")
            evidence.append({
                "index": index,
                "prompt": result.prompt,
                "model": result.model,
                "generation_seconds": round(float(result.elapsed_seconds), 3),
                "bytes": len(data),
                "width": int(result.width),
                "height": int(result.height),
            })
            images.append(data)

        total = round(time.perf_counter() - started, 3)
        generation_sum = round(sum(item["generation_seconds"] for item in evidence), 3)
        with _LOCK:
            _STATE.clear()
            _STATE.update({
                "status": "success",
                "command": COMMAND,
                "total_seconds": total,
                "generation_sum_seconds": generation_sum,
                "overhead_seconds": round(max(0.0, total - generation_sum), 3),
                "evidence": evidence,
                "images": images,
            })
    except Exception as exc:
        with _LOCK:
            _STATE.clear()
            _STATE.update({
                "status": "failure",
                "error": f"{type(exc).__name__}: {exc}"[:4000],
                "total_seconds": round(time.perf_counter() - started, 3),
                "images": [],
            })
        raise
