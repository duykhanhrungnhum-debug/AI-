"""Temporary production benchmark for the subject-identity lock. Remove after verification."""
from __future__ import annotations

import threading
import time

from ai_agent.executors.image import ImageExecutor

_LOCK = threading.Lock()
_STATE: dict = {"status": "idle", "images": []}

COMMAND = (
    "Tạo hai ảnh riêng: ảnh thứ nhất là một con cua và một con ếch thật, phong cách ảnh chụp chân thực, "
    "cả hai cùng xuất hiện rõ trong một khung hình; ảnh thứ hai là một con cua và một con ếch hoạt hình 3D, "
    "cả hai cùng xuất hiện rõ trong một khung hình."
)


def snapshot() -> dict:
    with _LOCK:
        out = {k: v for k, v in _STATE.items() if k != "images"}
        out["image_count"] = len(_STATE.get("images", []))
        return out


def get_image(index: int) -> bytes:
    with _LOCK:
        return _STATE.get("images", [])[index]


def run() -> None:
    started = time.perf_counter()
    with _LOCK:
        _STATE.clear()
        _STATE.update({"status": "running", "images": []})
    try:
        results = ImageExecutor(kernel_slug="ai-agent-image-v2").execute_command_cold(
            COMMAND, width=1024, height=1024, max_images=2
        )
        if len(results) != 2:
            raise RuntimeError(f"expected 2 images, got {len(results)}")
        evidence = []
        images = []
        for i, result in enumerate(results, 1):
            data = result.data
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError(f"image {i} is not PNG")
            evidence.append({
                "index": i,
                "prompt": result.prompt,
                "generation_seconds": round(float(result.elapsed_seconds), 3),
                "bytes": len(data),
                "width": int(result.width),
                "height": int(result.height),
                "model": result.model,
            })
            images.append(data)
        with _LOCK:
            _STATE.clear()
            _STATE.update({
                "status": "success",
                "command": COMMAND,
                "total_seconds": round(time.perf_counter() - started, 3),
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
