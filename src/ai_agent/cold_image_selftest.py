"""One-shot production gate for the simple Cold Image V2 path.

This module is intentionally a temporary verification harness. It exercises the
real Kaggle provider once with a three-image natural-language command so portrait,
semantic batching, photoreal and 3D paths are verified in one GPU session.
"""
from __future__ import annotations

from io import BytesIO
import json
import time

from PIL import Image, ImageStat

from ai_agent.executors.image import ImageExecutor


COMMAND = (
    "Tạo ba ảnh riêng: "
    "ảnh thứ nhất là chân dung một phụ nữ Á Đông trưởng thành 25 tuổi, xinh đẹp, trẻ trung, "
    "phong cách ảnh chụp chân thực, ánh sáng tự nhiên; "
    "ảnh thứ hai là một con trâu nước Việt Nam chân thực đang đứng ngoài đồng lúa; "
    "ảnh thứ ba là một con trâu nước Việt Nam phong cách hoạt hình 3D cute, toàn thân."
)


def _validate_png(data: bytes, *, index: int) -> dict:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"image {index} is not PNG")
    if len(data) < 20_000:
        raise RuntimeError(f"image {index} is unexpectedly small: {len(data)} bytes")
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.size != (1024, 1024):
            raise RuntimeError(f"image {index} has wrong size: {image.size}")
        rgb = image.convert("RGB").resize((64, 64))
        stat = ImageStat.Stat(rgb)
        spread = sum(stat.var) / 3.0
        if spread < 25.0:
            raise RuntimeError(f"image {index} appears blank/near-uniform: variance={spread:.2f}")
        return {
            "bytes": len(data),
            "size": list(image.size),
            "variance": round(spread, 2),
        }


def run_cold_image_selftest() -> None:
    started = time.perf_counter()
    print("AIKA_COLD_E2E_BEGIN " + json.dumps({"expected_images": 3}, ensure_ascii=False), flush=True)
    try:
        results = ImageExecutor(kernel_slug="ai-agent-image-v2").execute_command_cold(
            COMMAND,
            width=1024,
            height=1024,
            max_images=3,
        )
        if len(results) != 3:
            raise RuntimeError(f"expected 3 images, got {len(results)}")

        prompt_requirements = (
            ("asian", "woman"),
            ("buffalo",),
            ("buffalo", "3d"),
        )
        evidence = []
        for index, (result, required) in enumerate(zip(results, prompt_requirements), 1):
            prompt_folded = result.prompt.casefold()
            missing = [token for token in required if token not in prompt_folded]
            if missing:
                raise RuntimeError(
                    f"image {index} prompt lost required semantics {missing}: {result.prompt[:500]}"
                )
            file_check = _validate_png(result.data, index=index)
            evidence.append({
                "index": index,
                "model": result.model,
                "prompt": result.prompt[:500],
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                **file_check,
            })

        print("AIKA_COLD_E2E_SUCCESS " + json.dumps({
            "status": "success",
            "image_count": len(results),
            "elapsed_total_seconds": round(time.perf_counter() - started, 3),
            "evidence": evidence,
        }, ensure_ascii=False), flush=True)
    except Exception as exc:
        print("AIKA_COLD_E2E_FAILURE " + json.dumps({
            "status": "failure",
            "error": f"{type(exc).__name__}: {exc}"[:4000],
            "elapsed_total_seconds": round(time.perf_counter() - started, 3),
        }, ensure_ascii=False), flush=True)
        raise
