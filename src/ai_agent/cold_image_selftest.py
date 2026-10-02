"""One-shot production gate for the simple Cold Image V2 path.

Temporary verification harness: one real three-image natural-language run covers
portrait, semantic batching, photoreal and 3D output in a single GPU session.
It intentionally uses only the Python standard library.
"""
from __future__ import annotations

import json
import struct
import time
import zlib

from ai_agent.executors.image import ImageExecutor


COMMAND = (
    "Tạo ba ảnh riêng: "
    "ảnh thứ nhất là chân dung một phụ nữ Á Đông trưởng thành 25 tuổi, xinh đẹp, trẻ trung, "
    "phong cách ảnh chụp chân thực, ánh sáng tự nhiên; "
    "ảnh thứ hai là một con trâu nước Việt Nam chân thực đang đứng ngoài đồng lúa; "
    "ảnh thứ ba là một con trâu nước Việt Nam phong cách hoạt hình 3D cute, toàn thân."
)


def _validate_png(data: bytes, *, index: int) -> dict:
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise RuntimeError(f"image {index} is not PNG")
    if len(data) < 20_000:
        raise RuntimeError(f"image {index} is unexpectedly small: {len(data)} bytes")
    if len(data) < 33 or data[12:16] != b"IHDR":
        raise RuntimeError(f"image {index} has no valid IHDR")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (1024, 1024):
        raise RuntimeError(f"image {index} has wrong size: {(width, height)}")

    offset = 8
    idat = bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
        if len(chunk_data) != length:
            raise RuntimeError(f"image {index} has truncated PNG chunk")
        if chunk_type == b"IDAT":
            idat.extend(chunk_data)
        offset += 12 + length
        if chunk_type == b"IEND":
            break
    if not idat:
        raise RuntimeError(f"image {index} has no IDAT payload")
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise RuntimeError(f"image {index} has invalid compressed pixels: {exc}") from exc
    # A valid 1024px generated image should have substantial scanline data and
    # many byte values. This cheaply catches blank/corrupt placeholder outputs.
    if len(raw) < 1_000_000 or len(set(raw[::max(1, len(raw) // 100_000)])) < 16:
        raise RuntimeError(f"image {index} appears blank or corrupt")
    return {
        "bytes": len(data),
        "size": [width, height],
        "sampled_byte_values": len(set(raw[::max(1, len(raw) // 100_000)])),
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
