"""One-shot Railway-local E2E client for AIKA warm image canary.

Runs outside the production service, authenticates through a Railway variable
reference, proves warm reuse by timing a second request, and verifies that one
natural command can return two separate PNG files from one job.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


def main() -> None:
    base = os.environ.get("AIKA_CANARY_URL", "").strip().rstrip("/")
    token = os.environ.get("AIKA_CANARY_TOKEN", "").strip()
    if not base:
        raise SystemExit("AIKA_CANARY_URL is required")
    if not token:
        raise SystemExit("AIKA_CANARY_TOKEN is required")

    out = Path(os.environ.get("AIKA_CANARY_OUTPUT", "/tmp/aika-warm-canary"))
    out.mkdir(parents=True, exist_ok=True)

    def request(method: str, path: str, payload: dict | None = None, *, timeout: int = 180):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            base + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "AIKA-Railway-Warm-Canary/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            if "application/json" in response.headers.get("content-type", ""):
                return json.loads(raw.decode("utf-8")) if raw else None
            return raw

    def wait_health() -> None:
        last_error = ""
        for _ in range(60):
            try:
                req = urllib.request.Request(base + "/health", headers={"User-Agent": "AIKA-Railway-Warm-Canary/1.0"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        return
            except Exception as exc:  # pragma: no cover - real E2E only
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
        raise RuntimeError("canary health did not become ready: " + last_error)

    def run_image(label: str, message: str) -> dict:
        wall_started = time.perf_counter()
        created = request("POST", "/v1/chat", {"message": message, "prompt": message})
        job_id = str(created.get("job_id") or "")
        if not job_id:
            raise RuntimeError(label + " did not return job_id")

        final = None
        for _ in range(220):
            time.sleep(3)
            final = request("GET", "/v1/chat/status?job_id=" + urllib.parse.quote(job_id))
            if final.get("status") == "done":
                break
            if final.get("status") == "error":
                raise RuntimeError(label + " failed: " + str(final.get("error") or final.get("worker_error")))
        if not final or final.get("status") != "done":
            raise TimeoutError(label + " did not complete")

        image_urls = final.get("image_urls") or ([final.get("image_url")] if final.get("image_url") else [])
        for index, url in enumerate(image_urls):
            image = request("GET", str(url), timeout=180)
            if not isinstance(image, bytes) or not image.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError(label + " returned a non-PNG image")
            (out / f"{label}-{index + 1}.png").write_bytes(image)

        return {
            "job_id": job_id,
            "wall_seconds": round(time.perf_counter() - wall_started, 3),
            "provider": final.get("provider"),
            "model": final.get("model"),
            "image_count": final.get("image_count"),
            "generation_seconds": final.get("elapsed_seconds"),
            "warm_image_enabled": final.get("warm_image_enabled"),
            "worker_state": final.get("worker_state"),
            "generation_prompts": final.get("generation_prompts"),
        }

    wait_health()
    first = run_image("first", "AIKA tạo một ảnh chân thực quả táo đỏ trên bàn gỗ, nền sạch.")
    second = run_image("second", "AIKA tạo một ảnh chân thực quả cam tươi trên bàn gỗ, nền sạch.")
    batch = run_image(
        "batch",
        "AIKA tạo hai ảnh riêng biệt để kiểm tra: ảnh thứ nhất là một bông sen hồng chân thực trên mặt nước; "
        "ảnh thứ hai là một bông hướng dương chân thực ngoài đồng. Đây là hai file ảnh riêng, không ghép chung.",
    )

    for name, result in (("first", first), ("second", second), ("batch", batch)):
        if result["provider"] != "kaggle-image-warm":
            raise RuntimeError(f"{name} did not use warm provider: {result['provider']}")
        if result["warm_image_enabled"] is not True:
            raise RuntimeError(f"{name} warm flag was not enabled")
    if int(first.get("image_count") or 0) != 1 or int(second.get("image_count") or 0) != 1:
        raise RuntimeError("single-image warm jobs did not return exactly one image")
    if int(batch.get("image_count") or 0) != 2:
        raise RuntimeError(f"batch expected 2 images, got {batch.get('image_count')}")
    if float(second["wall_seconds"]) >= 150:
        raise RuntimeError(f"second warm job took too long: {second['wall_seconds']}s")

    report = {"first": first, "second": second, "batch": batch}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("AIKA_WARM_CANARY_SUCCESS", flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
