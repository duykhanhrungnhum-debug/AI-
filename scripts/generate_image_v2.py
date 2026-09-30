#!/usr/bin/env python3
"""Generate images through the single V2 production path."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.media_v2 import ImageRequestV2, KaggleImageV2Provider


def load_requests() -> tuple[ImageRequestV2, ...]:
    raw = os.environ.get("MEDIA_COMMANDS_JSON", "").strip()
    if not raw:
        raise ValueError("MEDIA_COMMANDS_JSON is required")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("MEDIA_COMMANDS_JSON must be a non-empty list")
    requests = []
    seen = set()
    for index, item in enumerate(payload):
        if isinstance(item, str):
            item = {"id": f"item-{index + 1}", "command": item}
        if not isinstance(item, dict):
            raise ValueError("each request must be text or an object")
        item_id = str(item.get("id") or f"item-{index + 1}").strip()
        command = str(item.get("command") or "").strip()
        if not item_id or not command or item_id in seen:
            raise ValueError("requests require unique ids and non-empty commands")
        seen.add(item_id)
        seed = item.get("seed")
        if seed is None:
            seed = int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")
        requests.append(ImageRequestV2(
            item_id=item_id,
            command=command,
            seed=int(seed),
            width=int(item.get("width") or 1024),
            height=int(item.get("height") or 1024),
        ))
    return tuple(requests)


def main() -> int:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token or not username:
        raise RuntimeError("KAGGLE_API_TOKEN and KAGGLE_USERNAME are required")
    output_dir = Path(os.environ.get("MEDIA_OUTPUT_DIR", "media-v2-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = KaggleImageV2Provider(
        worker=KaggleGpuWorker(
            api_token=token,
            username=username,
            timeout=120,
            submission_retry_attempts=5,
            submission_retry_delay_seconds=30,
        ),
        kernel_slug=os.environ.get("MEDIA_V2_KERNEL_SLUG", "ai-agent-image-v2").strip(),
        poll_interval=3,
        max_poll_attempts=int(os.environ.get("MEDIA_V2_MAX_POLL_ATTEMPTS", "600")),
    )
    results = provider.generate_many(load_requests())
    manifest = {
        "architecture": "natural_request->single_recaption->flux2_klein",
        "runtime_visual_qa": False,
        "items": {},
    }
    for result in results:
        target = output_dir / result.item_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "image.png").write_bytes(result.data)
        item = {
            "command": result.command,
            "prompt": result.prompt,
            "model": result.model,
            "seed": result.seed,
            "width": result.width,
            "height": result.height,
            "elapsed_seconds": result.elapsed_seconds,
            "evidence": list(result.evidence),
        }
        (target / "result.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["items"][result.item_id] = item
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("AI_AGENT_MEDIA_V2_OK")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
