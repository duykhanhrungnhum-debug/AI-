#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_agent.core.kaggle_worker import KaggleGpuWorker


def stream_download(url: str, path: Path) -> None:
    req = Request(url, headers={"User-Agent": "Hidden-Beyond-Longform/1.0"})
    with urlopen(req, timeout=180) as src, path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    required = (
        "source_id", "series_id", "source_video_id", "source_url",
        "title", "episode_number",
    )
    missing = [k for k in required if k not in config]
    if missing:
        raise SystemExit(f"Missing config fields: {missing}")

    template = Path("hidden_beyond/longform_worker.py").read_text(encoding="utf-8")
    marker = "# __CONFIG_INJECT__"
    if marker not in template:
        raise SystemExit("Worker template config marker missing")
    source = template.replace(marker, "CONFIG = " + repr(config), 1)

    worker = KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        timeout=120,
        submission_retry_attempts=4,
        submission_retry_delay_seconds=30,
    )
    slug = os.environ.get("KAGGLE_KERNEL_SLUG", "hidden-beyond-longform-first")
    reuse_existing = False
    current = None
    for probe in range(1, 5):
        try:
            current = worker.status(slug)
            print(f"LONGFORM_EXISTING_STATUS {probe} {current.status} {current.failure_message}", flush=True)
            reuse_existing = (not current.terminal) or current.successful
            break
        except HTTPError as exc:
            if exc.code not in (403, 404):
                raise
            print(f"LONGFORM_EXISTING_STATUS_TRANSIENT {probe} HTTP_{exc.code}", flush=True)
            time.sleep(8)

    if reuse_existing:
        print("LONGFORM_KAGGLE_REUSE", slug, current.status if current else "unknown", flush=True)
    else:
        submission = worker.submit_script(
            slug=slug,
            title="Hidden Beyond Longform First Upload",
            source=source,
            enable_internet=True,
            is_private=True,
        )
        print("LONGFORM_KAGGLE_SUBMITTED", submission.ref, submission.version_number, flush=True)
        time.sleep(15)

    for attempt in range(1, 361):
        try:
            status = worker.status(slug)
        except HTTPError as exc:
            if exc.code in (403, 404) and attempt <= 12:
                print(f"LONGFORM_KAGGLE_STATUS_TRANSIENT {attempt} HTTP_{exc.code}", flush=True)
                time.sleep(10)
                continue
            raise
        print(f"LONGFORM_KAGGLE_POLL {attempt} {status.status} {status.failure_message}", flush=True)
        if status.terminal:
            if not status.successful:
                try:
                    print(worker.logs(slug)[-12000:], flush=True)
                except Exception as exc:
                    print("LOG_READ_FAILED", repr(exc), flush=True)
                raise SystemExit(f"Kaggle failed: {status.status} {status.failure_message}")
            break
        time.sleep(20)
    else:
        raise SystemExit("Timed out waiting for long-form Kaggle job")

    meta = worker.output_metadata(slug)
    files = meta.get("files")
    if not isinstance(files, list):
        raise SystemExit(f"Kaggle output missing files: {meta}")

    wanted = {"processed.mp4", "metadata.json", "vi.srt"}
    urls: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        name = item.get("fileName", item.get("file_name"))
        url = item.get("url")
        if name in wanted and isinstance(url, str) and url.startswith(("http://", "https://")):
            urls[str(name)] = url

    missing_files = sorted(wanted - set(urls))
    if missing_files:
        print(worker.logs(slug)[-12000:], flush=True)
        raise SystemExit(f"Kaggle output missing expected files: {missing_files}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "vi.srt", "processed.mp4"):
        print("DOWNLOADING", name, flush=True)
        stream_download(urls[name], out / name)

    report = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    if report.get("ok") is not True:
        raise SystemExit(f"Long-form metadata not OK: {report}")
    if int(report.get("size_bytes") or 0) != (out / "processed.mp4").stat().st_size:
        raise SystemExit("Downloaded processed.mp4 size does not match metadata")
    print("LONGFORM_OUTPUT_READY", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
