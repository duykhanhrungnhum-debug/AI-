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


def write_failure(out_dir: Path, reason: str, detail: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": False,
        "reason": reason,
        "detail": detail,
        "timestamp": int(time.time()),
    }
    (out_dir / "failure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("LONGFORM_FAIL", reason, detail, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
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
    submitted = False
    try:
        submission = worker.submit_script(
            slug=slug,
            title="Hidden Beyond Longform First Upload",
            source=source,
            enable_internet=True,
            is_private=True,
        )
        submitted = True
        print("LONGFORM_KAGGLE_SUBMITTED", submission.ref, submission.version_number, flush=True)
    except HTTPError as exc:
        if exc.code != 409:
            raise
        print("LONGFORM_KAGGLE_REUSE_ACTIVE HTTP_409", slug, flush=True)

    # Private kernels can transiently return 403 from the status endpoint even
    # after Kaggle accepted them. Poll persisted output metadata instead.
    expected_names = {"processed.mp4", "metadata.json", "vi.srt"}
    meta = None
    transient_streak = 0
    started = time.monotonic()
    for attempt in range(1, 271):
        elapsed = int(time.monotonic() - started)
        try:
            candidate = worker.output_metadata(slug)
            transient_streak = 0
            files = candidate.get("files") if isinstance(candidate, dict) else None
            names = {
                str(item.get("fileName", item.get("file_name")))
                for item in (files or [])
                if isinstance(item, dict)
            }
            print(
                f"LONGFORM_WATCHDOG attempt={attempt} elapsed={elapsed}s "
                f"state=reachable files={sorted(names)}",
                flush=True,
            )
            if expected_names.issubset(names):
                meta = candidate
                break
        except HTTPError as exc:
            if exc.code not in (403, 404, 409):
                write_failure(out, "kaggle_monitor_http_error", f"HTTP_{exc.code}")
                raise
            transient_streak += 1
            print(
                f"LONGFORM_WATCHDOG attempt={attempt} elapsed={elapsed}s "
                f"state=unreachable http={exc.code} streak={transient_streak}",
                flush=True,
            )
            if transient_streak >= 12:
                reason = (
                    f"Kaggle monitoring stayed unreachable for {transient_streak} checks "
                    f"({transient_streak * 20}s). Aborting instead of hanging silently."
                )
                write_failure(out, "kaggle_monitor_stalled", reason)
                raise SystemExit(reason)

        if attempt % 15 == 0:
            try:
                logs = worker.logs(slug)
                tail = logs[-4000:]
                if tail:
                    print("LONGFORM_KAGGLE_LOG_TAIL", tail, flush=True)
            except Exception as exc:
                print("LONGFORM_LOG_TRANSIENT", repr(exc), flush=True)
        time.sleep(20)
    else:
        reason = "Timed out after 90 minutes waiting for long-form Kaggle output"
        write_failure(out, "kaggle_output_timeout", reason)
        raise SystemExit(reason)

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
