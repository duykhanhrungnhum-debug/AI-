"""Authenticated smoke-test utility for AIKA.

Two explicit modes are supported:
- local mode (default): start a local chat API and test it;
- remote mode (AIKA_SMOKE_BASE_URL): test an already-running deployment without
  restarting or changing it.

Secrets are read from environment variables and are never logged.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request

from ai_agent.chat_api import main as chat_main


def _request(
    method: str,
    path: str,
    token: str,
    payload: dict | None = None,
    *,
    base_url: str | None = None,
) -> dict:
    if base_url:
        root = base_url.rstrip("/")
    else:
        port = os.environ.get("PORT", os.environ.get("AI_AGENT_API_PORT", "8080"))
        root = f"http://127.0.0.1:{port}"
    url = root + path
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def run_smoke(*, base_url: str | None = None) -> dict:
    token = os.environ.get("AI_AGENT_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("AI_AGENT_API_TOKEN is required")

    prompt = os.environ.get("AIKA_SMOKE_PROMPT", "Chi tra loi dung: AIKA_SMOKE_OK")
    created = _request(
        "POST",
        "/v1/chat",
        token,
        {"message": prompt},
        base_url=base_url,
    )
    job_id = str(created["job_id"])
    print(f"AIKA_SMOKE_JOB={job_id}", flush=True)

    deadline = time.time() + float(os.environ.get("AIKA_SMOKE_TIMEOUT", "600"))
    last: dict = {}
    while time.time() < deadline:
        query = urllib.parse.urlencode({"job_id": job_id})
        last = _request(
            "GET",
            f"/v1/chat/status?{query}",
            token,
            base_url=base_url,
        )
        status = str(last.get("status", ""))
        if status in {"done", "error"}:
            result = {
                "job_id": job_id,
                "status": status,
                "text": last.get("text", ""),
                "provider": last.get("provider", ""),
                "model": last.get("model", ""),
                "error": last.get("error", ""),
                "worker_state": last.get("worker_state", ""),
                "worker_error": last.get("worker_error", ""),
                "decision": last.get("decision", ""),
                "tool": last.get("tool", ""),
            }
            print("AIKA_SMOKE_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
            return result
        time.sleep(5.0)

    result = dict(last)
    result["job_id"] = job_id
    print("AIKA_SMOKE_TIMEOUT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    remote_base = os.environ.get("AIKA_SMOKE_BASE_URL", "").strip()
    if remote_base:
        result = run_smoke(base_url=remote_base)
        raise SystemExit(0 if result.get("status") == "done" else 2)

    server = threading.Thread(target=chat_main, name="aika-chat-api", daemon=True)
    server.start()
    time.sleep(2.0)
    result = run_smoke()
    if result.get("status") != "done":
        raise SystemExit(2)
    server.join()


if __name__ == "__main__":
    main()
