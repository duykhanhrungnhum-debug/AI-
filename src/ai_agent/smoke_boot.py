"""One-shot authenticated production smoke test for AIKA.

Starts the normal chat API, submits one local authenticated request, polls until
AIKA completes or errors, prints a compact result, then keeps the API serving.
No secrets are logged.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request

from ai_agent.chat_api import main as chat_main


def _request(method: str, path: str, token: str, payload: dict | None = None) -> dict:
    port = os.environ.get("PORT", os.environ.get("AI_AGENT_API_PORT", "8080"))
    url = f"http://127.0.0.1:{port}{path}"
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
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    token = os.environ.get("AI_AGENT_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("AI_AGENT_API_TOKEN is required")

    server = threading.Thread(target=chat_main, name="aika-chat-api", daemon=True)
    server.start()
    time.sleep(2.0)

    prompt = os.environ.get("AIKA_SMOKE_PROMPT", "Chi tra loi dung: AIKA_SMOKE_OK")
    created = _request("POST", "/v1/chat", token, {"message": prompt})
    job_id = str(created["job_id"])
    print(f"AIKA_SMOKE_JOB={job_id}", flush=True)

    deadline = time.time() + float(os.environ.get("AIKA_SMOKE_TIMEOUT", "600"))
    last: dict = {}
    while time.time() < deadline:
        query = urllib.parse.urlencode({"job_id": job_id})
        last = _request("GET", f"/v1/chat/status?{query}", token)
        status = str(last.get("status", ""))
        if status in {"done", "error"}:
            result = {
                "status": status,
                "text": last.get("text", ""),
                "provider": last.get("provider", ""),
                "model": last.get("model", ""),
                "error": last.get("error", ""),
                "decision": last.get("decision", ""),
                "tool": last.get("tool", ""),
            }
            print("AIKA_SMOKE_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
            break
        time.sleep(5.0)
    else:
        print("AIKA_SMOKE_TIMEOUT=" + json.dumps(last, ensure_ascii=False), flush=True)

    server.join()


if __name__ == "__main__":
    main()
