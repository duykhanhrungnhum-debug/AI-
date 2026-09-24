"""Web-chat API layer with asynchronous Kaggle chat sessions."""
from __future__ import annotations

import hmac
import json
import os
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ai_agent.api import AIRequestHandler
from ai_agent.chat_session import CHAT_BROKER


class ChatRequestHandler(AIRequestHandler):
    server_version = "AI-Agent-Chat-API/0.1"

    def _worker_authorized(self) -> bool:
        expected = os.environ.get("AI_CHAT_WORKER_TOKEN", "")
        supplied = self.headers.get("authorization", "")
        return bool(expected) and hmac.compare_digest(supplied, f"Bearer {expected}")

    def _read_body(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/internal/chat/pull":
            if not self._worker_authorized():
                self._json(401, {"error": "unauthorized"})
                return
            job = CHAT_BROKER.pull_job()
            if job is None:
                self.send_response(204)
                self.send_header("content-length", "0")
                self.end_headers()
            else:
                self._json(200, job)
            return

        if parsed.path == "/v1/chat/status":
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            job_id = parse_qs(parsed.query).get("job_id", [""])[0].strip()
            if not job_id:
                self._json(400, {"error": "job_id is required"})
                return
            try:
                self._json(200, CHAT_BROKER.get_job(job_id))
            except KeyError:
                self._json(404, {"error": "job_not_found"})
            return

        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/internal/chat/heartbeat":
                if not self._worker_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                self._read_body()
                CHAT_BROKER.heartbeat()
                self._json(200, {"status": "ok"})
                return

            if path == "/internal/chat/result":
                if not self._worker_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                CHAT_BROKER.finish_job(self._read_body())
                self._json(200, {"status": "ok"})
                return

            if path == "/v1/chat":
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                body = self._read_body()
                job = CHAT_BROKER.create_job(str(body.get("prompt", "")))
                self._json(202, {
                    "job_id": job.job_id,
                    "status": job.status,
                    "worker_state": "starting",
                })
                return

            super().do_POST()
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except KeyError:
            self._json(404, {"error": "not_found"})
        except Exception as exc:
            self._json(502, {"error": "chat_failure", "detail": str(exc)})


def main() -> None:
    host = os.environ.get("AI_AGENT_API_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("AI_AGENT_API_PORT", "8080")))
    if not os.environ.get("AI_AGENT_API_TOKEN"):
        raise SystemExit("AI_AGENT_API_TOKEN is required")
    if not os.environ.get("AI_CHAT_WORKER_TOKEN"):
        raise SystemExit("AI_CHAT_WORKER_TOKEN is required")
    ThreadingHTTPServer((host, port), ChatRequestHandler).serve_forever()


if __name__ == "__main__":
    main()
