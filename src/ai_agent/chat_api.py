"""Web-chat API layer with asynchronous AIKA skill execution."""
from __future__ import annotations

import hmac
import json
import os
from http.server import ThreadingHTTPServer
import threading
from urllib.parse import parse_qs, urlparse

from ai_agent.api import AIRequestHandler
from ai_agent.chat_session import CHAT_BROKER


class ChatRequestHandler(AIRequestHandler):
    server_version = "AIKA-Chat-API/0.5"

    def _worker_authorized(self) -> bool:
        expected = os.environ.get("AI_AGENT_API_TOKEN", "").strip()
        supplied = self.headers.get("authorization", "").strip()
        return bool(expected) and hmac.compare_digest(supplied, f"Bearer {expected}")

    def _worker_session_id(self) -> str:
        return self.headers.get("x-aika-worker-session", "").strip()

    def _read_body(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _binary(self, status: int, data: bytes, content_type: str, *, filename: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.send_header("content-disposition", f'inline; filename="{filename}"')
        self.send_header("cache-control", "private, no-store")
        self.end_headers()
        self.wfile.write(data)

    def _empty_or_json(self, payload: dict | None) -> None:
        if payload is None:
            self.send_response(204)
            self.send_header("content-length", "0")
            self.end_headers()
        else:
            self._json(200, payload)

    def _subject_lock_benchmark_path(self, parsed_path: str) -> bool:
        key = os.environ.get("AIKA_SUBJECT_LOCK_BENCHMARK_KEY", "").strip()
        if not key:
            return False
        prefix = f"/_aika_subject_lock/{key}/"
        if not parsed_path.startswith(prefix):
            return False
        from ai_agent import image_subject_lock_benchmark_temp as bench

        tail = parsed_path[len(prefix):]
        if tail == "status":
            self._json(200, bench.snapshot())
            return True
        if tail in {"1.png", "2.png"}:
            try:
                index = int(tail[0]) - 1
                data = bench.get_image(index)
            except (ValueError, IndexError):
                self._json(404, {"error": "image_not_ready"})
                return True
            self._binary(200, data, "image/png", filename=f"AIKA-subject-lock-{index + 1}.png")
            return True
        self._json(404, {"error": "not_found"})
        return True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if self._subject_lock_benchmark_path(parsed.path):
            return
        if parsed.path == "/internal/chat/pull":
            if not self._worker_authorized():
                self._json(401, {"error": "unauthorized"})
                return
            self._empty_or_json(CHAT_BROKER.pull_job())
            return

        if parsed.path == "/internal/image/pull":
            if not self._worker_authorized():
                self._json(401, {"error": "unauthorized"})
                return
            try:
                self._empty_or_json(CHAT_BROKER.pull_image_job(self._worker_session_id()))
            except PermissionError as exc:
                self._json(409, {"error": "stale_worker_session", "detail": str(exc)})
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

        if parsed.path == "/v1/chat/image":
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            query = parse_qs(parsed.query)
            job_id = query.get("job_id", [""])[0].strip()
            if not job_id:
                self._json(400, {"error": "job_id is required"})
                return
            raw_index = query.get("index", ["0"])[0].strip()
            try:
                index = int(raw_index)
            except ValueError:
                self._json(400, {"error": "index must be an integer"})
                return
            if index < 0:
                self._json(400, {"error": "index must be non-negative"})
                return
            try:
                data, mime = CHAT_BROKER.get_image(job_id, index=index)
                self._binary(200, data, mime, filename=f"AIKA-{job_id[:12]}-{index + 1}.png")
            except KeyError:
                self._json(404, {"error": "job_not_found"})
            except IndexError:
                self._json(404, {"error": "image_not_found"})
            except ValueError as exc:
                self._json(409, {"error": str(exc)})
            return

        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/internal/chat/heartbeat":
                if not self._worker_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                body = self._read_body()
                CHAT_BROKER.heartbeat(str(body.get("state", "ready")))
                self._json(200, {"status": "ok"})
                return

            if path == "/internal/chat/result":
                if not self._worker_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                CHAT_BROKER.finish_job(self._read_body())
                self._json(200, {"status": "ok"})
                return

            if path == "/internal/image/heartbeat":
                if not self._worker_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                body = self._read_body()
                CHAT_BROKER.image_heartbeat(
                    self._worker_session_id(),
                    str(body.get("state", "ready")),
                    current_job_id=str(body.get("current_job_id", "")),
                )
                self._json(200, {"status": "ok"})
                return

            if path == "/internal/image/result":
                if not self._worker_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                CHAT_BROKER.finish_image_job(
                    self._read_body(),
                    session_id=self._worker_session_id(),
                )
                self._json(200, {"status": "ok"})
                return

            if path == "/v1/chat":
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                body = self._read_body()
                message = str(body.get("message", ""))
                prompt = str(body.get("prompt", ""))
                job = CHAT_BROKER.create_job(prompt, message=message)
                state = CHAT_BROKER.get_job(job.job_id)
                self._json(202, {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "status": job.status,
                    "worker_state": state["worker_state"],
                })
                return

            super().do_POST()
        except PermissionError as exc:
            self._json(409, {"error": "stale_worker_session", "detail": str(exc)})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except KeyError:
            self._json(404, {"error": "not_found"})
        except Exception as exc:
            self._json(502, {"error": "chat_failure", "detail": str(exc)})


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def main() -> None:
    host = os.environ.get("AI_AGENT_API_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("AI_AGENT_API_PORT", "8080")))
    if not os.environ.get("AI_AGENT_API_TOKEN", "").strip():
        raise SystemExit("AI_AGENT_API_TOKEN is required")

    server = ThreadingHTTPServer((host, port), ChatRequestHandler)
    if _enabled("AIKA_SUBJECT_LOCK_BENCHMARK"):
        from ai_agent.image_subject_lock_benchmark_temp import run
        threading.Thread(target=run, name="aika-subject-lock-benchmark", daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
