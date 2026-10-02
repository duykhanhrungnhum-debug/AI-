"""AIKA web API.

Every user request goes to AIKA first. AIKA may answer directly or call one of
its media tools. The HTTP layer never classifies media intent itself.
"""
from __future__ import annotations

import hmac
import json
import os
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ai_agent.agent_broker import AGENT_BROKER
from ai_agent.api import AIRequestHandler
from ai_agent.media_broker import MEDIA_BROKER


class ChatRequestHandler(AIRequestHandler):
    server_version = "AIKA-Chat-API/0.7"

    def _worker_authorized(self) -> bool:
        expected = os.environ.get("AI_AGENT_API_TOKEN", "").strip()
        supplied = self.headers.get("authorization", "").strip()
        return bool(expected) and hmac.compare_digest(supplied, f"Bearer {expected}")

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

    @staticmethod
    def _job_status(job_id: str) -> dict:
        # Once AIKA calls a media tool, that tool owns the same public job id.
        # Until then, status belongs to the AIKA brain job.
        try:
            return MEDIA_BROKER.get_job(job_id)
        except KeyError:
            return AGENT_BROKER.get_job(job_id)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/internal/chat/pull":
            if not self._worker_authorized():
                self._json(401, {"error": "unauthorized"})
                return
            self._empty_or_json(AGENT_BROKER.pull_job())
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
                self._json(200, self._job_status(job_id))
            except KeyError:
                self._json(404, {"error": "job_not_found"})
            return

        if parsed.path == "/v1/chat/image":
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            query = parse_qs(parsed.query)
            job_id = query.get("job_id", [""])[0].strip()
            raw_index = query.get("index", ["0"])[0].strip()
            if not job_id:
                self._json(400, {"error": "job_id is required"})
                return
            try:
                index = int(raw_index)
                data, mime = MEDIA_BROKER.get_image(job_id, index=index)
                self._binary(200, data, mime, filename=f"AIKA-{job_id[:12]}-{index + 1}.png")
            except ValueError as exc:
                self._json(409, {"error": str(exc)})
            except IndexError:
                self._json(404, {"error": "image_not_found"})
            except KeyError:
                self._json(404, {"error": "job_not_found"})
            return

        if parsed.path == "/v1/chat/video":
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            job_id = parse_qs(parsed.query).get("job_id", [""])[0].strip()
            if not job_id:
                self._json(400, {"error": "job_id is required"})
                return
            try:
                data, mime = MEDIA_BROKER.get_video(job_id)
                self._binary(200, data, mime, filename=f"AIKA-{job_id[:12]}.mp4")
            except ValueError as exc:
                self._json(409, {"error": str(exc)})
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
                body = self._read_body()
                AGENT_BROKER.heartbeat(str(body.get("state", "ready")))
                self._json(200, {"status": "ok"})
                return

            if path == "/internal/chat/result":
                if not self._worker_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                AGENT_BROKER.finish_job(self._read_body())
                self._json(200, {"status": "ok"})
                return

            if path == "/v1/chat":
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                body = self._read_body()
                message = str(body.get("message", "")).strip()
                prompt = str(body.get("prompt", "")).strip()
                command = message or prompt
                if not command:
                    raise ValueError("message or prompt is required")

                # No server-side media routing: AIKA sees and decides every request.
                job = AGENT_BROKER.create_job(command)
                state = AGENT_BROKER.get_job(job.job_id)
                self._json(202, {
                    "job_id": job.job_id,
                    "kind": "agent",
                    "status": state["status"],
                    "worker_state": state["worker_state"],
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
    if not os.environ.get("AI_AGENT_API_TOKEN", "").strip():
        raise SystemExit("AI_AGENT_API_TOKEN is required")
    ThreadingHTTPServer((host, port), ChatRequestHandler).serve_forever()


if __name__ == "__main__":
    main()
