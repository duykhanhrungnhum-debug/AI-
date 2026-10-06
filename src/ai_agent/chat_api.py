"""AIKA web API.

Every user request goes to AIKA first. AIKA may answer directly or call one of
its media tools. Explicit sibling-bot media calls may use the direct media
endpoints to avoid spending a brain/GPU pass re-classifying a request whose
operation is already known by the caller.

Money Agent compatibility endpoints intentionally reuse the same AIKA brokers:
Railway remains the control plane and Kaggle remains the GPU execution layer.
No second media stack is introduced.
"""
from __future__ import annotations

from cgi import FieldStorage
import hashlib
import hmac
import json
import os
import time
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

from ai_agent.agent_broker import AGENT_BROKER
from ai_agent.api import AIRequestHandler
from ai_agent.media_broker import MEDIA_BROKER


_MONEY_TEXT_MODELS = ("gpt-4.1", "gpt-4o-2024-08-06", "gpt-5.2")
_MONEY_IMAGE_MODEL = "aika-image"
_MONEY_VIDEO_MODEL = "aika-video"


class ChatRequestHandler(AIRequestHandler):
    server_version = "AIKA-Chat-API/0.9"

    def _worker_authorized(self) -> bool:
        expected = os.environ.get("AI_AGENT_API_TOKEN", "").strip()
        supplied = self.headers.get("authorization", "").strip()
        return bool(expected) and hmac.compare_digest(supplied, f"Bearer {expected}")

    def _money_authorized(self) -> bool:
        expected = os.environ.get("MONEY_AGENT_API_TOKEN", "").strip()
        supplied = self.headers.get("authorization", "").strip()
        return bool(expected) and hmac.compare_digest(supplied, f"Bearer {expected}")

    def _read_body(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _read_media_request(self) -> tuple[dict, bytes | None]:
        content_type = self.headers.get("content-type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            return self._read_body(), None
        form = FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("content-length", "0"),
            },
            keep_blank_values=True,
        )
        body: dict[str, object] = {}
        reference: bytes | None = None
        for field in form.list or []:
            if field.name in {"image", "image[]"} and getattr(field, "file", None) is not None:
                if reference is None:
                    reference = field.file.read()
                continue
            body[field.name] = field.value
        return body, reference

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
        try:
            return MEDIA_BROKER.get_job(job_id)
        except KeyError:
            return AGENT_BROKER.get_job(job_id)

    @staticmethod
    def _message_text(messages: object) -> str:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        rendered: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip() or "user"
            content = item.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                        parts.append(str(part.get("text", "")))
                text = "\n".join(parts)
            else:
                text = str(content or "")
            if text.strip():
                rendered.append(f"{role}: {text.strip()}")
        if not rendered:
            raise ValueError("messages contain no text")
        return "\n\n".join(rendered)

    @staticmethod
    def _wait_status(job_id: str, *, timeout: float, media_only: bool = False) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = MEDIA_BROKER.get_job(job_id) if media_only else ChatRequestHandler._job_status(job_id)
            status = str(state.get("status", "")).casefold()
            if status == "done":
                return state
            if status == "error":
                raise RuntimeError(str(state.get("error") or "AIKA job failed"))
            time.sleep(1.0)
        raise TimeoutError(f"AIKA job timed out after {timeout:.0f}s")

    @staticmethod
    def _asset_signature(kind: str, job_id: str, index: int = 0) -> str:
        secret = os.environ.get("MONEY_AGENT_API_TOKEN", "").strip()
        if not secret:
            raise RuntimeError("MONEY_AGENT_API_TOKEN is required")
        payload = f"{kind}:{job_id}:{index}".encode("utf-8")
        return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    def _asset_url(self, kind: str, job_id: str, index: int = 0) -> str:
        domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
        if domain:
            base = f"https://{domain}"
        else:
            proto = self.headers.get("x-forwarded-proto", "https").split(",", 1)[0].strip() or "https"
            host = self.headers.get("host", "").strip()
            if not host:
                raise RuntimeError("cannot determine public API host")
            base = f"{proto}://{host}"
        query = urlencode({
            "job_id": job_id,
            "index": index,
            "sig": self._asset_signature(kind, job_id, index),
        })
        return f"{base}/v1/openai/assets/{kind}?{query}"

    def _signed_asset_authorized(self, kind: str, job_id: str, index: int, supplied: str) -> bool:
        try:
            expected = self._asset_signature(kind, job_id, index)
        except RuntimeError:
            return False
        return bool(supplied) and hmac.compare_digest(supplied, expected)

    def _openai_chat_response(self, body: dict) -> None:
        model = str(body.get("model") or "gpt-5.2").strip()
        messages = self._message_text(body.get("messages"))
        command = (
            "This is a text-only Money Agent request. Answer in plain text only. "
            "Do not call image or video tools. Follow the conversation below faithfully.\n\n"
            + messages
        )
        job = AGENT_BROKER.create_job(command)
        timeout = float(os.environ.get("MONEY_AGENT_TEXT_TIMEOUT", "900"))
        state = self._wait_status(job.job_id, timeout=timeout)
        text = str(state.get("text", "")).strip()
        if not text:
            raise RuntimeError("AIKA returned empty text")
        created = int(time.time())
        response_id = f"chatcmpl-{job.job_id[:24]}"
        if bool(body.get("stream")):
            chunks = [
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
                },
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            payload = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks)
            payload += "data: [DONE]\n\n"
            data = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream; charset=utf-8")
            self.send_header("cache-control", "no-cache")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(200, {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/internal/chat/pull":
            if not self._worker_authorized():
                self._json(401, {"error": "unauthorized"})
                return
            self._empty_or_json(AGENT_BROKER.pull_job())
            return

        if parsed.path == "/v1/models":
            if not self._money_authorized():
                self._json(401, {"error": "unauthorized"})
                return
            self._json(200, {
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "created": 0, "owned_by": "aika"}
                    for model in _MONEY_TEXT_MODELS
                ],
            })
            return

        if parsed.path == "/v2/models":
            if not self._money_authorized():
                self._json(401, {"error": "unauthorized"})
                return
            self._json(200, [
                {
                    "id": _MONEY_IMAGE_MODEL,
                    "output": ["image"],
                    "inputs": {"text": True, "image": False},
                },
                {
                    "id": _MONEY_VIDEO_MODEL,
                    "output": ["video"],
                    "inputs": {"text": True, "image": True},
                },
            ])
            return

        if parsed.path in {"/v1/openai/assets/image", "/v1/openai/assets/video"}:
            kind = "image" if parsed.path.endswith("/image") else "video"
            query = parse_qs(parsed.query)
            job_id = query.get("job_id", [""])[0].strip()
            raw_index = query.get("index", ["0"])[0].strip()
            supplied = query.get("sig", [""])[0].strip()
            try:
                index = int(raw_index)
            except ValueError:
                self._json(400, {"error": "invalid index"})
                return
            if not job_id or not self._signed_asset_authorized(kind, job_id, index, supplied):
                self._json(401, {"error": "unauthorized"})
                return
            try:
                if kind == "image":
                    data, mime = MEDIA_BROKER.get_image(job_id, index=index)
                    filename = f"AIKA-{job_id[:12]}-{index + 1}.png"
                else:
                    data, mime = MEDIA_BROKER.get_video(job_id)
                    filename = f"AIKA-{job_id[:12]}.mp4"
                self._binary(200, data, mime, filename=filename)
            except ValueError as exc:
                self._json(409, {"error": str(exc)})
            except (IndexError, KeyError):
                self._json(404, {"error": "asset_not_found"})
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

            if path == "/v1/chat/completions":
                if not self._money_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                self._openai_chat_response(self._read_body())
                return

            if path in {"/v1/openai/images/generations", "/v1/openai/videos/generations"}:
                if not self._money_authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                body, reference = self._read_media_request()
                prompt = str(body.get("prompt") or "").strip()
                if not prompt:
                    raise ValueError("prompt is required")
                kind = "image" if "/images/" in path else "video"
                job = MEDIA_BROKER.create_job(
                    kind,
                    prompt,
                    reference_image=reference if kind == "video" else None,
                )
                timeout_default = "900" if kind == "image" else "1800"
                timeout = float(os.environ.get(f"MONEY_AGENT_{kind.upper()}_TIMEOUT", timeout_default))
                state = self._wait_status(job.job_id, timeout=timeout, media_only=True)
                if kind == "image":
                    url = self._asset_url("image", job.job_id, 0)
                else:
                    url = self._asset_url("video", job.job_id, 0)
                self._json(200, {
                    "created": int(time.time()),
                    "data": [{"url": url}],
                    "model": str(state.get("model", "")),
                })
                return

            if path in {"/v1/media/image", "/v1/media/video"}:
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                body = self._read_body()
                command = str(body.get("prompt") or body.get("message") or "").strip()
                if not command:
                    raise ValueError("prompt or message is required")
                kind = "image" if path.endswith("/image") else "video"
                job = MEDIA_BROKER.create_job(kind, command)
                state = MEDIA_BROKER.get_job(job.job_id)
                self._json(202, {
                    "job_id": job.job_id,
                    "kind": kind,
                    "status": state["status"],
                })
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
        except TimeoutError as exc:
            self._json(504, {"error": "timeout", "detail": str(exc)})
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