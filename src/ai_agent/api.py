"""HTTP API exposing AI- as the shared intelligence service for sibling bots."""
from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_agent.core.model_factory import build_model_provider


def _translate_prompt(text: str, source_language: str, target_language: str) -> str:
    return (
        "Translate the following dialogue faithfully for Vietnamese dubbing. "
        "Preserve names, meaning, tone, and line order. Return only the translated text.\n"
        f"Source language: {source_language}\nTarget language: {target_language}\n\n{text}"
    )


class AIRequestHandler(BaseHTTPRequestHandler):
    server_version = "AI-Agent-API/0.1"

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        expected = os.environ.get("AI_AGENT_API_TOKEN", "")
        if not expected:
            return False
        supplied = self.headers.get("authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "service": "AI-"})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/v1/generate":
                prompt = str(body.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("prompt is required")
            elif self.path == "/v1/translate":
                text = str(body.get("text", "")).strip()
                if not text:
                    raise ValueError("text is required")
                prompt = _translate_prompt(
                    text,
                    str(body.get("source_language", "auto")),
                    str(body.get("target_language", "Vietnamese")),
                )
            else:
                self._json(404, {"error": "not_found"})
                return

            result = build_model_provider().generate(prompt)
            self._json(200, {
                "text": result.text,
                "provider": result.provider,
                "model": result.model,
            })
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(502, {"error": "model_failure", "detail": str(exc)})

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = os.environ.get("AI_AGENT_API_HOST", "0.0.0.0")
    port = int(os.environ.get("AI_AGENT_API_PORT", "8080"))
    if not os.environ.get("AI_AGENT_API_TOKEN"):
        raise SystemExit("AI_AGENT_API_TOKEN is required")
    ThreadingHTTPServer((host, port), AIRequestHandler).serve_forever()


if __name__ == "__main__":
    main()
