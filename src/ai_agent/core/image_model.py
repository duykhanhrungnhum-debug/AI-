"""Local image-generation provider boundary with a ComfyUI adapter."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

from .invariants import assert_core_invariants


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("image prompt must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class ImageArtifact:
    data: bytes
    mime_type: str
    provider: str
    model: str
    evidence: tuple[str, ...]


class ImageProvider(Protocol):
    def generate(self, request: ImageGenerationRequest) -> ImageArtifact:
        """Generate one image and return bytes plus provenance/evidence."""


class WorkflowFactory(Protocol):
    def __call__(self, request: ImageGenerationRequest) -> dict:
        """Build a ComfyUI API-format workflow for the request."""


@dataclass
class ComfyUIImageProvider:
    """Generate images through a local ComfyUI server.

    The workflow factory keeps checkpoint/node choices outside AI-'s core so a
    consuming project can supply its own exported ComfyUI API workflow.
    """

    workflow_factory: WorkflowFactory
    model: str
    base_url: str = "http://127.0.0.1:8188"
    output_node_id: str | None = None
    timeout: float = 120.0
    poll_interval: float = 0.5
    max_poll_attempts: int = 240
    provider: str = "comfyui-local"

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use HTTP(S)")
        if not self.model.strip():
            raise ValueError("model is required")
        if self.timeout <= 0 or self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("timeout/poll configuration must be positive")

    def generate(self, request: ImageGenerationRequest) -> ImageArtifact:
        assert_core_invariants()
        workflow = self.workflow_factory(request)
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError("workflow_factory must return a non-empty dict")

        client_id = str(uuid.uuid4())
        queued = self._json_request(
            self.base_url.rstrip("/") + "/prompt",
            data={"prompt": workflow, "client_id": client_id},
        )
        prompt_id = queued.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("ComfyUI response does not contain prompt_id")

        for _ in range(self.max_poll_attempts):
            history = self._json_request(self.base_url.rstrip("/") + f"/history/{prompt_id}")
            image_meta = self._find_image(history, prompt_id)
            if image_meta is not None:
                image_data = self._download_image(image_meta)
                if not image_data:
                    raise ValueError("ComfyUI returned an empty image")
                digest = sha256(image_data).hexdigest()
                return ImageArtifact(
                    data=image_data,
                    mime_type=self._mime_type(str(image_meta.get("filename", ""))),
                    provider=self.provider,
                    model=self.model,
                    evidence=(
                        f"comfyui_prompt_id:{prompt_id}",
                        f"image_sha256:{digest}",
                        f"dimensions_requested:{request.width}x{request.height}",
                    ),
                )
            if self.poll_interval:
                time.sleep(self.poll_interval)

        raise TimeoutError(f"ComfyUI did not produce an image after {self.max_poll_attempts} polls")

    def _json_request(self, url: str, *, data: dict | None = None) -> dict:
        if data is None:
            request: Request | str = Request(url, headers={"User-Agent": "AI-Agent-Image/0.1"})
        else:
            payload = json.dumps(data).encode("utf-8")
            request = Request(
                url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json", "User-Agent": "AI-Agent-Image/0.1"},
            )
        with urlopen(request, timeout=self.timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("ComfyUI JSON response must be an object")
        return parsed

    def _find_image(self, history: dict, prompt_id: str) -> dict | None:
        entry = history.get(prompt_id)
        if not isinstance(entry, dict):
            return None
        outputs = entry.get("outputs")
        if not isinstance(outputs, dict):
            return None

        nodes = [self.output_node_id] if self.output_node_id else list(outputs)
        for node_id in nodes:
            node = outputs.get(node_id)
            if not isinstance(node, dict):
                continue
            images = node.get("images")
            if isinstance(images, list) and images and isinstance(images[0], dict):
                return images[0]
        return None

    def _download_image(self, meta: dict) -> bytes:
        filename = meta.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError("ComfyUI image metadata has no filename")
        params = urlencode({
            "filename": filename,
            "subfolder": str(meta.get("subfolder", "")),
            "type": str(meta.get("type", "output")),
        })
        url = self.base_url.rstrip("/") + "/view?" + params
        request = Request(url, headers={"User-Agent": "AI-Agent-Image/0.1"})
        with urlopen(request, timeout=self.timeout) as response:
            return response.read()

    @staticmethod
    def _mime_type(filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            return "image/jpeg"
        if lower.endswith(".webp"):
            return "image/webp"
        return "image/png"
