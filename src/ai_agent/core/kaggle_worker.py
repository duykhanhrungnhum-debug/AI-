"""On-demand Kaggle GPU worker using Kaggle's public REST API.

This adapter deliberately avoids Kaggle CLI token introspection. New KGAT API
tokens can be sent directly as Bearer credentials to the classic REST endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .invariants import assert_core_invariants


@dataclass(frozen=True)
class KaggleKernelSubmission:
    owner: str
    slug: str
    version_number: int | None
    kernel_id: int | None
    url: str | None

    @property
    def ref(self) -> str:
        return f"{self.owner}/{self.slug}"


@dataclass(frozen=True)
class KaggleKernelStatus:
    status: str
    failure_message: str = ""

    @property
    def terminal(self) -> bool:
        return self.status.upper() in {"COMPLETE", "ERROR", "CANCELLED", "FAILED"}

    @property
    def successful(self) -> bool:
        return self.status.upper() == "COMPLETE"


@dataclass
class KaggleGpuWorker:
    """Submit and inspect private Kaggle GPU scripts with a Bearer API token."""

    api_token: str
    username: str
    base_url: str = "https://www.kaggle.com/api/v1"
    timeout: float = 60.0
    user_agent: str = "AI-Agent-Kaggle/0.1"

    def __post_init__(self) -> None:
        if not self.api_token.strip():
            raise ValueError("api_token is required")
        if not self.username.strip():
            raise ValueError("username is required")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use HTTP(S)")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def submit_script(
        self,
        *,
        slug: str,
        title: str,
        source: str,
        machine_shape: str = "NvidiaTeslaT4",
        enable_internet: bool = False,
        is_private: bool = True,
    ) -> KaggleKernelSubmission:
        assert_core_invariants()
        slug = slug.strip()
        title = title.strip()
        if not slug or "/" in slug:
            raise ValueError("slug must be a non-empty kernel slug without an owner prefix")
        if not title:
            raise ValueError("title is required")
        if not source.strip():
            raise ValueError("source must not be empty")
        if not machine_shape.strip():
            raise ValueError("machine_shape is required")

        full_slug = f"{self.username.strip()}/{slug}"
        payload = {
            "slug": full_slug,
            "newTitle": title,
            "text": source,
            "language": "python",
            "kernelType": "script",
            "isPrivate": is_private,
            "enableGpu": True,
            "enableTpu": False,
            "enableInternet": enable_internet,
            "machineShape": machine_shape,
        }
        data = self._request_json("POST", "/kernels/push", payload=payload)

        error = data.get("error")
        if error:
            raise RuntimeError(f"Kaggle rejected kernel submission: {error}")

        version = data.get("versionNumber", data.get("version_number"))
        kernel_id = data.get("kernelId", data.get("kernel_id"))
        url = data.get("url")
        return KaggleKernelSubmission(
            owner=self.username.strip(),
            slug=slug,
            version_number=int(version) if version is not None else None,
            kernel_id=int(kernel_id) if kernel_id is not None else None,
            url=str(url) if url else None,
        )

    def status(self, slug: str) -> KaggleKernelStatus:
        assert_core_invariants()
        slug = slug.strip()
        if not slug or "/" in slug:
            raise ValueError("slug must be a non-empty kernel slug without an owner prefix")
        query = urlencode({"userName": self.username.strip(), "kernelSlug": slug})
        data = self._request_json("GET", f"/kernels/status?{query}")
        raw_status = data.get("status", "")
        if not isinstance(raw_status, str) or not raw_status.strip():
            raise ValueError("Kaggle status response does not contain status")
        failure = data.get("failureMessage", data.get("failure_message", ""))
        return KaggleKernelStatus(
            status=raw_status.strip(),
            failure_message=str(failure or "").strip(),
        )

    def _request_json(self, method: str, path: str, *, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url.rstrip("/") + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Kaggle returned non-JSON response") from exc
        if not isinstance(data, dict):
            raise ValueError("Kaggle response must be a JSON object")
        return data
