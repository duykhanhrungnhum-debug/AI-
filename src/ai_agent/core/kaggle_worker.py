"""On-demand Kaggle GPU worker using Kaggle's public REST API.

This adapter deliberately avoids Kaggle CLI token introspection. New KGAT API
tokens can be sent directly as Bearer credentials to the classic REST endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from time import sleep
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
    submission_retry_attempts: int = 5
    submission_retry_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.api_token.strip():
            raise ValueError("api_token is required")
        if not self.username.strip():
            raise ValueError("username is required")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use HTTP(S)")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.submission_retry_attempts <= 0:
            raise ValueError("submission_retry_attempts must be positive")
        if self.submission_retry_delay_seconds < 0:
            raise ValueError("submission_retry_delay_seconds must be non-negative")

    def submit_script(
        self,
        *,
        slug: str,
        title: str,
        source: str,
        machine_shape: str | None = None,
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
        if machine_shape is not None and not machine_shape.strip():
            raise ValueError("machine_shape must be non-empty when provided")

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
        }
        if machine_shape is not None:
            payload["machineShape"] = machine_shape
        data = {}
        for attempt in range(1, self.submission_retry_attempts + 1):
            data = self._request_json("POST", "/kernels/push", payload=payload)
            error = data.get("error")
            if not error:
                break
            message = str(error)
            capacity_limited = (
                "maximum batch gpu session count" in message.casefold()
                and "reached" in message.casefold()
            )
            if not capacity_limited or attempt >= self.submission_retry_attempts:
                raise RuntimeError(f"Kaggle rejected kernel submission: {message}")
            sleep(self.submission_retry_delay_seconds * attempt)

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

    def logs(self, slug: str) -> str:
        """Return persisted/latest execution logs for a Kaggle kernel."""
        assert_core_invariants()
        slug = slug.strip()
        if not slug or "/" in slug:
            raise ValueError("slug must be a non-empty kernel slug without an owner prefix")
        url = (
            self.base_url.rstrip("/")
            + f"/kernels/logs/stream/{self.username.strip()}/{slug}"
        )
        request = Request(
            url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.api_token.strip()}",
                "Accept": "*/*",
                "User-Agent": self.user_agent,
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw

        events = parsed if isinstance(parsed, list) else [parsed]
        parts: list[str] = []
        for event in events:
            if isinstance(event, dict):
                value = event.get("data")
                if value is not None:
                    parts.append(str(value))
            elif event is not None:
                parts.append(str(event))
        return "\n".join(parts)

    def output_metadata(self, slug: str) -> dict:
        """Return metadata for the latest kernel output."""
        assert_core_invariants()
        slug = slug.strip()
        if not slug or "/" in slug:
            raise ValueError("slug must be a non-empty kernel slug without an owner prefix")
        query = urlencode({"userName": self.username.strip(), "kernelSlug": slug})
        return self._request_json("GET", f"/kernels/output?{query}")

    def download_output_file(self, slug: str, filename: str) -> bytes:
        """Download one named output file using Kaggle's signed output URL."""
        filename = filename.strip()
        if not filename:
            raise ValueError("filename is required")
        metadata = self.output_metadata(slug)
        files = metadata.get("files")
        if not isinstance(files, list):
            raise ValueError("Kaggle output response does not contain files")

        for item in files:
            if not isinstance(item, dict):
                continue
            item_name = item.get("fileName", item.get("file_name"))
            if item_name != filename:
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ValueError(f"Kaggle output file {filename} has no download URL")
            request = Request(url, headers={"User-Agent": self.user_agent})
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        raise FileNotFoundError(f"Kaggle output file not found: {filename}")

    def _request_json(self, method: str, path: str, *, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url.rstrip("/") + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_token.strip()}",
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
