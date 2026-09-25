"""Durable production lessons for media tasks."""
from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Callable
from typing import Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ProductionLesson:
    task_type: str
    success: bool
    lesson: str
    failure_kind: str | None = None
    config: dict | None = None
    metrics: dict | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "success": self.success,
            "failure_kind": self.failure_kind,
            "lesson": self.lesson,
            "config": dict(self.config or {}),
            "metrics": dict(self.metrics or {}),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ProductionLesson":
        return cls(
            task_type=str(value.get("task_type") or ""),
            success=bool(value.get("success")),
            failure_kind=str(value["failure_kind"]) if value.get("failure_kind") else None,
            lesson=str(value.get("lesson") or ""),
            config=dict(value.get("config") or {}),
            metrics=dict(value.get("metrics") or {}),
            evidence=tuple(str(x) for x in value.get("evidence") or ()),
        )


class ProductionLearningStore(Protocol):
    def relevant(self, task_type: str, *, limit: int = 10) -> tuple[ProductionLesson, ...]:
        ...

    def record(self, lesson: ProductionLesson) -> ProductionLesson:
        ...


@dataclass
class HttpProductionLearningStore:
    base_url: str
    bearer_token: str = ""
    token_provider: Callable[[], str] | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use HTTP(S)")
        if not self.bearer_token.strip() and self.token_provider is None:
            raise ValueError("bearer_token or token_provider is required")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def _token(self) -> str:
        token = self.token_provider() if self.token_provider is not None else self.bearer_token
        token = str(token or "").strip()
        if not token:
            raise RuntimeError("production learning bearer token is unavailable")
        return token

    def _post(self, route: str, body: dict) -> dict:
        request = Request(
            f"{self.base_url}/{route.lstrip('/')}",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token()}",
                "Content-Type": "application/json",
                "User-Agent": "AI-Agent-Production-Learning/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"production learning API HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        if not isinstance(parsed, dict) or parsed.get("ok") is not True:
            raise RuntimeError(f"production learning API rejected request: {parsed}")
        return parsed

    def relevant(self, task_type: str, *, limit: int = 10) -> tuple[ProductionLesson, ...]:
        if not task_type.strip():
            raise ValueError("task_type is required")
        if limit <= 0:
            raise ValueError("limit must be positive")
        result = self._post("production-learning/query", {"task_type": task_type, "limit": limit})
        rows = result.get("lessons") or []
        if not isinstance(rows, list):
            raise ValueError("production learning API lessons must be an array")
        return tuple(ProductionLesson.from_dict(row) for row in rows if isinstance(row, dict))

    def record(self, lesson: ProductionLesson) -> ProductionLesson:
        if not lesson.task_type.strip() or not lesson.lesson.strip():
            raise ValueError("task_type and lesson are required")
        result = self._post("production-learning/record", lesson.to_dict())
        row = result.get("lesson")
        if not isinstance(row, dict):
            raise ValueError("production learning API did not return a lesson")
        return ProductionLesson.from_dict(row)


def tuned_reference_scale(
    lessons: tuple[ProductionLesson, ...] | list[ProductionLesson],
    *,
    base: float = 0.75,
) -> float:
    """Increase image adherence after identity failures, bounded to avoid prompt collapse."""
    scale = base
    for item in lessons:
        if item.success:
            prior = (item.config or {}).get("reference_scale")
            if isinstance(prior, (int, float)):
                scale = max(scale, float(prior))
            break
        if (item.failure_kind or "").casefold() == "identity_drift":
            scale += 0.05
    return min(0.92, max(0.50, round(scale, 3)))
