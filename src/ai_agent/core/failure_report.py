"""Deterministic failure reports for autonomous runs.

Reports are intentionally model-free so a failed AI/model step can still explain
where it stopped and what should happen next.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re


def _safe_message(error: BaseException, *, limit: int = 1200) -> str:
    message = str(error).strip() or error.__class__.__name__
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer <redacted>", message)
    message = re.sub(
        r"(?i)((?:api[_-]?key|token|authorization)\s*[:=]\s*)\S+",
        r"\1<redacted>",
        message,
    )
    message = re.sub(r"\s+", " ", message).strip()
    return message[:limit]


@dataclass(frozen=True)
class FailureReport:
    status: str
    stage: str
    error_type: str
    error_message: str
    retryable: bool
    next_action: str

    @classmethod
    def from_exception(cls, *, stage: str, error: BaseException) -> "FailureReport":
        if not stage.strip():
            raise ValueError("stage is required")
        message = _safe_message(error)
        lowered = message.casefold()

        if "maximum batch gpu session count" in lowered or "gpu session count" in lowered:
            retryable = True
            next_action = "Wait for a Kaggle GPU slot, then retry this same step."
        elif (
            "invalid format specifier" in lowered
            or "syntaxerror" in lowered
            or "indentationerror" in lowered
        ):
            retryable = False
            next_action = "Fix the local worker/template code before retrying; do not rerun unchanged code."
        elif "narrative verification failed" in lowered or "missing fact" in lowered:
            retryable = False
            next_action = "Inspect failed fact IDs and adjust the narrative review/repair rules before retrying."
        else:
            retryable = False
            next_action = "Inspect this failure and change the responsible step before retrying."

        return cls(
            status="BLOCKED",
            stage=stage.strip(),
            error_type=error.__class__.__name__,
            error_message=message,
            retryable=retryable,
            next_action=next_action,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    def console_line(self) -> str:
        return (
            f"AI_AGENT_BLOCKED | stage={self.stage} | error={self.error_type}: "
            f"{self.error_message} | retryable={str(self.retryable).lower()} | "
            f"next={self.next_action}"
        )

    def markdown(self) -> str:
        return (
            "## AI Agent blocked\n\n"
            f"- Stage: `{self.stage}`\n"
            f"- Error: `{self.error_type}` — {self.error_message}\n"
            f"- Retry unchanged: **{'yes' if self.retryable else 'no'}**\n"
            f"- Next action: {self.next_action}\n"
        )
