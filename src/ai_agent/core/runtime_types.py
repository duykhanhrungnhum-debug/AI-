"""Shared runtime result types kept independent to avoid import cycles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionExecution:
    """Result returned by an executor; evidence is independently checkable."""

    success: bool
    outcome: str
    evidence: tuple[str, ...] = ()
