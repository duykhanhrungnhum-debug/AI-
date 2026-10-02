"""Durable lifecycle helpers for AIKA's on-demand Kaggle brain worker.

This module keeps worker startup idempotent across Railway restarts/deploys.
Kaggle is treated as the external source of truth before a new notebook version
is submitted, so a healthy/starting worker is reused instead of cancelled by a
second push.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core.kaggle_worker import KaggleGpuWorker, KaggleKernelStatus


ACTIVE_KERNEL_STATES = frozenset({
    "running",
    "queued",
    "pending",
    "starting",
    "submitted",
    "validating",
    "preparing",
})

TERMINAL_KERNEL_STATES = frozenset({
    "complete",
    "completed",
    "error",
    "failed",
    "cancelled",
    "canceled",
    "cancelacknowledged",
})


def normalize_kernel_state(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def kernel_is_active(status: KaggleKernelStatus) -> bool:
    return normalize_kernel_state(status.status) in ACTIVE_KERNEL_STATES


def kernel_is_terminal(status: KaggleKernelStatus) -> bool:
    return normalize_kernel_state(status.status) in TERMINAL_KERNEL_STATES


def missing_kernel_error(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return (
        "kaggle http 404" in message
        or "not found" in message
        or "does not exist" in message
    )


@dataclass(frozen=True)
class KernelProbe:
    exists: bool
    status: KaggleKernelStatus | None = None

    @property
    def active(self) -> bool:
        return bool(self.status and kernel_is_active(self.status))

    @property
    def terminal(self) -> bool:
        return bool(self.status and kernel_is_terminal(self.status))


def probe_kernel(worker: KaggleGpuWorker, slug: str) -> KernelProbe:
    """Read Kaggle before any push; only a real 404 is treated as missing."""
    try:
        status = worker.status(slug)
    except RuntimeError as exc:
        if missing_kernel_error(exc):
            return KernelProbe(exists=False)
        raise
    return KernelProbe(exists=True, status=status)
