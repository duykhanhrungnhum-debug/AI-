"""Long-running runtime loop with bounded retries and durable-friendly callbacks."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .invariants import assert_core_invariants


@dataclass(frozen=True)
class RuntimeResult:
    cycles: int
    failures: int
    stopped: bool


class AgentRuntime:
    """Keep invoking a work callback until stopped or a configured cycle limit is reached."""

    def __init__(self, work: Callable[[], bool], *, interval_seconds: float = 1.0,
                 max_cycles: int | None = None, max_consecutive_failures: int = 3) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be positive when provided")
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        self.work = work
        self.interval_seconds = interval_seconds
        self.max_cycles = max_cycles
        self.max_consecutive_failures = max_consecutive_failures
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> RuntimeResult:
        assert_core_invariants()
        cycles = 0
        failures = 0
        consecutive_failures = 0
        while not self._stop and (self.max_cycles is None or cycles < self.max_cycles):
            try:
                ok = bool(self.work())
            except Exception:
                ok = False
            cycles += 1
            if ok:
                consecutive_failures = 0
            else:
                failures += 1
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    break
            if not self._stop and self.interval_seconds:
                time.sleep(self.interval_seconds)
        return RuntimeResult(cycles=cycles, failures=failures, stopped=self._stop)
