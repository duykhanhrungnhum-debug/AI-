"""Safe first-party tool execution adapters for the model runtime.

The model proposes what to do; this layer decides what is actually executable.
It never turns model text into an arbitrary shell command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .invariants import assert_core_invariants
from .model_agent import ModelDecision
from .model_runtime import ActionExecution


@dataclass(frozen=True)
class Tool:
    """Named callable with an explicit, application-owned execution boundary."""

    name: str
    run: Callable[[ModelDecision], ActionExecution]


class ToolExecutor:
    """Execute only explicitly registered tools; unknown tools are rejected."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools = {tool.name.strip(): tool for tool in (tools or []) if tool.name.strip()}

    def register(self, tool: Tool) -> None:
        assert_core_invariants()
        if not tool.name.strip():
            raise ValueError("tool name must not be empty")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def execute(self, decision: ModelDecision) -> ActionExecution:
        """Run a tool selected by the host, not by interpreting arbitrary model text."""
        assert_core_invariants()
        if not self._tools:
            return ActionExecution(False, "No tools are registered.")
        # A model decision is data, not executable code. The host must select
        # a registered tool explicitly before calling this adapter.
        return ActionExecution(False, "No explicit tool selection was supplied.")

    def execute_named(self, name: str, decision: ModelDecision) -> ActionExecution:
        assert_core_invariants()
        tool = self._tools.get(name.strip())
        if tool is None:
            return ActionExecution(False, f"Unknown tool: {name}")
        try:
            return tool.run(decision)
        except Exception as exc:
            return ActionExecution(False, f"Tool {name} failed: {exc}")
