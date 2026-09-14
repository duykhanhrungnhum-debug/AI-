"""Safe first-party tool execution adapters for the model runtime.

The model may select among explicitly registered tools through a strict JSON
contract. Tool names and execution remain application-owned; arbitrary model
text is never treated as executable code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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

    @property
    def names(self) -> tuple[str, ...]:
        """Return the registered tool names exposed to the model."""
        return tuple(sorted(self._tools))

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
        return ActionExecution(False, "No explicit tool selection was supplied.")

    def execute_selected(self, decision: ModelDecision) -> ActionExecution:
        """Route a strict JSON model selection to a registered tool.

        Only the registered tool name is interpreted by this generic router;
        each tool owns validation of its request-specific arguments.
        """
        assert_core_invariants()
        try:
            payload = json.loads(decision.response.text)
        except json.JSONDecodeError:
            return ActionExecution(False, "Model tool selection must be valid JSON.")
        if not isinstance(payload, dict):
            return ActionExecution(False, "Model tool selection must be a JSON object.")
        name = payload.get("tool")
        if not isinstance(name, str) or not name.strip():
            return ActionExecution(False, "Model tool selection requires a non-empty tool name.")
        return self.execute_named(name, decision)

    def execute_named(self, name: str, decision: ModelDecision) -> ActionExecution:
        assert_core_invariants()
        tool = self._tools.get(name.strip())
        if tool is None:
            return ActionExecution(False, f"Unknown tool: {name}")
        try:
            return tool.run(decision)
        except Exception as exc:
            return ActionExecution(False, f"Tool {name} failed: {exc}")
