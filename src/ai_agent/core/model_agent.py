"""Model-backed reasoning boundary for agent step execution.

The model may propose actions and expected checks, but it cannot self-verify.
Actual completion evidence must come from execution or an independent verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from .invariants import assert_core_invariants
from .model import ModelProvider, ModelResponse


@dataclass(frozen=True)
class ModelDecision:
    """A model proposal with provenance; not proof of task completion."""

    task_title: str
    step_id: str
    step_title: str
    response: ModelResponse

    @property
    def provenance(self) -> str:
        return f"model={self.response.model}; provider={self.response.provider}"


class ModelAgent:
    """Use a configured model for reasoning without allowing it to self-verify."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def decide(
        self,
        task_title: str,
        step_id: str,
        step_title: str,
        *,
        available_tools: tuple[str, ...] = (),
    ) -> ModelDecision:
        assert_core_invariants()
        if not task_title.strip():
            raise ValueError("task_title must not be empty")
        if not step_id.strip() or not step_title.strip():
            raise ValueError("step_id and step_title must not be empty")

        tool_instruction = ""
        if available_tools:
            names = ", ".join(available_tools)
            tool_instruction = (
                f"\nAvailable tools: {names}\n"
                "If a tool is needed, return ONLY a JSON object containing "
                "the selected tool name plus its request fields. For web "
                "research use one of: web_search with query/max_sources, "
                "web_fetch with url, or web_research with query/max_sources. "
                "Do not invent tool names."
            )
        prompt = (
            "You are the reasoning component of an AI agent.\n"
            f"Task: {task_title.strip()}\n"
            f"Step ID: {step_id.strip()}\n"
            f"Step: {step_title.strip()}\n"
            f"{tool_instruction}\n\n"
            "Return a concise proposed action and checks needed to independently "
            "verify the result. Do not claim the step is verified merely because "
            "you generated this response."
        )
        response = self.provider.generate(prompt)
        if not response.text.strip():
            raise ValueError("model returned an empty response")
        return ModelDecision(task_title.strip(), step_id.strip(), step_title.strip(), response)
