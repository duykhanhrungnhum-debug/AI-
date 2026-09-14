from ai_agent.core.model import ModelResponse
from ai_agent.core.model_agent import ModelDecision
from ai_agent.core.model_runtime import ActionExecution
from ai_agent.core.tool_executor import Tool, ToolExecutor


def decision() -> ModelDecision:
    return ModelDecision(
        task_title="test task",
        step_id="step-1",
        step_title="test step",
        response=ModelResponse("proposed action", "fake", "fake-model"),
    )


def test_executor_requires_explicit_tool_selection() -> None:
    executor = ToolExecutor([Tool("echo", lambda _: ActionExecution(True, "ok", ("observed",)))])
    result = executor.execute(decision())
    assert not result.success
    assert "No explicit tool selection" in result.outcome


def test_executor_runs_registered_tool() -> None:
    executor = ToolExecutor([Tool("echo", lambda _: ActionExecution(True, "ok", ("observed",)))])
    result = executor.execute_named("echo", decision())
    assert result.success
    assert result.evidence == ("observed",)


def test_unknown_tool_is_failure() -> None:
    result = ToolExecutor().execute_named("missing", decision())
    assert not result.success
    assert "Unknown tool" in result.outcome


def test_tool_failure_is_captured() -> None:
    def broken(_: ModelDecision) -> ActionExecution:
        raise RuntimeError("boom")

    result = ToolExecutor([Tool("broken", broken)]).execute_named("broken", decision())
    assert not result.success
    assert "boom" in result.outcome
