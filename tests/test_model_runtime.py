from ai_agent.core.engine import ExecutionEngine
from ai_agent.core.model import ModelResponse
from ai_agent.core.model_agent import ModelAgent, ModelDecision
from ai_agent.core.model_runtime import ActionExecution, ModelRuntime
from ai_agent.core.planner import PlanStepSpec, Planner
from ai_agent.core.project_state import ProjectState, StateStore
from ai_agent.core.recovery import RecoveryGuard


class FakeModel:
    def generate(self, prompt: str) -> ModelResponse:
        assert "Build the feature" in prompt
        return ModelResponse("proposed action", "fake", "test-model")


class FakeExecutor:
    def __init__(self, result: ActionExecution):
        self.result = result
        self.received: ModelDecision | None = None

    def execute(self, decision: ModelDecision) -> ActionExecution:
        self.received = decision
        return self.result


def make_runtime(tmp_path, result: ActionExecution):
    state = ProjectState(id="p1", name="demo")
    Planner().apply(state, "task-1", "Build the feature", [PlanStepSpec("step-1", "Implement it")])
    store = StateStore(tmp_path / "state.json")
    engine = ExecutionEngine(store, RecoveryGuard())
    executor = FakeExecutor(result)
    runtime = ModelRuntime(ModelAgent(FakeModel()), executor, engine)
    return state, runtime, executor


def test_model_runtime_connects_reasoning_execution_and_verification(tmp_path):
    state, runtime, executor = make_runtime(
        tmp_path, ActionExecution(True, "implemented", ("test suite passed",))
    )

    decision = runtime.run_current_step(state)

    step = state.current_task().current_step()
    assert decision.response.text == "proposed action"
    assert executor.received is decision
    assert step.status.value == "verified"
    assert step.evidence == ["test suite passed"]


def test_model_runtime_does_not_verify_without_evidence(tmp_path):
    state, runtime, _ = make_runtime(tmp_path, ActionExecution(True, "implemented"))

    runtime.run_current_step(state)

    step = state.current_task().current_step()
    assert step.status.value == "running"
    assert step.evidence == []
