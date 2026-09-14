from ai_agent.core.engine import ExecutionEngine
from ai_agent.core.knowledge import KnowledgeItem, KnowledgeSource
from ai_agent.core.knowledge_retrieval import KnowledgeRetriever
from ai_agent.core.model import ModelResponse
from ai_agent.core.model_agent import ModelAgent, ModelDecision
from ai_agent.core.model_runtime import ActionExecution, ModelRuntime
from ai_agent.core.planner import PlanStepSpec, Planner
from ai_agent.core.project_state import ProjectState, StateStore
from ai_agent.core.recovery import RecoveryGuard


class FakeModel:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> ModelResponse:
        self.prompts.append(prompt)
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
    model = FakeModel()
    runtime = ModelRuntime(ModelAgent(model), executor, engine)
    return state, runtime, executor, model


def test_model_runtime_connects_reasoning_execution_and_verification(tmp_path):
    state, runtime, executor, _ = make_runtime(
        tmp_path, ActionExecution(True, "implemented", ("test suite passed",))
    )

    decision = runtime.run_current_step(state)

    step = state.current_task().current_step()
    assert decision.response.text == "proposed action"
    assert executor.received is decision
    assert step.status.value == "verified"
    assert step.evidence == ["test suite passed"]


def test_model_runtime_does_not_verify_without_evidence(tmp_path):
    state, runtime, _, _ = make_runtime(tmp_path, ActionExecution(True, "implemented"))

    runtime.run_current_step(state)

    step = state.current_task().current_step()
    assert step.status.value == "running"
    assert step.evidence == []


def test_model_runtime_feeds_retrieved_knowledge_to_model(tmp_path):
    state, _, _, _ = make_runtime(tmp_path, ActionExecution(True, "implemented", ("evidence",)))
    verified = KnowledgeItem(
        "Build feature steps should be tested independently",
        sources=[KnowledgeSource(uri="https://example.com/guide", content_hash="hash")],
    )
    verified.verify(["guide"])
    model = FakeModel()
    executor = FakeExecutor(ActionExecution(True, "implemented", ("evidence",)))
    runtime = ModelRuntime(
        ModelAgent(model),
        executor,
        ExecutionEngine(StateStore(tmp_path / "state2.json"), RecoveryGuard()),
        knowledge_retriever=KnowledgeRetriever([verified]),
    )
    runtime.run_current_step(state)

    assert "Build feature steps should be tested independently" in model.prompts[0]
    assert "[verified]" in model.prompts[0]
    assert "status is authoritative" in model.prompts[0]
