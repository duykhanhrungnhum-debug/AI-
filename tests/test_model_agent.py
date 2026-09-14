from ai_agent.core.model import ModelResponse
from ai_agent.core.model_agent import ModelAgent


class FakeModel:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> ModelResponse:
        self.prompts.append(prompt)
        return ModelResponse("proposed action and independent checks", "fake", "fake-model")


def test_model_agent_returns_proposal_with_provenance():
    provider = FakeModel()
    decision = ModelAgent(provider).decide("Build the agent", "step-1", "Design execution")

    assert decision.response.text.startswith("proposed action")
    assert decision.provenance == "model=fake-model; provider=fake"
    assert "Build the agent" in provider.prompts[0]
    assert "independently verify" in provider.prompts[0]


def test_model_agent_never_treats_model_output_as_verification():
    provider = FakeModel()
    decision = ModelAgent(provider).decide("Task", "step-1", "Do work")

    assert not hasattr(decision, "verified")
