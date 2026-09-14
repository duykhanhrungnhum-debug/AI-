from ai_agent.core.verifier import Verifier


def test_verifier_rejects_success_without_evidence():
    result = Verifier().verify(success=True, evidence=None)

    assert result.verified is False
    assert "No evidence" in result.reason


def test_verifier_accepts_success_with_evidence():
    result = Verifier().verify(success=True, evidence=["command output: OK"])

    assert result.verified is True
    assert result.evidence == ("command output: OK",)


def test_verifier_rejects_failed_execution_even_with_evidence():
    result = Verifier().verify(success=False, evidence=["command output: OK"])

    assert result.verified is False
    assert result.evidence == ("command output: OK",)


def test_verifier_ignores_empty_evidence():
    result = Verifier().verify(success=True, evidence=["", "  "])

    assert result.verified is False
