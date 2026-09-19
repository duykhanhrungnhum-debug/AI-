from ai_agent.core.narrative_pipeline import NarrativeBrief, NarrativeResult, ScriptQualityReport
from ai_agent.core.narrative_verifier import NarrativeVerifier


def make_result(*, passed=True, issues=()):
    return NarrativeResult(
        brief=NarrativeBrief(("Lan",), ("Lan finds a letter",), ("Keep the letter",)),
        script="Lan tìm thấy lá thư.",
        review=ScriptQualityReport(
            passed=passed,
            issues=tuple(issues),
            checks=("deterministic:non_empty", "review:fidelity_and_coherence"),
            reviewer="local-reviewer/model",
        ),
        revision_count=1,
    )


def test_narrative_verifier_binds_evidence_to_script():
    verification = NarrativeVerifier().verify(make_result())

    assert verification.verified is True
    assert any(item.startswith("script_sha256:") for item in verification.evidence)
    assert "reviewer:local-reviewer/model" in verification.evidence
    assert "revision_count:1" in verification.evidence


def test_narrative_verifier_rejects_failed_quality_gate():
    verification = NarrativeVerifier().verify(make_result(passed=False, issues=("wrong meaning",)))

    assert verification.verified is False
    assert "wrong meaning" in verification.evidence
