"""Evidence-based verification for completed agent steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of checking whether an execution result is proven."""

    verified: bool
    reason: str
    evidence: tuple[str, ...] = ()


class Verifier:
    """Allow VERIFIED only when execution succeeded and evidence is present."""

    def verify(self, *, success: bool, evidence: list[str] | None) -> VerificationResult:
        """Validate the minimum proof contract before a step becomes VERIFIED."""
        cleaned = tuple(item.strip() for item in (evidence or []) if item and item.strip())
        if not success:
            return VerificationResult(False, "Execution did not succeed.", cleaned)
        if not cleaned:
            return VerificationResult(False, "No evidence was provided; the step cannot be VERIFIED.")
        return VerificationResult(True, "Execution succeeded and evidence was provided.", cleaned)
