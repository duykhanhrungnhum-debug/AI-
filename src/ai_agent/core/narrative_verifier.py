"""Narrative-specific quality gate producing auditable verification evidence."""
from __future__ import annotations

from hashlib import sha256

from .narrative_pipeline import NarrativeResult
from .verifier import VerificationResult


class NarrativeVerifier:
    """Verify that the narrative quality gate passed and bind evidence to the script."""

    def verify(self, result: NarrativeResult) -> VerificationResult:
        if not result.script.strip():
            return VerificationResult(False, "Narrative script is empty.")
        if not result.review.passed or result.review.issues:
            return VerificationResult(
                False,
                "Narrative quality review did not pass.",
                tuple(result.review.issues),
            )
        digest = sha256(result.script.encode("utf-8")).hexdigest()
        evidence = (
            f"script_sha256:{digest}",
            *result.review.checks,
            f"reviewer:{result.review.reviewer}",
            f"revision_count:{result.revision_count}",
        )
        return VerificationResult(
            True,
            "Narrative quality gate passed with script-bound review evidence.",
            evidence,
        )
