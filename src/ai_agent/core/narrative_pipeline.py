"""Verification-oriented narrative transformation pipeline.

This module is domain-agnostic. A consuming project can use it for stories,
articles, or other narrative material without coupling that project to AI-.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .invariants import assert_core_invariants
from .model import ModelProvider


@dataclass(frozen=True)
class NarrativeBrief:
    characters: tuple[str, ...]
    events: tuple[str, ...]
    must_preserve: tuple[str, ...]


@dataclass(frozen=True)
class ScriptQualityReport:
    passed: bool
    issues: tuple[str, ...]
    checks: tuple[str, ...]
    reviewer: str


@dataclass(frozen=True)
class NarrativeResult:
    brief: NarrativeBrief
    script: str
    review: ScriptQualityReport
    revision_count: int


class NarrativeProcessor:
    """Analyze, rewrite, review, and bounded-repair narrative text."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        review_provider: ModelProvider | None = None,
        max_revisions: int = 2,
    ):
        if max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        self.provider = provider
        self.review_provider = review_provider or provider
        self.max_revisions = max_revisions

    def process(self, source_text: str, *, target_language: str = "Vietnamese") -> NarrativeResult:
        assert_core_invariants()
        source_text = source_text.strip()
        if not source_text:
            raise ValueError("source_text must not be empty")
        if not target_language.strip():
            raise ValueError("target_language must not be empty")

        brief = self._analyze(source_text)
        script = self._draft(source_text, brief, target_language)
        review = self._review(source_text, brief, script)

        revisions = 0
        seen_scripts = {self._fingerprint(script)}
        while not review.passed and revisions < self.max_revisions:
            repaired = self._revise(source_text, brief, script, review, target_language)
            revisions += 1
            fingerprint = self._fingerprint(repaired)
            if fingerprint in seen_scripts:
                review = ScriptQualityReport(
                    passed=False,
                    issues=tuple(dict.fromkeys((*review.issues, "repair loop detected: repeated script"))),
                    checks=review.checks,
                    reviewer=review.reviewer,
                )
                script = repaired
                break
            seen_scripts.add(fingerprint)
            script = repaired
            review = self._review(source_text, brief, script)

        return NarrativeResult(brief=brief, script=script, review=review, revision_count=revisions)

    def _analyze(self, source_text: str) -> NarrativeBrief:
        prompt = (
            "NARRATIVE_ANALYSIS\n"
            "Read the source carefully. Extract only information supported by the source. "
            "Return ONLY JSON with keys characters, events, must_preserve; each value must be an array of strings. "
            "Do not translate or rewrite yet.\nSOURCE:\n" + source_text
        )
        data = self._json_response(self.provider, prompt, "narrative analysis")
        return NarrativeBrief(
            characters=self._string_tuple(data.get("characters"), "characters"),
            events=self._string_tuple(data.get("events"), "events"),
            must_preserve=self._string_tuple(data.get("must_preserve"), "must_preserve"),
        )

    def _draft(self, source_text: str, brief: NarrativeBrief, target_language: str) -> str:
        prompt = (
            "NARRATIVE_REWRITE\n"
            f"Rewrite the source naturally in {target_language}. Preserve characters, event order, relationships, "
            "causality, and important facts. Do not invent new plot facts. Avoid repetitive wording and broken sentences. "
            "Return only the rewritten script.\n"
            f"BRIEF:\n{self._brief_json(brief)}\nSOURCE:\n{source_text}"
        )
        text = self.provider.generate(prompt).text.strip()
        if not text:
            raise ValueError("model returned an empty narrative draft")
        return text

    def _review(self, source_text: str, brief: NarrativeBrief, script: str) -> ScriptQualityReport:
        deterministic = self._deterministic_issues(script)
        prompt = (
            "NARRATIVE_REVIEW\n"
            "Compare SCRIPT against SOURCE and BRIEF. Check wrong meaning, changed character identity, missing or reordered "
            "major events, invented facts, contradictions, nonsense, and excessive repetition. "
            'Return ONLY JSON: {"passed": true|false, "issues": [strings]}. '
            "passed may be true only when no material issue is found.\n"
            f"BRIEF:\n{self._brief_json(brief)}\nSOURCE:\n{source_text}\nSCRIPT:\n{script}"
        )
        response = self.review_provider.generate(prompt)
        data = self._parse_json(response.text, "narrative review")
        model_passed = data.get("passed")
        if not isinstance(model_passed, bool):
            raise ValueError("narrative review passed must be boolean")
        model_issues = self._string_tuple(data.get("issues"), "issues")

        issues = tuple(dict.fromkeys((*deterministic, *model_issues)))
        reviewer = f"{response.provider}/{response.model}"
        checks = (
            "deterministic:non_empty",
            "deterministic:no_exact_duplicate_paragraphs",
            "deterministic:no_placeholder_markers",
            "review:fidelity_and_coherence",
        )
        return ScriptQualityReport(
            passed=(not issues and model_passed),
            issues=issues,
            checks=checks,
            reviewer=reviewer,
        )

    def _revise(
        self,
        source_text: str,
        brief: NarrativeBrief,
        script: str,
        review: ScriptQualityReport,
        target_language: str,
    ) -> str:
        issues = "\n".join(f"- {issue}" for issue in review.issues) or "- review did not pass"
        prompt = (
            "NARRATIVE_REPAIR\n"
            f"Repair the script in {target_language}. Correct every listed issue while preserving the source facts and "
            "event order. Do not introduce unrelated changes or new plot facts. Return only the repaired script.\n"
            f"ISSUES:\n{issues}\nBRIEF:\n{self._brief_json(brief)}\nSOURCE:\n{source_text}\nCURRENT SCRIPT:\n{script}"
        )
        text = self.provider.generate(prompt).text.strip()
        if not text:
            raise ValueError("model returned an empty repaired script")
        return text

    def _json_response(self, provider: ModelProvider, prompt: str, label: str) -> dict:
        return self._parse_json(provider.generate(prompt).text, label)

    @staticmethod
    def _parse_json(raw: str, label: str) -> dict:
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{label} must be a JSON object")
        return data

    @staticmethod
    def _string_tuple(value: object, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be an array of strings")
        return tuple(item.strip() for item in value if item.strip())

    @staticmethod
    def _brief_json(brief: NarrativeBrief) -> str:
        return json.dumps({
            "characters": brief.characters,
            "events": brief.events,
            "must_preserve": brief.must_preserve,
        }, ensure_ascii=False)

    @staticmethod
    def _fingerprint(script: str) -> str:
        normalized = re.sub(r"\s+", " ", script).strip().casefold()
        return sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _deterministic_issues(script: str) -> tuple[str, ...]:
        issues: list[str] = []
        if not script.strip():
            issues.append("script is empty")

        paragraphs = [
            re.sub(r"\s+", " ", p).strip().casefold()
            for p in re.split(r"\n\s*\n", script)
            if len(p.strip()) >= 20
        ]
        seen: set[str] = set()
        if any(p in seen or seen.add(p) for p in paragraphs):
            issues.append("exact duplicate paragraph detected")

        if re.search(r"\b(?:TODO|TBD)\b|\[(?:insert|placeholder)[^]]*\]", script, re.IGNORECASE):
            issues.append("placeholder marker detected")
        return tuple(issues)
