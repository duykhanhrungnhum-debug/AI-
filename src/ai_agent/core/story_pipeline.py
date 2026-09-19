"""Source-story quality gate feeding the verified media production pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .episode_pipeline import MediaProductionPipeline, MediaProductionResult
from .invariants import assert_core_invariants
from .narrative_pipeline import NarrativeResult


class NarrativeProcessorLike(Protocol):
    def process(self, source_text: str, *, target_language: str = "Vietnamese") -> NarrativeResult:
        """Transform and verify source narrative."""


@dataclass(frozen=True)
class StoryProductionResult:
    narrative: NarrativeResult
    media: MediaProductionResult | None
    blockers: tuple[str, ...]
    evidence: tuple[str, ...]

    @property
    def production_ready(self) -> bool:
        return (
            self.narrative.review.passed
            and self.media is not None
            and self.media.media_ready
            and not self.blockers
        )


@dataclass
class StoryProductionPipeline:
    narrative_processor: NarrativeProcessorLike
    media_pipeline: MediaProductionPipeline

    def produce(
        self,
        source_text: str,
        output_dir: str | Path,
        *,
        target_language: str = "Vietnamese",
        visual_style: str = "cinematic realistic",
    ) -> StoryProductionResult:
        assert_core_invariants()
        source_text = source_text.strip()
        if not source_text:
            raise ValueError("source_text must not be empty")

        narrative = self.narrative_processor.process(
            source_text,
            target_language=target_language,
        )
        narrative_evidence = (
            *narrative.evidence,
            f"script_review_passed:{str(narrative.review.passed).lower()}",
            f"script_reviewer:{narrative.review.reviewer}",
            f"script_revision_count:{narrative.revision_count}",
            *(f"script_check:{check}" for check in narrative.review.checks),
        )
        if not narrative.review.passed:
            blockers = narrative.review.issues or ("script review did not pass",)
            return StoryProductionResult(
                narrative=narrative,
                media=None,
                blockers=tuple(blockers),
                evidence=narrative_evidence,
            )

        media = self.media_pipeline.produce(
            narrative.script,
            output_dir,
            visual_style=visual_style,
        )
        blockers: list[str] = []
        if not media.media_ready:
            blockers.append("media verification did not pass")
        return StoryProductionResult(
            narrative=narrative,
            media=media,
            blockers=tuple(blockers),
            evidence=(*narrative_evidence, *media.evidence),
        )
