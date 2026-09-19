from ai_agent.core.episode_pipeline import MediaProductionResult
from ai_agent.core.kaggle_image_batch import BatchImageResult
from ai_agent.core.narrative_pipeline import (
    NarrativeBrief,
    NarrativeResult,
    ScriptQualityReport,
)
from ai_agent.core.story_pipeline import StoryProductionPipeline


class FakeNarrative:
    def __init__(self, passed=True):
        self.passed = passed

    def process(self, source_text, *, target_language="Vietnamese"):
        return NarrativeResult(
            brief=NarrativeBrief(("Lan",), ("event",), ("fact",)),
            script="Kịch bản đã kiểm tra.",
            review=ScriptQualityReport(
                passed=self.passed,
                issues=() if self.passed else ("meaning changed",),
                checks=("review:fidelity",),
                reviewer="local/model",
            ),
            revision_count=1,
        )


class FakeMedia:
    def __init__(self):
        self.called = False

    def produce(self, script, output_dir, *, visual_style):
        self.called = True

        class ReadyMedia:
            media_ready = True
            evidence = ("media:verified",)

        return ReadyMedia()


def test_story_pipeline_blocks_media_when_script_review_fails(tmp_path):
    media = FakeMedia()
    pipeline = StoryProductionPipeline(FakeNarrative(False), media)

    result = pipeline.produce("source story", tmp_path)

    assert result.production_ready is False
    assert result.media is None
    assert media.called is False
    assert "meaning changed" in result.blockers


def test_story_pipeline_runs_media_only_after_script_passes(tmp_path):
    media = FakeMedia()
    pipeline = StoryProductionPipeline(FakeNarrative(True), media)

    result = pipeline.produce("source story", tmp_path)

    assert result.production_ready is True
    assert media.called is True
    assert "media:verified" in result.evidence
    assert "script_review_passed:true" in result.evidence
