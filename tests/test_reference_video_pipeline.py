from pathlib import Path

from ai_agent.core.audio_quality import AudioQualityReport
from ai_agent.core.episode_pipeline import MediaProductionPipeline
from ai_agent.core.image_model import ImageArtifact
from ai_agent.core.kaggle_image_batch import BatchImageResult, SceneImageResult
from ai_agent.core.production_learning import ProductionLesson, tuned_reference_scale
from ai_agent.core.scene_planner import VisualScene, VisualScenePlan
from ai_agent.core.tts_model import AudioArtifact
from ai_agent.core.video_builder import VideoArtifact


class FakeScenePlanner:
    def plan(self, script, *, visual_style):
        return VisualScenePlan((
            VisualScene("s1", "n1", "the hero enters a lantern-lit hall, 16:9 widescreen composition"),
            VisualScene("s2", "n2", "the hero studies an ancient map, 16:9 widescreen composition"),
        ))


class FakeReferenceImageProvider:
    def __init__(self):
        self.requests = []
        self.reference_image = None
        self.reference_scale = None

    def generate_with_retries(
        self,
        requests,
        *,
        max_rounds,
        reference_image=None,
        reference_scale=0.75,
    ):
        self.requests = list(requests)
        self.reference_image = reference_image
        self.reference_scale = reference_scale
        scenes = []
        for index, request in enumerate(requests):
            scenes.append(SceneImageResult(
                scene_id=request.scene_id,
                artifact=ImageArtifact(
                    data=b"PNG" + request.scene_id.encode(),
                    mime_type="image/png",
                    provider="fake-reference-image",
                    model="fake",
                    evidence=(
                        f"scene_id:{request.scene_id}",
                        f"identity_score:{0.81 + index * 0.01:.6f}",
                        f"reference_scale:{reference_scale:.3f}",
                    ),
                ),
                verified=True,
                issues=(),
            ))
        return BatchImageResult(tuple(scenes), rounds=1)


class FakeSpeech:
    def synthesize(self, text):
        return AudioArtifact(
            data=b"WAV",
            mime_type="audio/wav",
            provider="fake-tts",
            model="voice",
            duration_seconds=2.0,
            sample_rate=22050,
            channels=1,
            evidence=("audio:verified",),
        )


class FakeAudioVerifier:
    def verify(self, artifact):
        return AudioQualityReport(
            passed=True,
            issues=(),
            rms_ratio=0.1,
            peak_ratio=0.5,
            silence_ratio=0.1,
            evidence=("audio_signal:verified",),
        )


class FakeVideo:
    def build(self, image_paths, audio_path, output_path):
        assert len(image_paths) == 2
        assert Path(audio_path).read_bytes() == b"WAV"
        Path(output_path).write_bytes(b"MP4")
        return VideoArtifact(
            path=output_path,
            duration_seconds=2.0,
            width=1920,
            height=1080,
            has_video=True,
            has_audio=True,
            evidence=("video:verified",),
        )


class FakeLearningStore:
    def __init__(self):
        self.recorded = []

    def relevant(self, task_type, *, limit=10):
        assert task_type == "reference-character-video"
        return (
            ProductionLesson(
                task_type=task_type,
                success=False,
                failure_kind="identity_drift",
                lesson="Prior run drifted from the character reference.",
                config={"reference_scale": 0.75},
            ),
        )

    def record(self, lesson):
        self.recorded.append(lesson)
        return lesson


def test_reference_image_pipeline_uses_prior_lessons_and_records_success(tmp_path):
    images = FakeReferenceImageProvider()
    learning = FakeLearningStore()
    pipeline = MediaProductionPipeline(
        scene_planner=FakeScenePlanner(),
        image_provider=images,
        speech_provider=FakeSpeech(),
        video_builder=FakeVideo(),
        learning_store=learning,
        audio_verifier=FakeAudioVerifier(),
    )

    reference = b"REFERENCE-CHARACTER"
    result = pipeline.produce(
        "Kịch bản đã được xác minh.",
        tmp_path,
        reference_image=reference,
    )

    assert result.media_ready is True
    assert images.reference_image == reference
    assert images.reference_scale == 0.80
    assert all("supplied reference image" in request.prompt for request in images.requests)
    assert all("changed face" in request.negative_prompt for request in images.requests)
    assert any(item == "visual_mode:reference_conditioned_story_video" for item in result.evidence)
    assert any(item.startswith("reference_sha256:") for item in result.evidence)
    assert any(item == "production_lessons_loaded:1" for item in result.evidence)
    assert learning.recorded
    assert learning.recorded[-1].success is True
    assert learning.recorded[-1].metrics["min_identity_score"] == 0.81


def test_reference_scale_learns_from_identity_failures_and_is_bounded():
    lessons = tuple(
        ProductionLesson(
            task_type="reference-character-video",
            success=False,
            failure_kind="identity_drift",
            lesson="identity drift",
        )
        for _ in range(10)
    )
    assert tuned_reference_scale(lessons, base=0.75) == 0.92


class FailingLearningStore:
    def relevant(self, task_type, *, limit=10):
        return ()

    def record(self, lesson):
        raise RuntimeError("lesson store unavailable")


class FailingReferenceImageProvider(FakeReferenceImageProvider):
    def generate_with_retries(
        self,
        requests,
        *,
        max_rounds,
        reference_image=None,
        reference_scale=0.75,
    ):
        request = list(requests)[0]
        return BatchImageResult((
            SceneImageResult(
                scene_id=request.scene_id,
                artifact=ImageArtifact(
                    data=b"PNG-failed",
                    mime_type="image/png",
                    provider="fake-reference-image",
                    model="fake",
                    evidence=(),
                ),
                verified=False,
                issues=("image lacks visual variation",),
            ),
        ), rounds=1)


def test_successful_media_does_not_silently_drop_learning_failure(tmp_path):
    pipeline = MediaProductionPipeline(
        scene_planner=FakeScenePlanner(),
        image_provider=FakeReferenceImageProvider(),
        speech_provider=FakeSpeech(),
        video_builder=FakeVideo(),
        learning_store=FailingLearningStore(),
        audio_verifier=FakeAudioVerifier(),
    )

    import pytest
    with pytest.raises(RuntimeError, match="production learning persistence failed"):
        pipeline.produce(
            "Kịch bản đã được xác minh.",
            tmp_path,
            reference_image=b"REFERENCE-CHARACTER",
        )


def test_primary_image_failure_is_preserved_when_learning_also_fails(tmp_path):
    pipeline = MediaProductionPipeline(
        scene_planner=FakeScenePlanner(),
        image_provider=FailingReferenceImageProvider(),
        speech_provider=FakeSpeech(),
        video_builder=FakeVideo(),
        learning_store=FailingLearningStore(),
        audio_verifier=FakeAudioVerifier(),
    )

    import pytest
    with pytest.raises(RuntimeError) as exc:
        pipeline.produce(
            "Kịch bản đã được xác minh.",
            tmp_path,
            reference_image=b"REFERENCE-CHARACTER",
        )

    message = str(exc.value)
    assert "image verification failed for scenes" in message
    assert "production learning persistence also failed" in message
