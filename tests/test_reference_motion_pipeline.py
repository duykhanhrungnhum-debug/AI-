from pathlib import Path

from ai_agent.core.audio_quality import AudioQualityReport
from ai_agent.core.image_model import ImageArtifact
from ai_agent.core.kaggle_i2v import (
    BatchImageToVideoResult,
    SceneImageToVideoArtifact,
    SceneImageToVideoResult,
)
from ai_agent.core.kaggle_image_batch import BatchImageResult, SceneImageResult
from ai_agent.core.production_learning import ProductionLesson
from ai_agent.core.reference_motion_pipeline import ReferenceMotionVideoPipeline
from ai_agent.core.scene_planner import VisualScene, VisualScenePlan
from ai_agent.core.tts_model import AudioArtifact
from ai_agent.core.video_builder import VideoArtifact


class FakePlanner:
    def plan(self, script, *, visual_style, composition):
        assert composition == "16:9 widescreen"
        return VisualScenePlan((
            VisualScene("s1", "n1", "hero walks through an old library"),
            VisualScene("s2", "n2", "hero studies an antique map"),
        ))


class FakeImages:
    ip_adapter_weight = "ip-adapter-full-face_sd15.bin"

    def __init__(self):
        self.reference_scale = None

    def generate_with_retries(
        self,
        requests,
        *,
        max_rounds,
        reference_image=None,
        reference_scale=0.75,
    ):
        self.reference_scale = reference_scale
        scenes = []
        for index, request in enumerate(requests):
            scenes.append(SceneImageResult(
                scene_id=request.scene_id,
                artifact=ImageArtifact(
                    data=b"PNG-" + request.scene_id.encode(),
                    mime_type="image/png",
                    provider="fake-keyframe",
                    model="fake",
                    evidence=(
                        f"identity_score:{0.82 + index * 0.01:.6f}",
                        f"reference_scale:{reference_scale:.3f}",
                    ),
                ),
                verified=True,
                issues=(),
            ))
        return BatchImageResult(tuple(scenes), rounds=1)


class FakeMotion:
    def __init__(self):
        self.images = []

    def generate_with_retries(self, requests, *, max_rounds):
        scenes = []
        for index, request in enumerate(requests):
            self.images.append(request.image)
            scenes.append(SceneImageToVideoResult(
                scene_id=request.scene_id,
                artifact=SceneImageToVideoArtifact(
                    data=b"\x00\x00\x00\x18ftypisom" + bytes([65 + index]) * 64,
                    mime_type="video/mp4",
                    provider="fake-i2v",
                    model="fake-svd",
                    duration_seconds=2.0,
                    width=request.width,
                    height=request.height,
                    fps=float(request.fps),
                    evidence=(
                        "first_frame_similarity:0.910000",
                        f"last_frame_similarity:{0.79 - index * 0.01:.6f}",
                        f"motion_delta:{7.0 + index:.6f}",
                    ),
                ),
                verified=True,
                issues=(),
            ))
        return BatchImageToVideoResult(tuple(scenes), rounds=1)


class FakeSpeech:
    def synthesize(self, text):
        return AudioArtifact(
            data=b"WAV",
            mime_type="audio/wav",
            provider="fake-tts",
            model="voice",
            duration_seconds=4.0,
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


class FakeBuilder:
    def build_from_clips(self, clip_paths, audio_path, output_path):
        assert len(clip_paths) == 2
        assert all(Path(path).exists() for path in clip_paths)
        assert Path(audio_path).read_bytes() == b"WAV"
        Path(output_path).write_bytes(b"MP4")
        return VideoArtifact(
            path=output_path,
            duration_seconds=4.0,
            width=1280,
            height=720,
            has_video=True,
            has_audio=True,
            evidence=("video:verified",),
        )


class FakeLearning:
    def __init__(self):
        self.recorded = []

    def relevant(self, task_type, *, limit=10):
        assert task_type == "reference-motion-video"
        return ()

    def record(self, lesson):
        self.recorded.append(lesson)
        return lesson


def test_reference_motion_pipeline_produces_verified_video_and_learns(tmp_path):
    images = FakeImages()
    motion = FakeMotion()
    learning = FakeLearning()
    pipeline = ReferenceMotionVideoPipeline(
        scene_planner=FakePlanner(),
        image_provider=images,
        motion_provider=motion,
        speech_provider=FakeSpeech(),
        video_builder=FakeBuilder(),
        learning_store=learning,
        audio_verifier=FakeAudioVerifier(),
    )

    result = pipeline.produce(
        "Kịch bản thật đã xác minh.",
        b"REFERENCE-CHARACTER",
        tmp_path,
    )

    assert result.media_ready is True
    assert images.reference_scale == 0.50
    assert motion.images == [b"PNG-s1", b"PNG-s2"]
    assert (tmp_path / "scene_0000_keyframe.png").exists()
    assert (tmp_path / "scene_0000.mp4").exists()
    assert any(x == "visual_mode:reference_conditioned_motion_video" for x in result.evidence)
    assert learning.recorded[-1].success is True
    assert learning.recorded[-1].metrics["min_keyframe_identity"] == 0.82
    assert learning.recorded[-1].metrics["min_last_motion_identity"] == 0.78
    assert learning.recorded[-1].metrics["mean_motion_delta"] == 7.5


class FailedMotion(FakeMotion):
    def generate_with_retries(self, requests, *, max_rounds):
        request = list(requests)[0]
        return BatchImageToVideoResult((
            SceneImageToVideoResult(
                scene_id=request.scene_id,
                artifact=SceneImageToVideoArtifact(
                    data=b"bad",
                    mime_type="video/mp4",
                    provider="fake-i2v",
                    model="fake",
                    duration_seconds=0.0,
                    width=request.width,
                    height=request.height,
                    fps=float(request.fps),
                    evidence=(),
                ),
                verified=False,
                issues=("identity similarity below threshold",),
            ),
        ), rounds=2)


def test_reference_motion_pipeline_records_motion_failure(tmp_path):
    learning = FakeLearning()
    pipeline = ReferenceMotionVideoPipeline(
        scene_planner=FakePlanner(),
        image_provider=FakeImages(),
        motion_provider=FailedMotion(),
        speech_provider=FakeSpeech(),
        video_builder=FakeBuilder(),
        learning_store=learning,
        audio_verifier=FakeAudioVerifier(),
    )

    import pytest
    with pytest.raises(RuntimeError, match="motion verification failed"):
        pipeline.produce(
            "Kịch bản thật đã xác minh.",
            b"REFERENCE-CHARACTER",
            tmp_path,
        )

    assert learning.recorded[-1].success is False
    assert learning.recorded[-1].failure_kind == "motion_identity_drift"
