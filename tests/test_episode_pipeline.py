from pathlib import Path

from ai_agent.core.audio_quality import AudioQualityReport
from ai_agent.core.episode_pipeline import MediaProductionPipeline
from ai_agent.core.image_model import ImageArtifact
from ai_agent.core.kaggle_image_batch import BatchImageResult, SceneImageResult
from ai_agent.core.scene_planner import VisualScene, VisualScenePlan
from ai_agent.core.tts_model import AudioArtifact
from ai_agent.core.video_builder import VideoArtifact


class FakeScenePlanner:
    def plan(self, script, *, visual_style):
        return VisualScenePlan((
            VisualScene("s1", "n1", "prompt one, 16:9 widescreen composition"),
            VisualScene("s2", "n2", "prompt two, 16:9 widescreen composition"),
        ))


class FakeImageProvider:
    def __init__(self, verified=True):
        self.requests = None
        self.verified = verified

    def generate_with_retries(self, requests, *, max_rounds):
        self.requests = list(requests)
        scenes = []
        for request in requests:
            issues = () if self.verified else ("bad image",)
            scenes.append(SceneImageResult(
                scene_id=request.scene_id,
                artifact=ImageArtifact(
                    data=b"PNG" + request.scene_id.encode(),
                    mime_type="image/png",
                    provider="fake-image",
                    model="fake",
                    evidence=(f"scene:{request.scene_id}",),
                ),
                verified=self.verified,
                issues=issues,
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


def test_media_pipeline_connects_script_to_verified_media(tmp_path):
    images = FakeImageProvider()
    pipeline = MediaProductionPipeline(
        scene_planner=FakeScenePlanner(),
        image_provider=images,
        speech_provider=FakeSpeech(),
        video_builder=FakeVideo(),
        audio_verifier=FakeAudioVerifier(),
    )

    result = pipeline.produce("Đây là kịch bản đã được kiểm tra.", tmp_path)

    assert result.media_ready is True
    assert len(result.scene_plan.scenes) == 2
    assert len(images.requests) == 2
    assert all(request.width == 768 and request.height == 432 for request in images.requests)
    assert (tmp_path / "scene_0000.png").exists()
    assert (tmp_path / "scene_0001.png").exists()
    assert (tmp_path / "narration.wav").exists()
    assert (tmp_path / "video.mp4").exists()
    assert any(item.startswith("script_sha256:") for item in result.evidence)
    assert "scene_count:2" in result.evidence
    assert result.audio_quality.passed is True
    assert "audio:verified" in result.evidence
    assert "audio_signal:verified" in result.evidence
    assert "video:verified" in result.evidence
