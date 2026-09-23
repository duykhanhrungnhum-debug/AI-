from io import BytesIO
from pathlib import Path
import math
import struct
import wave

from ai_agent.core.generative_video_pipeline import GenerativeVideoPipeline
from ai_agent.core.kaggle_video import (
    BatchVideoResult,
    SceneVideoArtifact,
    SceneVideoResult,
)
from ai_agent.core.scene_planner import VisualScene, VisualScenePlan
from ai_agent.core.tts_model import AudioArtifact
from ai_agent.core.video_builder import FFmpegVideoBuilder, VideoArtifact


def _wav_bytes(seconds=1.0, sample_rate=8000):
    out = BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = []
        for i in range(int(seconds * sample_rate)):
            value = int(5000 * math.sin(2 * math.pi * 220 * i / sample_rate))
            frames.append(struct.pack("<h", value))
        wav.writeframes(b"".join(frames))
    return out.getvalue()


class FakePlanner:
    def plan(self, script, *, visual_style, composition="16:9 widescreen"):
        return VisualScenePlan((
            VisualScene("s1", "one", "Lan walks toward the food stall, widescreen"),
            VisualScene("s2", "two", "Tu hides behind a motorbike, widescreen"),
        ))


class FakeMotion:
    def generate_with_retries(self, requests, *, max_rounds=2):
        scenes = []
        for index, request in enumerate(requests):
            artifact = SceneVideoArtifact(
                data=b"\x00\x00\x00\x18ftypisom" + bytes([65 + index]) * 64,
                mime_type="video/mp4",
                provider="fake-motion",
                model="fake-model",
                duration_seconds=1.0,
                width=request.width,
                height=request.height,
                fps=float(request.fps),
                evidence=(f"video_sha256:fake-{index}",),
            )
            scenes.append(SceneVideoResult(request.scene_id, artifact, True, ()))
        return BatchVideoResult(tuple(scenes), rounds=1)


class FakeSpeech:
    def synthesize(self, text):
        data = _wav_bytes()
        return AudioArtifact(
            data=data,
            mime_type="audio/wav",
            provider="fake-speech",
            model="fake",
            duration_seconds=1.0,
            sample_rate=8000,
            channels=1,
            evidence=("audio:fake",),
        )


class FakeBuilder:
    def build_from_clips(self, clip_paths, audio_path, output_path):
        assert len(clip_paths) == 2
        assert all(Path(path).exists() for path in clip_paths)
        assert Path(audio_path).exists()
        Path(output_path).write_bytes(b"final")
        return VideoArtifact(
            path=output_path,
            duration_seconds=1.0,
            width=832,
            height=480,
            has_video=True,
            has_audio=True,
            evidence=("video:fake",),
        )


def test_generative_video_pipeline_produces_verified_motion_video(tmp_path):
    pipeline = GenerativeVideoPipeline(
        scene_planner=FakePlanner(),
        video_provider=FakeMotion(),
        speech_provider=FakeSpeech(),
        video_builder=FakeBuilder(),
        width=832,
        height=480,
        num_frames=17,
        fps=16,
    )
    result = pipeline.produce(
        "Lan mua đồ ăn rồi chị Tư chạy trốn chồng.",
        tmp_path,
        visual_style="realistic Vietnamese street comedy",
    )

    assert result.media_ready is True
    assert len(result.clips.scenes) == 2
    assert (tmp_path / "scene_0000.mp4").exists()
    assert (tmp_path / "scene_0001.mp4").exists()
    assert (tmp_path / "narration.wav").exists()
    assert "visual_mode:generative_motion" in result.evidence


def test_ffmpeg_builder_builds_from_generated_clips(tmp_path, monkeypatch):
    clip1 = tmp_path / "1.mp4"
    clip2 = tmp_path / "2.mp4"
    audio = tmp_path / "voice.wav"
    output = tmp_path / "video.mp4"
    clip1.write_bytes(b"CLIP1")
    clip2.write_bytes(b"CLIP2")
    audio.write_bytes(b"AUDIO")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            target = command[-1]
            if target.endswith("voice.wav"):
                payload = {"format": {"duration": "4.0"}, "streams": [{"codec_type": "audio"}]}
            elif target.endswith("video.mp4"):
                payload = {
                    "format": {"duration": "4.0"},
                    "streams": [
                        {"codec_type": "video", "width": 1920, "height": 1080},
                        {"codec_type": "audio"},
                    ],
                }
            else:
                payload = {
                    "format": {"duration": "1.0"},
                    "streams": [{"codec_type": "video", "width": 832, "height": 480}],
                }
            return __import__("subprocess").CompletedProcess(
                command, 0, stdout=__import__("json").dumps(payload), stderr=""
            )

        Path(command[-1]).write_bytes(b"VIDEO")
        return __import__("subprocess").CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ai_agent.core.video_builder.subprocess.run", fake_run)

    artifact = FFmpegVideoBuilder().build_from_clips(
        [str(clip1), str(clip2)],
        str(audio),
        str(output),
    )

    assert artifact.has_video is True
    assert artifact.has_audio is True
    assert "source_clip_count:2" in artifact.evidence
    assert "visual_mode:generated_motion_clips" in artifact.evidence
    assert sum(command[0] == "ffmpeg" for command in calls) == 4
