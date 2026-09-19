from pathlib import Path
import json
import subprocess

import pytest

from ai_agent.core.video_builder import FFmpegVideoBuilder


def test_ffmpeg_builder_creates_and_verifies_video(tmp_path, monkeypatch):
    image1 = tmp_path / "1.png"
    image2 = tmp_path / "2.png"
    audio = tmp_path / "voice.wav"
    output = tmp_path / "video.mp4"
    image1.write_bytes(b"IMG1")
    image2.write_bytes(b"IMG2")
    audio.write_bytes(b"AUDIO")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            target = command[-1]
            if target.endswith("voice.wav"):
                payload = {"format": {"duration": "4.0"}, "streams": [{"codec_type": "audio"}]}
            else:
                payload = {
                    "format": {"duration": "4.0"},
                    "streams": [
                        {"codec_type": "video", "width": 1920, "height": 1080},
                        {"codec_type": "audio"},
                    ],
                }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        Path(command[-1]).write_bytes(b"VIDEO")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ai_agent.core.video_builder.subprocess.run", fake_run)

    artifact = FFmpegVideoBuilder().build([str(image1), str(image2)], str(audio), str(output))

    assert artifact.has_video is True
    assert artifact.has_audio is True
    assert artifact.duration_seconds == pytest.approx(4.0)
    assert artifact.width == 1920
    assert artifact.height == 1080
    assert any(item.startswith("video_sha256:") for item in artifact.evidence)
    assert any(command[0] == "ffmpeg" for command in calls)
    assert sum(command[0] == "ffprobe" for command in calls) == 2


def test_ffmpeg_builder_rejects_missing_audio_stream(tmp_path, monkeypatch):
    image = tmp_path / "1.png"
    audio = tmp_path / "voice.wav"
    output = tmp_path / "video.mp4"
    image.write_bytes(b"IMG")
    audio.write_bytes(b"AUDIO")

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"VIDEO")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        target = command[-1]
        if target.endswith("voice.wav"):
            payload = {"format": {"duration": "2.0"}, "streams": [{"codec_type": "audio"}]}
        else:
            payload = {
                "format": {"duration": "2.0"},
                "streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
            }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("ai_agent.core.video_builder.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="both video and audio"):
        FFmpegVideoBuilder().build([str(image)], str(audio), str(output))


def test_ffmpeg_builder_surfaces_process_failure(tmp_path, monkeypatch):
    image = tmp_path / "1.png"
    audio = tmp_path / "voice.wav"
    output = tmp_path / "video.mp4"
    image.write_bytes(b"IMG")
    audio.write_bytes(b"AUDIO")

    def fake_run(command, **kwargs):
        if command[0] == "ffprobe":
            payload = {"format": {"duration": "2.0"}, "streams": [{"codec_type": "audio"}]}
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="encoder failed")

    monkeypatch.setattr("ai_agent.core.video_builder.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="encoder failed"):
        FFmpegVideoBuilder().build([str(image)], str(audio), str(output))
