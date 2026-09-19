from pathlib import Path
import subprocess
import wave

import pytest

from ai_agent.core.tts_model import PiperTTSProvider


def write_wav(path: Path, *, seconds=1.0, sample_rate=22050):
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def test_piper_provider_runs_local_binary_and_validates_wav(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        output = Path(command[command.index("--output_file") + 1])
        write_wav(output)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ai_agent.core.tts_model.subprocess.run", fake_run)
    provider = PiperTTSProvider(model_path="/models/vi.onnx", speaker=1, use_cuda=True)

    artifact = provider.synthesize("Xin chào thế giới")

    assert captured["input"] == "Xin chào thế giới"
    assert captured["command"][:3] == ["piper", "--model", "/models/vi.onnx"]
    assert "--speaker" in captured["command"]
    assert "--cuda" in captured["command"]
    assert artifact.provider == "piper-local"
    assert artifact.mime_type == "audio/wav"
    assert artifact.duration_seconds == pytest.approx(1.0)
    assert artifact.sample_rate == 22050
    assert artifact.channels == 1
    assert any(item.startswith("audio_sha256:") for item in artifact.evidence)


def test_piper_provider_surfaces_process_failure(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="voice model failed")

    monkeypatch.setattr("ai_agent.core.tts_model.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="voice model failed"):
        PiperTTSProvider(model_path="/models/vi.onnx").synthesize("Xin chào")


def test_piper_provider_rejects_empty_audio(monkeypatch):
    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output_file") + 1])
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(b"")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ai_agent.core.tts_model.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="invalid audio metadata"):
        PiperTTSProvider(model_path="/models/vi.onnx").synthesize("Xin chào")
