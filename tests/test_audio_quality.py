from array import array
from io import BytesIO
import math
import wave

from ai_agent.core.audio_quality import AudioSignalVerifier
from ai_agent.core.tts_model import AudioArtifact


def make_wav(samples, sample_rate=22050):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        pcm = array("h", samples)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def artifact(data):
    return AudioArtifact(
        data=data,
        mime_type="audio/wav",
        provider="test",
        model="voice",
        duration_seconds=1.0,
        sample_rate=22050,
        channels=1,
        evidence=(),
    )


def test_audio_signal_verifier_accepts_clear_signal():
    samples = [
        int(7000 * math.sin(2 * math.pi * 440 * i / 22050))
        for i in range(22050)
    ]
    report = AudioSignalVerifier().verify(artifact(make_wav(samples)))

    assert report.passed is True
    assert report.rms_ratio > 0
    assert report.peak_ratio > 0
    assert any(item.startswith("audio_rms_ratio:") for item in report.evidence)


def test_audio_signal_verifier_rejects_silence():
    report = AudioSignalVerifier().verify(artifact(make_wav([0] * 22050)))

    assert report.passed is False
    assert any("weak" in issue or "silent" in issue for issue in report.issues)


def test_audio_signal_verifier_rejects_heavy_clipping():
    samples = [32767 if i % 2 == 0 else -32767 for i in range(22050)]
    report = AudioSignalVerifier().verify(artifact(make_wav(samples)))

    assert report.passed is False
    assert any("clipping" in issue for issue in report.issues)
