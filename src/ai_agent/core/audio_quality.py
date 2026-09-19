"""Deterministic audio quality checks for generated narration."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from io import BytesIO
import math
import sys
import wave

from .invariants import assert_core_invariants
from .tts_model import AudioArtifact


@dataclass(frozen=True)
class AudioQualityReport:
    passed: bool
    issues: tuple[str, ...]
    rms_ratio: float
    peak_ratio: float
    silence_ratio: float
    evidence: tuple[str, ...]


@dataclass
class AudioSignalVerifier:
    """Reject silent, clipped, malformed, or implausibly weak narration audio."""

    min_duration_seconds: float = 0.2
    min_rms_ratio: float = 0.002
    max_clipping_ratio: float = 0.05
    max_silence_ratio: float = 0.985
    silence_threshold_ratio: float = 0.003

    def __post_init__(self) -> None:
        if self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be non-negative")
        if not 0 <= self.min_rms_ratio <= 1:
            raise ValueError("min_rms_ratio must be between 0 and 1")
        if not 0 <= self.max_clipping_ratio <= 1:
            raise ValueError("max_clipping_ratio must be between 0 and 1")
        if not 0 <= self.max_silence_ratio <= 1:
            raise ValueError("max_silence_ratio must be between 0 and 1")
        if not 0 <= self.silence_threshold_ratio <= 1:
            raise ValueError("silence_threshold_ratio must be between 0 and 1")

    def verify(self, artifact: AudioArtifact) -> AudioQualityReport:
        assert_core_invariants()
        issues: list[str] = []
        try:
            with wave.open(BytesIO(artifact.data), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
                frames = wav.getnframes()
                raw = wav.readframes(frames)
        except (wave.Error, EOFError) as exc:
            raise ValueError("audio artifact is not a valid WAV file") from exc

        if channels <= 0 or sample_rate <= 0 or frames <= 0:
            issues.append("invalid WAV metadata")
        duration = frames / sample_rate if sample_rate > 0 else 0.0
        if duration < self.min_duration_seconds:
            issues.append(
                f"audio duration too short: {duration:.3f}s < {self.min_duration_seconds:.3f}s"
            )

        if sample_width != 2:
            issues.append(f"unsupported PCM sample width: {sample_width} bytes")
            return AudioQualityReport(
                passed=False,
                issues=tuple(issues),
                rms_ratio=0.0,
                peak_ratio=0.0,
                silence_ratio=1.0,
                evidence=(
                    f"audio_duration_seconds:{duration:.3f}",
                    f"audio_sample_width_bytes:{sample_width}",
                ),
            )

        samples = array("h")
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            issues.append("audio contains no PCM samples")
            return AudioQualityReport(
                False,
                tuple(issues),
                0.0,
                0.0,
                1.0,
                (f"audio_duration_seconds:{duration:.3f}",),
            )

        maximum = 32767.0
        abs_values = [abs(value) for value in samples]
        peak_ratio = max(abs_values) / maximum
        rms_ratio = math.sqrt(sum(value * value for value in samples) / len(samples)) / maximum
        silence_limit = maximum * self.silence_threshold_ratio
        silence_ratio = sum(value <= silence_limit for value in abs_values) / len(abs_values)
        clipping_ratio = sum(value >= maximum * 0.999 for value in abs_values) / len(abs_values)

        if rms_ratio < self.min_rms_ratio:
            issues.append(f"audio signal too weak: rms_ratio={rms_ratio:.6f}")
        if silence_ratio > self.max_silence_ratio:
            issues.append(f"audio mostly silent: silence_ratio={silence_ratio:.6f}")
        if clipping_ratio > self.max_clipping_ratio:
            issues.append(f"audio clipping too high: clipping_ratio={clipping_ratio:.6f}")

        evidence = (
            f"audio_duration_seconds:{duration:.3f}",
            f"audio_rms_ratio:{rms_ratio:.6f}",
            f"audio_peak_ratio:{peak_ratio:.6f}",
            f"audio_silence_ratio:{silence_ratio:.6f}",
            f"audio_clipping_ratio:{clipping_ratio:.6f}",
            f"audio_signal_sample_rate:{sample_rate}",
            f"audio_signal_channels:{channels}",
        )
        return AudioQualityReport(
            passed=not issues,
            issues=tuple(issues),
            rms_ratio=rms_ratio,
            peak_ratio=peak_ratio,
            silence_ratio=silence_ratio,
            evidence=evidence,
        )
