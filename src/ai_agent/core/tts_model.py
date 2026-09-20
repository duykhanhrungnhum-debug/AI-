"""Local text-to-speech providers with auditable WAV validation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import tempfile
import wave

from .invariants import assert_core_invariants


@dataclass(frozen=True)
class AudioArtifact:
    data: bytes
    mime_type: str
    provider: str
    model: str
    duration_seconds: float
    sample_rate: int
    channels: int
    evidence: tuple[str, ...]


@dataclass
class PiperTTSProvider:
    """Synthesize speech through a local Piper executable."""

    model_path: str
    binary: str = "piper"
    speaker: int | None = None
    use_cuda: bool = False
    timeout: float = 300.0
    min_duration_seconds: float = 0.05
    provider: str = "piper-local"

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path is required")
        if not self.binary.strip():
            raise ValueError("binary is required")
        if self.speaker is not None and self.speaker < 0:
            raise ValueError("speaker must be non-negative")
        if self.timeout <= 0 or self.min_duration_seconds < 0:
            raise ValueError("timeout must be positive and minimum duration non-negative")

    def synthesize_many(self, texts: list[str] | tuple[str, ...]) -> tuple[AudioArtifact, ...]:
        """Synthesize multiple utterances while loading the Piper voice only once."""
        assert_core_invariants()
        items = tuple(text.strip() for text in texts)
        if not items or any(not text for text in items):
            raise ValueError("texts must contain non-empty strings")

        try:
            from piper import PiperVoice
            from piper.config import SynthesisConfig
        except ImportError as exc:
            raise RuntimeError("piper-tts Python API is required for batch synthesis") from exc

        with tempfile.TemporaryDirectory(prefix="ai-agent-piper-batch-") as temp_dir:
            root = Path(temp_dir)
            paths = [root / f"speech-{index:05d}.wav" for index in range(len(items))]
            voice = PiperVoice.load(self.model_path, use_cuda=self.use_cuda)
            syn_config = SynthesisConfig(speaker_id=self.speaker)

            for text, output in zip(items, paths, strict=True):
                try:
                    with wave.open(str(output), "wb") as wav_file:
                        voice.synthesize_wav(text, wav_file, syn_config=syn_config)
                except Exception as exc:
                    raise RuntimeError(
                        f"Piper Python batch synthesis failed for {output.name}: {exc}"
                    ) from exc

            artifacts: list[AudioArtifact] = []
            for output in paths:
                if not output.exists():
                    raise RuntimeError(f"Piper batch synthesis did not create {output.name}")
                data = output.read_bytes()
                duration, sample_rate, channels = self._inspect_wav(output)
                if duration < self.min_duration_seconds:
                    raise ValueError(
                        f"Piper WAV duration {duration:.3f}s is below minimum "
                        f"{self.min_duration_seconds:.3f}s"
                    )
                digest = sha256(data).hexdigest()
                artifacts.append(AudioArtifact(
                    data=data,
                    mime_type="audio/wav",
                    provider=self.provider,
                    model=self.model_path,
                    duration_seconds=duration,
                    sample_rate=sample_rate,
                    channels=channels,
                    evidence=(
                        f"audio_sha256:{digest}",
                        f"duration_seconds:{duration:.3f}",
                        f"sample_rate:{sample_rate}",
                        f"channels:{channels}",
                        f"batch_size:{len(items)}",
                        "runtime:piper-python-api-single-model-load",
                    ),
                ))
        return tuple(artifacts)

    def synthesize(self, text: str) -> AudioArtifact:
        assert_core_invariants()
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")

        with tempfile.TemporaryDirectory(prefix="ai-agent-piper-") as temp_dir:
            output = Path(temp_dir) / "speech.wav"
            command = [
                self.binary,
                "--model",
                self.model_path,
                "--output_file",
                str(output),
            ]
            if self.speaker is not None:
                command.extend(["--speaker", str(self.speaker)])
            if self.use_cuda:
                command.append("--cuda")

            completed = subprocess.run(
                command,
                input=text,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown Piper failure").strip()
                raise RuntimeError(f"Piper synthesis failed: {detail}")
            if not output.exists():
                raise RuntimeError("Piper exited successfully but did not create the WAV file")

            data = output.read_bytes()
            duration, sample_rate, channels = self._inspect_wav(output)
            if duration < self.min_duration_seconds:
                raise ValueError(
                    f"Piper WAV duration {duration:.3f}s is below minimum "
                    f"{self.min_duration_seconds:.3f}s"
                )

        digest = sha256(data).hexdigest()
        return AudioArtifact(
            data=data,
            mime_type="audio/wav",
            provider=self.provider,
            model=self.model_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            evidence=(
                f"audio_sha256:{digest}",
                f"duration_seconds:{duration:.3f}",
                f"sample_rate:{sample_rate}",
                f"channels:{channels}",
            ),
        )

    @staticmethod
    def _inspect_wav(path: Path) -> tuple[float, int, int]:
        try:
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
        except (wave.Error, EOFError) as exc:
            raise ValueError("Piper output is not a valid WAV file") from exc
        if sample_rate <= 0 or channels <= 0 or frames <= 0:
            raise ValueError("Piper WAV has invalid audio metadata")
        return frames / sample_rate, sample_rate, channels
