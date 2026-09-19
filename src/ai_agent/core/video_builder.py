"""Local FFmpeg video assembly with ffprobe-based verification."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import tempfile

from .invariants import assert_core_invariants


@dataclass(frozen=True)
class VideoArtifact:
    path: str
    duration_seconds: float
    width: int
    height: int
    has_video: bool
    has_audio: bool
    evidence: tuple[str, ...]


@dataclass
class FFmpegVideoBuilder:
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    timeout: float = 900.0

    def __post_init__(self) -> None:
        if not self.ffmpeg_binary.strip() or not self.ffprobe_binary.strip():
            raise ValueError("ffmpeg and ffprobe binaries are required")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0 or self.timeout <= 0:
            raise ValueError("video dimensions, fps, and timeout must be positive")

    def build(self, image_paths: list[str] | tuple[str, ...], audio_path: str, output_path: str) -> VideoArtifact:
        assert_core_invariants()
        images = tuple(Path(path) for path in image_paths)
        if not images:
            raise ValueError("at least one image is required")
        if any(not path.exists() for path in images):
            raise FileNotFoundError("one or more input images do not exist")

        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError("audio input does not exist")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        audio_probe = self._probe(audio)
        audio_duration = self._duration(audio_probe)
        if audio_duration <= 0:
            raise ValueError("audio duration must be positive")
        duration_per_image = audio_duration / len(images)

        with tempfile.TemporaryDirectory(prefix="ai-agent-video-") as temp_dir:
            concat_file = Path(temp_dir) / "images.txt"
            lines: list[str] = []
            for image in images:
                escaped = str(image.resolve()).replace("'", "'\\''")
                lines.append(f"file '{escaped}'")
                lines.append(f"duration {duration_per_image:.6f}")
            last = str(images[-1].resolve()).replace("'", "'\\''")
            lines.append(f"file '{last}'")
            concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            command = [
                self.ffmpeg_binary,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-i",
                str(audio),
                "-vf",
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2",
                "-r",
                str(self.fps),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(output),
            ]
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown FFmpeg failure").strip()
                raise RuntimeError(f"FFmpeg build failed: {detail}")

        if not output.exists() or output.stat().st_size <= 0:
            raise RuntimeError("FFmpeg exited successfully but did not create a non-empty video")

        output_probe = self._probe(output)
        duration = self._duration(output_probe)
        streams = output_probe.get("streams")
        if not isinstance(streams, list):
            raise ValueError("ffprobe output has no stream list")

        video_stream = next(
            (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"),
            None,
        )
        if video_stream is None or audio_stream is None:
            raise ValueError("video verification requires both video and audio streams")
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        if width != self.width or height != self.height:
            raise ValueError(
                f"video resolution mismatch: expected {self.width}x{self.height}, got {width}x{height}"
            )
        if duration <= 0:
            raise ValueError("output video duration must be positive")

        digest = sha256(output.read_bytes()).hexdigest()
        return VideoArtifact(
            path=str(output),
            duration_seconds=duration,
            width=width,
            height=height,
            has_video=True,
            has_audio=True,
            evidence=(
                f"video_sha256:{digest}",
                f"duration_seconds:{duration:.3f}",
                f"resolution:{width}x{height}",
                "stream:video",
                "stream:audio",
            ),
        )

    def _probe(self, path: Path) -> dict:
        command = [
            self.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown ffprobe failure").strip()
            raise RuntimeError(f"ffprobe failed: {detail}")
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("ffprobe returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("ffprobe output must be a JSON object")
        return data

    @staticmethod
    def _duration(probe: dict) -> float:
        format_data = probe.get("format")
        if not isinstance(format_data, dict):
            return 0.0
        try:
            return float(format_data.get("duration", 0))
        except (TypeError, ValueError):
            return 0.0
