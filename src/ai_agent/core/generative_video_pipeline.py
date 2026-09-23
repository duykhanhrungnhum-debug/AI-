"""Verified script-to-motion-video pipeline using an on-demand local video model."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from .audio_quality import AudioQualityReport, AudioSignalVerifier
from .kaggle_video import BatchVideoResult, SceneVideoRequest
from .scene_planner import ScenePlanner, VisualScenePlan
from .tts_model import AudioArtifact
from .video_builder import VideoArtifact
from .invariants import assert_core_invariants


class SpeechProvider(Protocol):
    def synthesize(self, text: str) -> AudioArtifact:
        """Synthesize narration audio."""


class MotionVideoProvider(Protocol):
    def generate_with_retries(
        self,
        requests: tuple[SceneVideoRequest, ...] | list[SceneVideoRequest],
        *,
        max_rounds: int = 2,
    ) -> BatchVideoResult:
        """Generate verified motion clips."""


class ClipVideoAssembler(Protocol):
    def build_from_clips(
        self,
        clip_paths: list[str] | tuple[str, ...],
        audio_path: str,
        output_path: str,
    ) -> VideoArtifact:
        """Loop/normalize clips, concatenate them, and mux narration."""


@dataclass(frozen=True)
class GenerativeVideoResult:
    scene_plan: VisualScenePlan
    clips: BatchVideoResult
    audio: AudioArtifact
    audio_quality: AudioQualityReport
    video: VideoArtifact
    output_dir: str
    evidence: tuple[str, ...]

    @property
    def media_ready(self) -> bool:
        return (
            self.clips.verified
            and self.audio_quality.passed
            and self.video.has_video
            and self.video.has_audio
        )


@dataclass
class GenerativeVideoPipeline:
    """Create moving video scenes rather than a slideshow, then add verified narration."""

    scene_planner: ScenePlanner
    video_provider: MotionVideoProvider
    speech_provider: SpeechProvider
    video_builder: ClipVideoAssembler
    width: int = 832
    height: int = 480
    num_frames: int = 17
    fps: int = 16
    max_video_rounds: int = 2
    audio_verifier: AudioSignalVerifier = field(default_factory=AudioSignalVerifier)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("video dimensions must be positive and divisible by 16")
        if self.num_frames <= 0 or (self.num_frames - 1) % 4:
            raise ValueError("num_frames must follow Wan's 4*k+1 rule")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_video_rounds <= 0:
            raise ValueError("max_video_rounds must be positive")

    def produce(
        self,
        verified_script: str,
        output_dir: str | Path,
        *,
        visual_style: str = "cinematic realistic",
    ) -> GenerativeVideoResult:
        assert_core_invariants()
        verified_script = verified_script.strip()
        if not verified_script:
            raise ValueError("verified_script must not be empty")

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        scene_plan = self.scene_planner.plan(verified_script, visual_style=visual_style)
        requests = [
            SceneVideoRequest(
                scene_id=scene.scene_id,
                prompt=self._motion_prompt(scene.image_prompt, visual_style),
                negative_prompt=scene.negative_prompt,
                width=self.width,
                height=self.height,
                num_frames=self.num_frames,
                fps=self.fps,
                seed=self._seed(scene.scene_id),
            )
            for scene in scene_plan.scenes
        ]
        clips = self.video_provider.generate_with_retries(
            requests,
            max_rounds=self.max_video_rounds,
        )
        if not clips.verified:
            raise RuntimeError(
                "video verification failed for scenes: "
                + ", ".join(clips.failed_scene_ids)
            )

        clip_paths: list[str] = []
        clip_evidence: list[str] = []
        for index, scene in enumerate(clips.scenes):
            path = target / f"scene_{index:04d}.mp4"
            path.write_bytes(scene.artifact.data)
            clip_paths.append(str(path))
            clip_evidence.extend(scene.artifact.evidence)

        audio = self.speech_provider.synthesize(verified_script)
        audio_path = target / "narration.wav"
        audio_path.write_bytes(audio.data)
        audio_quality = self.audio_verifier.verify(audio)
        if not audio_quality.passed:
            raise RuntimeError(
                "audio verification failed: " + "; ".join(audio_quality.issues)
            )

        video_path = target / "video.mp4"
        video = self.video_builder.build_from_clips(
            clip_paths,
            str(audio_path),
            str(video_path),
        )

        evidence = (
            f"script_sha256:{sha256(verified_script.encode('utf-8')).hexdigest()}",
            f"scene_count:{len(scene_plan.scenes)}",
            f"video_rounds:{clips.rounds}",
            "visual_mode:generative_motion",
            *clip_evidence,
            *audio.evidence,
            *audio_quality.evidence,
            *video.evidence,
        )
        return GenerativeVideoResult(
            scene_plan=scene_plan,
            clips=clips,
            audio=audio,
            audio_quality=audio_quality,
            video=video,
            output_dir=str(target),
            evidence=evidence,
        )

    @staticmethod
    def _motion_prompt(image_prompt: str, visual_style: str) -> str:
        return (
            f"{image_prompt}. {visual_style}. "
            "Continuous cinematic live-action motion, natural body movement, subtle environmental motion, "
            "coherent anatomy, stable identity, consistent clothing, realistic camera movement, "
            "one connected shot, no cuts, no text, no subtitles, no watermark."
        )

    @staticmethod
    def _seed(scene_id: str) -> int:
        digest = sha256(("video:" + scene_id).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")
