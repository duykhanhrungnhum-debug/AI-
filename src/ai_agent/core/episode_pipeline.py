"""Compose verified script, scene planning, image batch, TTS, and video assembly."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from .audio_quality import AudioQualityReport, AudioSignalVerifier
from .image_model import ImageArtifact
from .kaggle_image_batch import (
    BatchImageResult,
    KaggleBatchImageProvider,
    SceneImageRequest,
)
from .scene_planner import ScenePlanner, VisualScenePlan
from .tts_model import AudioArtifact
from .video_builder import VideoArtifact
from .invariants import assert_core_invariants


class SpeechProvider(Protocol):
    def synthesize(self, text: str) -> AudioArtifact:
        """Synthesize narration audio."""


class VideoAssembler(Protocol):
    def build(self, image_paths: list[str] | tuple[str, ...], audio_path: str, output_path: str) -> VideoArtifact:
        """Assemble images and narration into a video."""


@dataclass(frozen=True)
class MediaProductionResult:
    scene_plan: VisualScenePlan
    images: BatchImageResult
    audio: AudioArtifact
    audio_quality: AudioQualityReport
    video: VideoArtifact
    output_dir: str
    evidence: tuple[str, ...]

    @property
    def media_ready(self) -> bool:
        return (
            self.images.verified
            and self.audio_quality.passed
            and self.video.has_video
            and self.video.has_audio
        )


@dataclass
class MediaProductionPipeline:
    scene_planner: ScenePlanner
    image_provider: KaggleBatchImageProvider
    speech_provider: SpeechProvider
    video_builder: VideoAssembler
    image_width: int = 768
    image_height: int = 432
    max_image_rounds: int = 2
    audio_verifier: AudioSignalVerifier = field(default_factory=AudioSignalVerifier)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.image_width % 8 or self.image_height % 8:
            raise ValueError("image dimensions must be divisible by 8")
        if self.max_image_rounds <= 0:
            raise ValueError("max_image_rounds must be positive")

    def produce(
        self,
        verified_script: str,
        output_dir: str | Path,
        *,
        visual_style: str = "cinematic realistic",
    ) -> MediaProductionResult:
        """Produce media from a script that already passed the narrative verifier."""
        assert_core_invariants()
        verified_script = verified_script.strip()
        if not verified_script:
            raise ValueError("verified_script must not be empty")

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        scene_plan = self.scene_planner.plan(verified_script, visual_style=visual_style)
        requests = [
            SceneImageRequest(
                scene_id=scene.scene_id,
                prompt=scene.image_prompt,
                negative_prompt=scene.negative_prompt,
                width=self.image_width,
                height=self.image_height,
                seed=self._seed(scene.scene_id),
            )
            for scene in scene_plan.scenes
        ]
        image_result = self.image_provider.generate_with_retries(
            requests,
            max_rounds=self.max_image_rounds,
        )
        if not image_result.verified:
            raise RuntimeError(
                "image verification failed for scenes: "
                + ", ".join(image_result.failed_scene_ids)
            )

        image_paths: list[str] = []
        image_evidence: list[str] = []
        for index, scene in enumerate(image_result.scenes):
            path = target / f"scene_{index:04d}.png"
            path.write_bytes(scene.artifact.data)
            image_paths.append(str(path))
            image_evidence.extend(scene.artifact.evidence)

        audio = self.speech_provider.synthesize(verified_script)
        audio_path = target / "narration.wav"
        audio_path.write_bytes(audio.data)
        audio_quality = self.audio_verifier.verify(audio)
        if not audio_quality.passed:
            raise RuntimeError(
                "audio verification failed: " + "; ".join(audio_quality.issues)
            )

        video_path = target / "video.mp4"
        video = self.video_builder.build(image_paths, str(audio_path), str(video_path))

        evidence = (
            f"script_sha256:{sha256(verified_script.encode('utf-8')).hexdigest()}",
            f"scene_count:{len(scene_plan.scenes)}",
            f"image_rounds:{image_result.rounds}",
            *image_evidence,
            *audio.evidence,
            *audio_quality.evidence,
            *video.evidence,
        )
        return MediaProductionResult(
            scene_plan=scene_plan,
            images=image_result,
            audio=audio,
            audio_quality=audio_quality,
            video=video,
            output_dir=str(target),
            evidence=evidence,
        )

    @staticmethod
    def _seed(scene_id: str) -> int:
        digest = sha256(scene_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")
