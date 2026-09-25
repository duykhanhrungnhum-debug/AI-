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
from .production_learning import ProductionLearningStore, ProductionLesson, tuned_reference_scale
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
    learning_store: ProductionLearningStore | None = None
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
        reference_image: bytes | None = None,
    ) -> MediaProductionResult:
        """Produce media from a script that already passed the narrative verifier."""
        assert_core_invariants()
        verified_script = verified_script.strip()
        if not verified_script:
            raise ValueError("verified_script must not be empty")

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        lessons = ()
        adapter_weight = str(getattr(self.image_provider, "ip_adapter_weight", "") or "")
        base_reference_scale = 0.50 if "full-face" in adapter_weight else 0.75
        reference_scale = base_reference_scale
        reference_hash = None
        if reference_image is not None:
            if not reference_image:
                raise ValueError("reference_image must not be empty")
            reference_hash = sha256(reference_image).hexdigest()
            if self.learning_store is not None:
                try:
                    lessons = self.learning_store.relevant("reference-character-video", limit=10)
                except Exception:
                    lessons = ()
            reference_scale = tuned_reference_scale(lessons, base=base_reference_scale)

        scene_plan = self.scene_planner.plan(verified_script, visual_style=visual_style)
        requests = [
            SceneImageRequest(
                scene_id=scene.scene_id,
                prompt=(
                    scene.image_prompt
                    if reference_image is None
                    else scene.image_prompt
                    + ". Keep the main recurring character consistent with the supplied reference image: "
                    "same face, age, hair, body proportions, costume identity and distinctive features."
                ),
                negative_prompt=(
                    scene.negative_prompt
                    if reference_image is None
                    else scene.negative_prompt
                    + ", different person, changed face, face drift, inconsistent character identity"
                ),
                width=self.image_width,
                height=self.image_height,
                seed=self._seed(scene.scene_id),
            )
            for scene in scene_plan.scenes
        ]
        if reference_image is None:
            image_result = self.image_provider.generate_with_retries(
                requests,
                max_rounds=self.max_image_rounds,
            )
        else:
            image_result = self.image_provider.generate_with_retries(
                requests,
                max_rounds=self.max_image_rounds,
                reference_image=reference_image,
                reference_scale=reference_scale,
            )
        if not image_result.verified:
            if reference_image is not None:
                issues = tuple(issue for item in image_result.scenes for issue in item.issues)
                self._record_learning(ProductionLesson(
                    task_type="reference-character-video",
                    success=False,
                    failure_kind=(
                        "identity_drift"
                        if any("identity similarity" in issue for issue in issues)
                        else "model_output"
                    ),
                    lesson="Reference-conditioned scene generation failed verification; tune only failed factors next run.",
                    config={"reference_scale": reference_scale},
                    metrics={"failed_scene_ids": list(image_result.failed_scene_ids)},
                    evidence=issues[:20],
                ))
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

        identity_scores = []
        for item in image_evidence:
            if item.startswith("identity_score:"):
                try:
                    identity_scores.append(float(item.split(":", 1)[1]))
                except ValueError:
                    pass

        reference_evidence = ()
        if reference_hash is not None:
            reference_evidence = (
                f"reference_sha256:{reference_hash}",
                f"reference_scale:{reference_scale:.3f}",
                f"production_lessons_loaded:{len(lessons)}",
                f"min_identity_score:{min(identity_scores):.6f}" if identity_scores else "min_identity_score:unknown",
                "visual_mode:reference_conditioned_story_video",
            )

        evidence = (
            f"script_sha256:{sha256(verified_script.encode('utf-8')).hexdigest()}",
            f"scene_count:{len(scene_plan.scenes)}",
            f"image_rounds:{image_result.rounds}",
            *reference_evidence,
            *image_evidence,
            *audio.evidence,
            *audio_quality.evidence,
            *video.evidence,
        )
        if reference_hash is not None:
            self._record_learning(ProductionLesson(
                task_type="reference-character-video",
                success=True,
                lesson=(
                    "Verified reference-conditioned settings produced a complete narrated video; "
                    "reuse them unless later identity checks regress."
                ),
                config={"reference_scale": reference_scale},
                metrics={
                    "scene_count": len(scene_plan.scenes),
                    "min_identity_score": min(identity_scores) if identity_scores else None,
                    "mean_identity_score": (
                        sum(identity_scores) / len(identity_scores) if identity_scores else None
                    ),
                    "video_duration_seconds": video.duration_seconds,
                },
                evidence=evidence[:40],
            ))

        return MediaProductionResult(
            scene_plan=scene_plan,
            images=image_result,
            audio=audio,
            audio_quality=audio_quality,
            video=video,
            output_dir=str(target),
            evidence=evidence,
        )

    def _record_learning(self, lesson: ProductionLesson) -> None:
        if self.learning_store is None:
            return
        try:
            self.learning_store.record(lesson)
        except Exception:
            pass

    @staticmethod
    def _seed(scene_id: str) -> int:
        digest = sha256(scene_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")
