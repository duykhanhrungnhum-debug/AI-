"""Reference-character script-to-motion-video production pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from .audio_quality import AudioQualityReport, AudioSignalVerifier
from .kaggle_i2v import (
    BatchImageToVideoResult,
    KaggleBatchImageToVideoProvider,
    SceneImageToVideoRequest,
)
from .kaggle_image_batch import (
    BatchImageResult,
    KaggleBatchImageProvider,
    SceneImageRequest,
)
from .production_learning import ProductionLearningStore, ProductionLesson, tuned_reference_scale
from .scene_planner import ScenePlanner, VisualScenePlan
from .tts_model import AudioArtifact
from .video_builder import VideoArtifact
from .invariants import assert_core_invariants


class SpeechProvider(Protocol):
    def synthesize(self, text: str) -> AudioArtifact:
        ...


class ClipAssembler(Protocol):
    def build_from_clips(
        self,
        clip_paths: list[str] | tuple[str, ...],
        audio_path: str,
        output_path: str,
    ) -> VideoArtifact:
        ...


@dataclass(frozen=True)
class ReferenceMotionVideoResult:
    scene_plan: VisualScenePlan
    keyframes: BatchImageResult
    clips: BatchImageToVideoResult
    audio: AudioArtifact
    audio_quality: AudioQualityReport
    video: VideoArtifact
    output_dir: str
    evidence: tuple[str, ...]

    @property
    def media_ready(self) -> bool:
        return (
            self.keyframes.verified
            and self.clips.verified
            and self.audio_quality.passed
            and self.video.has_video
            and self.video.has_audio
        )


@dataclass
class ReferenceMotionVideoPipeline:
    """Plan scenes, preserve the reference character, animate scenes, narrate, and assemble."""

    scene_planner: ScenePlanner
    image_provider: KaggleBatchImageProvider
    motion_provider: KaggleBatchImageToVideoProvider
    speech_provider: SpeechProvider
    video_builder: ClipAssembler
    image_width: int = 512
    image_height: int = 288
    motion_frames: int = 14
    motion_fps: int = 7
    composition: str = "16:9 widescreen"
    max_image_rounds: int = 2
    max_motion_rounds: int = 2
    learning_store: ProductionLearningStore | None = None
    audio_verifier: AudioSignalVerifier = field(default_factory=AudioSignalVerifier)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.image_width % 8 or self.image_height % 8:
            raise ValueError("image dimensions must be divisible by 8")
        if self.motion_frames <= 1:
            raise ValueError("motion_frames must be greater than 1")
        if self.motion_fps <= 0:
            raise ValueError("motion_fps must be positive")
        if self.max_image_rounds <= 0 or self.max_motion_rounds <= 0:
            raise ValueError("retry rounds must be positive")
        if not self.composition.strip():
            raise ValueError("composition must not be empty")

    def produce(
        self,
        verified_script: str,
        reference_image: bytes,
        output_dir: str | Path,
        *,
        visual_style: str = "cinematic realistic",
    ) -> ReferenceMotionVideoResult:
        assert_core_invariants()
        script = verified_script.strip()
        if not script:
            raise ValueError("verified_script must not be empty")
        if not reference_image:
            raise ValueError("reference_image must not be empty")
        if len(reference_image) > 8 * 1024 * 1024:
            raise ValueError("reference_image must not exceed 8MB")

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        reference_hash = sha256(reference_image).hexdigest()
        lessons = ()
        adapter_weight = str(getattr(self.image_provider, "ip_adapter_weight", "") or "")
        base_scale = 0.50 if "full-face" in adapter_weight else 0.75
        if self.learning_store is not None:
            try:
                lessons = self.learning_store.relevant("reference-motion-video", limit=10)
            except Exception:
                lessons = ()
        reference_scale = tuned_reference_scale(lessons, base=base_scale)

        scene_plan = self.scene_planner.plan(
            script,
            visual_style=visual_style,
            composition=self.composition,
        )
        image_requests = [
            SceneImageRequest(
                scene_id=scene.scene_id,
                prompt=(
                    scene.image_prompt
                    + ". Keep the main recurring character consistent with the supplied reference image: "
                    "same face, age, hair, body proportions, costume identity and distinctive features."
                ),
                negative_prompt=(
                    scene.negative_prompt
                    + ", different person, changed face, face drift, inconsistent character identity"
                ),
                width=self.image_width,
                height=self.image_height,
                seed=self._seed("keyframe:" + scene.scene_id),
            )
            for scene in scene_plan.scenes
        ]

        keyframes = self.image_provider.generate_with_retries(
            image_requests,
            max_rounds=self.max_image_rounds,
            reference_image=reference_image,
            reference_scale=reference_scale,
        )
        if not keyframes.verified:
            issues = tuple(issue for item in keyframes.scenes for issue in item.issues)
            learning_error = self._record_failure(
                ProductionLesson(
                    task_type="reference-motion-video",
                    success=False,
                    failure_kind=(
                        "safety_blocked"
                        if any("safety checker blocked" in issue for issue in issues)
                        else (
                            "identity_drift"
                            if any("identity similarity" in issue for issue in issues)
                            else "keyframe_model_output"
                        )
                    ),
                    lesson="Reference keyframe generation failed; retry only failed scenes or tune reference adherence.",
                    config={"reference_scale": reference_scale},
                    metrics={"failed_scene_ids": list(keyframes.failed_scene_ids)},
                    evidence=issues[:20],
                )
            )
            message = "reference keyframe verification failed for scenes: " + ", ".join(keyframes.failed_scene_ids)
            if learning_error:
                message += "; production learning persistence also failed: " + learning_error
            raise RuntimeError(message)

        keyframe_paths: list[str] = []
        keyframe_evidence: list[str] = []
        motion_requests: list[SceneImageToVideoRequest] = []
        for index, (scene, result) in enumerate(zip(scene_plan.scenes, keyframes.scenes)):
            path = target / f"scene_{index:04d}_keyframe.png"
            path.write_bytes(result.artifact.data)
            keyframe_paths.append(str(path))
            keyframe_evidence.extend(result.artifact.evidence)
            motion_requests.append(
                SceneImageToVideoRequest(
                    scene_id=scene.scene_id,
                    image=result.artifact.data,
                    width=self.image_width,
                    height=self.image_height,
                    num_frames=self.motion_frames,
                    fps=self.motion_fps,
                    seed=self._seed("motion:" + scene.scene_id),
                )
            )

        clips = self.motion_provider.generate_with_retries(
            motion_requests,
            max_rounds=self.max_motion_rounds,
        )
        if not clips.verified:
            issues = tuple(issue for item in clips.scenes for issue in item.issues)
            learning_error = self._record_failure(
                ProductionLesson(
                    task_type="reference-motion-video",
                    success=False,
                    failure_kind=(
                        "motion_identity_drift"
                        if any("identity similarity" in issue for issue in issues)
                        else (
                            "insufficient_motion"
                            if any("insufficient motion" in issue for issue in issues)
                            else "motion_model_output"
                        )
                    ),
                    lesson="I2V animation failed verification; retry only failed motion scenes with adjusted motion controls.",
                    config={"reference_scale": reference_scale},
                    metrics={"failed_scene_ids": list(clips.failed_scene_ids)},
                    evidence=issues[:20],
                )
            )
            message = "motion verification failed for scenes: " + ", ".join(clips.failed_scene_ids)
            if learning_error:
                message += "; production learning persistence also failed: " + learning_error
            raise RuntimeError(message)

        clip_paths: list[str] = []
        clip_evidence: list[str] = []
        for index, result in enumerate(clips.scenes):
            path = target / f"scene_{index:04d}.mp4"
            path.write_bytes(result.artifact.data)
            clip_paths.append(str(path))
            clip_evidence.extend(result.artifact.evidence)

        audio = self.speech_provider.synthesize(script)
        audio_path = target / "narration.wav"
        audio_path.write_bytes(audio.data)
        audio_quality = self.audio_verifier.verify(audio)
        if not audio_quality.passed:
            raise RuntimeError("audio verification failed: " + "; ".join(audio_quality.issues))

        video_path = target / "video.mp4"
        video = self.video_builder.build_from_clips(
            clip_paths,
            str(audio_path),
            str(video_path),
        )
        if not (video.has_video and video.has_audio):
            raise RuntimeError("final video does not contain both video and audio")

        keyframe_scores = self._scores(keyframe_evidence, "identity_score:")
        first_motion_scores = self._scores(clip_evidence, "first_frame_similarity:")
        last_motion_scores = self._scores(clip_evidence, "last_frame_similarity:")
        motion_deltas = self._scores(clip_evidence, "motion_delta:")

        evidence = (
            f"script_sha256:{sha256(script.encode('utf-8')).hexdigest()}",
            f"reference_sha256:{reference_hash}",
            f"reference_scale:{reference_scale:.3f}",
            f"scene_count:{len(scene_plan.scenes)}",
            f"keyframe_rounds:{keyframes.rounds}",
            f"motion_rounds:{clips.rounds}",
            "visual_mode:reference_conditioned_motion_video",
            *keyframe_evidence,
            *clip_evidence,
            *audio.evidence,
            *audio_quality.evidence,
            *video.evidence,
        )

        success_lesson = ProductionLesson(
            task_type="reference-motion-video",
            success=True,
            lesson=(
                "Verified reference-conditioned keyframes and I2V motion clips produced a narrated final video; "
                "reuse the verified reference scale and motion settings unless later runs regress."
            ),
            config={
                "reference_scale": reference_scale,
                "motion_frames": self.motion_frames,
                "motion_fps": self.motion_fps,
            },
            metrics={
                "scene_count": len(scene_plan.scenes),
                "min_keyframe_identity": min(keyframe_scores) if keyframe_scores else None,
                "min_first_motion_identity": min(first_motion_scores) if first_motion_scores else None,
                "min_last_motion_identity": min(last_motion_scores) if last_motion_scores else None,
                "mean_motion_delta": (
                    sum(motion_deltas) / len(motion_deltas) if motion_deltas else None
                ),
                "duration_seconds": video.duration_seconds,
            },
            evidence=evidence[:60],
        )
        self._record_success(success_lesson)

        return ReferenceMotionVideoResult(
            scene_plan=scene_plan,
            keyframes=keyframes,
            clips=clips,
            audio=audio,
            audio_quality=audio_quality,
            video=video,
            output_dir=str(target),
            evidence=evidence,
        )

    def _record_failure(self, lesson: ProductionLesson) -> str | None:
        if self.learning_store is None:
            return None
        try:
            self.learning_store.record(lesson)
        except Exception as exc:
            return str(exc)
        return None

    def _record_success(self, lesson: ProductionLesson) -> None:
        if self.learning_store is None:
            return
        try:
            self.learning_store.record(lesson)
        except Exception as exc:
            raise RuntimeError(f"production learning persistence failed: {exc}") from exc

    @staticmethod
    def _scores(evidence: list[str], prefix: str) -> list[float]:
        values: list[float] = []
        for item in evidence:
            if item.startswith(prefix):
                try:
                    values.append(float(item[len(prefix):]))
                except ValueError:
                    pass
        return values

    @staticmethod
    def _seed(value: str) -> int:
        return int.from_bytes(sha256(value.encode("utf-8")).digest()[:4], "big")
