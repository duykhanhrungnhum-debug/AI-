"""AIKA Video Tool: keyframe first, then image-to-video.

AIKA sees one stable generate() call. If no reference image is supplied, this
tool asks the Image Tool for exactly one opening keyframe, then animates that
keyframe with the private I2V backend. Text-to-video is not a production path.

The tool does not expose an MP4 as complete until the backend evidence passes
AIKA's bounded visual-quality gate for meaningful duration, subject continuity,
and real motion.
"""
from __future__ import annotations

from hashlib import sha256
import os
from uuid import uuid4

from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.simple_i2v_tool import SimpleI2VArtifact, SimpleKaggleI2VTool
from ai_agent.executors.image import ImageExecutor


PRODUCTION_VIDEO_KERNEL = "ai-agent-video-i2v"
MIN_VIDEO_DURATION_SECONDS = 4.0
MIN_VIDEO_FRAMES = 33
MIN_FIRST_FRAME_SIMILARITY = 0.90
MIN_LAST_FRAME_SIMILARITY = 0.85
MIN_MOTION_DELTA = 0.25
MAX_MOTION_DELTA = 35.0

GENERIC_VIDEO_CONSTRAINTS = (
    "Keep every named subject visually distinct and anatomically coherent for the whole shot. "
    "Preserve exact subject identity and count. Do not merge, hybridize, morph, substitute, duplicate, "
    "or omit subjects. Keep every explicitly requested object and action visibly present in the shot. "
    "Requested motion must be real subject or environmental motion, not camera drift, zoom, blur, "
    "crossfade, dissolve, exposure change, or lighting-only change. Use one continuous coherent shot "
    "with natural motion and stable composition. No text, subtitles, logos, or watermarks unless requested."
)
KEYFRAME_CONSTRAINTS = (
    "Create one still opening keyframe for the requested video. Show every explicitly requested subject "
    "and important object clearly in the same frame, with stable anatomy and identity. Depict the moment "
    "immediately before the requested motion begins. No motion blur, duplicate subjects, text, logos, or watermarks."
)


def _video_prompt(command: str) -> str:
    command = command.strip()
    if not command:
        raise ValueError("video prompt is required")
    return f"{command}. {GENERIC_VIDEO_CONSTRAINTS}"


def _keyframe_prompt(command: str) -> str:
    command = command.strip()
    if not command:
        raise ValueError("video prompt is required")
    return f"{KEYFRAME_CONSTRAINTS} Requested video: {command}"


def _evidence_number(artifact: SimpleI2VArtifact, name: str) -> float:
    prefix = name + ":"
    for item in artifact.evidence:
        if item.startswith(prefix):
            raw = item[len(prefix):].strip()
            try:
                return float(raw)
            except ValueError as exc:
                raise RuntimeError(f"invalid video QA evidence {name}: {raw}") from exc
    raise RuntimeError(f"missing video QA evidence: {name}")


def assert_video_quality(artifact: SimpleI2VArtifact) -> None:
    """Hard gate applied by AIKA before a generated MP4 can be published.

    These checks deliberately use backend-produced frame evidence rather than
    provider/model names, so the backend remains replaceable behind the stable
    Video Tool interface.
    """
    if artifact.mime_type != "video/mp4" or not artifact.data:
        raise RuntimeError("video QA failed: missing MP4 artifact")
    if artifact.duration_seconds < MIN_VIDEO_DURATION_SECONDS:
        raise RuntimeError(
            f"video QA failed: duration {artifact.duration_seconds:.3f}s < {MIN_VIDEO_DURATION_SECONDS:.1f}s"
        )

    frames = int(_evidence_number(artifact, "frames"))
    first_similarity = _evidence_number(artifact, "first_frame_similarity")
    last_similarity = _evidence_number(artifact, "last_frame_similarity")
    motion_delta = _evidence_number(artifact, "motion_delta")

    if frames < MIN_VIDEO_FRAMES:
        raise RuntimeError(f"video QA failed: frames {frames} < {MIN_VIDEO_FRAMES}")
    if first_similarity < MIN_FIRST_FRAME_SIMILARITY:
        raise RuntimeError(
            f"video QA failed: opening identity similarity {first_similarity:.4f} < {MIN_FIRST_FRAME_SIMILARITY:.2f}"
        )
    if last_similarity < MIN_LAST_FRAME_SIMILARITY:
        raise RuntimeError(
            f"video QA failed: ending identity similarity {last_similarity:.4f} < {MIN_LAST_FRAME_SIMILARITY:.2f}"
        )
    if motion_delta < MIN_MOTION_DELTA:
        raise RuntimeError(f"video QA failed: motion {motion_delta:.4f} is effectively static")
    if motion_delta > MAX_MOTION_DELTA:
        raise RuntimeError(
            f"video QA failed: motion {motion_delta:.4f} exceeds continuity bound {MAX_MOTION_DELTA:.1f}"
        )


class VideoExecutor:
    """Stable Video Tool boundary: keyframe -> one bounded I2V GPU call -> QA."""

    def __init__(
        self,
        *,
        kernel_slug: str = PRODUCTION_VIDEO_KERNEL,
        inference_steps: int = 30,
        poll_interval: float = 5.0,
        max_poll_attempts: int = 180,
    ) -> None:
        self.kernel_slug = kernel_slug
        self.inference_steps = inference_steps
        self.poll_interval = poll_interval
        self.max_wait_seconds = max(1.0, poll_interval * max_poll_attempts)

    @staticmethod
    def seed_for(command: str) -> int:
        return int.from_bytes(sha256(command.encode("utf-8")).digest()[:4], "big")

    def _provider(self, *, job_slug: str) -> SimpleKaggleI2VTool:
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        if not token:
            raise RuntimeError("KAGGLE_API_TOKEN is required")
        if not username:
            raise RuntimeError("KAGGLE_USERNAME is required")
        return SimpleKaggleI2VTool(
            worker=KaggleGpuWorker(
                api_token=token,
                username=username,
                timeout=45,
                submission_retry_attempts=2,
                submission_retry_delay_seconds=10,
            ),
            kernel_slug=job_slug,
            poll_interval=self.poll_interval,
            max_wait_seconds=min(self.max_wait_seconds, 900.0),
            inference_steps=self.inference_steps,
            guidance_scale=3.0,
        )

    def _create_keyframe(self, command: str) -> bytes:
        # Use a unique image job so concurrent Video Tool calls cannot share or
        # confuse artifacts. The Image Tool remains independently replaceable.
        image_tool = ImageExecutor(kernel_slug=f"ai-agent-video-keyframe-{uuid4().hex[:10]}")
        results = image_tool.generate(
            _keyframe_prompt(command),
            width=768,
            height=432,
            max_images=1,
        )
        if len(results) != 1 or not results[0].data:
            raise RuntimeError("Image Tool did not return exactly one video keyframe")
        return results[0].data

    def generate(
        self,
        command: str,
        *,
        reference_image: bytes | None = None,
        width: int = 512,
        height: int = 288,
        num_frames: int = 33,
        fps: int = 8,
    ) -> SimpleI2VArtifact:
        """Generate and quality-gate one MP4 from an explicit or generated keyframe."""
        original = command.strip()
        if not original:
            raise ValueError("video prompt is required")
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("video dimensions must be positive and divisible by 32")
        if num_frames <= 1 or (num_frames - 1) % 8:
            raise ValueError("video num_frames must follow the 8*k+1 temporal grid")
        if fps <= 0:
            raise ValueError("fps must be positive")

        keyframe = reference_image or self._create_keyframe(original)
        if not keyframe:
            raise RuntimeError("Video Tool has no reference keyframe")
        job_slug = f"{self.kernel_slug}-{uuid4().hex[:10]}"
        artifact = self._provider(job_slug=job_slug).generate(
            _video_prompt(original),
            keyframe,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            seed=self.seed_for(original),
        )
        assert_video_quality(artifact)
        return artifact
