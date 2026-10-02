from __future__ import annotations

import inspect

from ai_agent.executors.video import GENERIC_VIDEO_CONSTRAINTS, VideoExecutor


def test_default_video_duration_is_meaningful() -> None:
    signature = inspect.signature(VideoExecutor.execute)
    num_frames = int(signature.parameters["num_frames"].default)
    fps = int(signature.parameters["fps"].default)

    assert (num_frames - 1) % 4 == 0
    assert num_frames / fps >= 4.0


def test_generic_video_constraints_reject_fake_motion_substitutes() -> None:
    text = GENERIC_VIDEO_CONSTRAINTS.casefold()
    assert "explicitly requested object and action" in text
    assert "camera drift" in text
    assert "blur" in text
    assert "crossfade" in text
    assert "dissolve" in text
    assert "lighting-only change" in text
