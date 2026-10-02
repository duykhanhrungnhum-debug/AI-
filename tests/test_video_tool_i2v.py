from __future__ import annotations

from ai_agent.core.simple_i2v_tool import SimpleI2VArtifact
from ai_agent.executors.video import VideoExecutor


KEYFRAME = b"keyframe-bytes"
VIDEO = b"\x00\x00\x00\x18ftypisom" + b"V" * 128


class FakeI2V:
    def __init__(self):
        self.image = b""
        self.prompt = ""

    def generate(self, prompt, image, **kwargs):
        self.prompt = prompt
        self.image = image
        assert kwargs["width"] == 512
        assert kwargs["height"] == 288
        assert kwargs["num_frames"] == 33
        assert kwargs["fps"] == 8
        return SimpleI2VArtifact(
            data=VIDEO,
            mime_type="video/mp4",
            provider="fake-i2v",
            model="Lightricks/LTX-Video",
            duration_seconds=4.125,
            width=512,
            height=288,
            fps=8.0,
            elapsed_seconds=1.0,
            evidence=("technical:test",),
        )


def test_video_tool_creates_keyframe_before_i2v_when_reference_missing():
    executor = VideoExecutor()
    fake = FakeI2V()
    calls = []
    executor._create_keyframe = lambda command: calls.append(command) or KEYFRAME
    executor._provider = lambda job_slug: fake

    artifact = executor.generate("A frog waves while a crab watches")

    assert calls == ["A frog waves while a crab watches"]
    assert fake.image == KEYFRAME
    assert "A frog waves while a crab watches" in fake.prompt
    assert artifact.data == VIDEO


def test_video_tool_uses_explicit_reference_without_generating_keyframe():
    executor = VideoExecutor()
    fake = FakeI2V()
    executor._create_keyframe = lambda command: (_ for _ in ()).throw(AssertionError("must not create keyframe"))
    executor._provider = lambda job_slug: fake

    artifact = executor.generate("gentle motion", reference_image=KEYFRAME)

    assert fake.image == KEYFRAME
    assert artifact.duration_seconds == 4.125
