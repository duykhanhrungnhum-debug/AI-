import time

from ai_agent.media_broker import MediaToolBroker


PNG = b"\x89PNG\r\n\x1a\n" + b"A" * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"V" * 128


class FakeImageResult:
    data = PNG
    prompt = "original image prompt"
    elapsed_seconds = 1.25
    model = "fake-image-model"


class FakeImageTool:
    def generate(self, command, **kwargs):
        assert command == "tạo ảnh con mèo"
        return (FakeImageResult(),)


class FakeVideoArtifact:
    data = MP4
    mime_type = "video/mp4"
    provider = "fake-video-provider"
    model = "fake-video-model"


class FakeVideoTool:
    def generate(self, command, **kwargs):
        assert command == "tạo video con mèo chạy"
        return FakeVideoArtifact()


def _wait_done(broker, job_id):
    deadline = time.time() + 2
    while time.time() < deadline:
        state = broker.get_job(job_id)
        if state["status"] in {"done", "error"}:
            return state
        time.sleep(0.01)
    raise AssertionError("media job did not finish")


def test_image_tool_broker_returns_png_without_backend_knowledge():
    broker = MediaToolBroker()
    broker._image = FakeImageTool()
    job = broker.create_job("image", "tạo ảnh con mèo")
    state = _wait_done(broker, job.job_id)
    assert state["status"] == "done"
    assert state["kind"] == "image"
    assert state["image_count"] == 1
    assert state["provider"] == "image-tool"
    assert state["has_video"] is False
    data, mime = broker.get_image(job.job_id)
    assert data == PNG
    assert mime == "image/png"


def test_video_tool_broker_returns_mp4_without_backend_knowledge():
    broker = MediaToolBroker()
    broker._video = FakeVideoTool()
    job = broker.create_job("video", "tạo video con mèo chạy")
    state = _wait_done(broker, job.job_id)
    assert state["status"] == "done"
    assert state["kind"] == "video"
    assert state["provider"] == "video-tool"
    assert state["has_image"] is False
    assert state["has_video"] is True
    assert state["video_url"].endswith(job.job_id)
    data, mime = broker.get_video(job.job_id)
    assert data == MP4
    assert mime == "video/mp4"
