from ai_agent.chat_session import ChatJob, ChatSessionBroker, classify_chat_intent


def test_aika_routes_vietnamese_image_commands():
    assert classify_chat_intent("AIKA tạo ảnh hai cô gái châu Á và châu Âu") == "image"
    assert classify_chat_intent("Vẽ hình một con trâu phong cách 3D") == "image"
    assert classify_chat_intent("Create a portrait of a woman") == "image"


def test_aika_keeps_normal_conversation_in_chat():
    assert classify_chat_intent("AIKA tên gì?") == "chat"
    assert classify_chat_intent("Giải thích quy trình tạo ảnh hiện tại") == "chat"
    assert classify_chat_intent("Xin chào") == "chat"


def test_image_job_status_exposes_authenticated_fetch_path():
    broker = ChatSessionBroker()
    job = ChatJob(
        job_id="abc123",
        prompt="tạo ảnh",
        message="tạo ảnh",
        kind="image",
        status="done",
        text="AIKA đã tạo ảnh xong.",
        provider="kaggle-image-v2",
        model="black-forest-labs/FLUX.2-klein-4B",
        image_data=b"\x89PNG\r\n\x1a\nmock",
    )
    broker._jobs[job.job_id] = job
    status = broker.get_job(job.job_id)
    assert status["kind"] == "image"
    assert status["has_image"] is True
    assert status["image_url"] == "/v1/chat/image?job_id=abc123"
    data, mime = broker.get_image(job.job_id)
    assert data.startswith(b"\x89PNG")
    assert mime == "image/png"
