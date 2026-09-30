import base64

from ai_agent.chat_session import ChatJob, ChatSessionBroker, classify_chat_intent


PNG1 = b"\x89PNG\r\n\x1a\n" + b"A" * 32
PNG2 = b"\x89PNG\r\n\x1a\n" + b"B" * 32


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
        image_data=PNG1,
    )
    broker._jobs[job.job_id] = job
    status = broker.get_job(job.job_id)
    assert status["kind"] == "image"
    assert status["has_image"] is True
    assert status["image_count"] == 1
    assert status["image_url"] == "/v1/chat/image?job_id=abc123&index=0"
    data, mime = broker.get_image(job.job_id)
    assert data.startswith(b"\x89PNG")
    assert mime == "image/png"


def test_multi_image_job_exposes_separate_urls_and_files():
    broker = ChatSessionBroker()
    job = ChatJob(
        job_id="multi123",
        prompt="tạo hai ảnh riêng",
        message="tạo hai ảnh riêng",
        kind="image",
        status="done",
        text="AIKA đã tạo xong 2 ảnh riêng.",
        image_datas=[PNG1, PNG2],
        image_mimes=["image/png", "image/png"],
        generation_prompts=["first", "second"],
        expected_images=2,
    )
    broker._jobs[job.job_id] = job
    status = broker.get_job(job.job_id)
    assert status["image_count"] == 2
    assert status["image_urls"] == [
        "/v1/chat/image?job_id=multi123&index=0",
        "/v1/chat/image?job_id=multi123&index=1",
    ]
    assert broker.get_image("multi123", index=0)[0] == PNG1
    assert broker.get_image("multi123", index=1)[0] == PNG2


def test_warm_results_complete_only_after_all_images_arrive():
    broker = ChatSessionBroker()
    job = ChatJob(
        job_id="warm123",
        prompt="tạo hai ảnh riêng",
        message="tạo hai ảnh riêng",
        kind="image",
        status="processing",
    )
    broker._jobs[job.job_id] = job

    broker.finish_image_job({
        "job_id": "warm123",
        "provider": "kaggle-image-warm",
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "image_index": 0,
        "image_total": 2,
        "image_b64": base64.b64encode(PNG1).decode("ascii"),
        "generation_prompt": "first",
        "elapsed_seconds": 2.0,
    })
    assert broker.get_job("warm123")["status"] == "processing"

    broker.finish_image_job({
        "job_id": "warm123",
        "provider": "kaggle-image-warm",
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "image_index": 1,
        "image_total": 2,
        "image_b64": base64.b64encode(PNG2).decode("ascii"),
        "generation_prompt": "second",
        "elapsed_seconds": 3.0,
    })
    status = broker.get_job("warm123")
    assert status["status"] == "done"
    assert status["image_count"] == 2
    assert status["elapsed_seconds"] == 5.0
