from ai_agent.executors.image_plan import plan_explicit_images, requested_image_count
from ai_agent.workers.image_manager_v6 import WarmImageWorkerManagerV6


def test_requested_count_handles_vietnamese_and_english():
    assert requested_image_count("AIKA tạo hai ảnh riêng") == 2
    assert requested_image_count("Create 3 images separately") == 3
    assert requested_image_count("Create two pictures") == 2
    assert requested_image_count("một ảnh chân dung") == 1


def test_explicit_vietnamese_items_split_deterministically():
    command = (
        "AIKA tạo hai ảnh riêng biệt để kiểm tra: "
        "ảnh thứ nhất là một bông sen hồng chân thực trên mặt nước; "
        "ảnh thứ hai là một bông hướng dương chân thực ngoài đồng."
    )
    items = plan_explicit_images(command)
    assert items is not None
    assert len(items) == 2
    assert items[0].index == 1
    assert "sen hồng" in items[0].command
    assert items[1].index == 2
    assert "hướng dương" in items[1].command


def test_explicit_english_items_split_deterministically():
    items = plan_explicit_images(
        "Create two images: first image is a red apple on white; second image is an orange on black."
    )
    assert items is not None
    assert len(items) == 2
    assert "red apple" in items[0].command
    assert "orange" in items[1].command


def test_multiple_subjects_without_file_boundaries_are_not_split():
    assert plan_explicit_images(
        "Tạo một ảnh có cô gái, con mèo và con chó đứng cùng nhau"
    ) is None
    assert plan_explicit_images(
        "Tạo ảnh hai cô gái châu Á và châu Âu đứng cùng một khung hình"
    ) is None


def test_inconsistent_explicit_count_fails_closed():
    assert plan_explicit_images(
        "Tạo ba ảnh riêng: ảnh thứ nhất là táo; ảnh thứ hai là cam"
    ) is None


def test_v6_worker_owns_file_count_structurally():
    source = WarmImageWorkerManagerV6()._worker_source(
        base_url="https://example.invalid",
        worker_token="token",
        session_id="session-v6",
    )
    assert "def split_explicit_items(command):" in source
    assert "def requested_count(command):" in source
    assert "render_one_recaption(recaptioner, item_command)" in source
    assert 'stage = f"busy:recaption:{item_index}/{item_total}"' in source
    assert "command explicitly requested" in source
    assert 'User-Agent": "AIKA-Warm-Image/6.3"' in source
