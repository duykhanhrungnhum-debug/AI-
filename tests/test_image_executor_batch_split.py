from ai_agent.executors.image import _split_explicit_image_items


def test_split_vietnamese_explicit_image_ordinals():
    command = (
        "Tạo ba ảnh riêng: "
        "ảnh thứ nhất là chân dung một phụ nữ Á Đông trưởng thành 25 tuổi; "
        "ảnh thứ hai là một con trâu nước Việt Nam chân thực ngoài đồng lúa; "
        "ảnh thứ ba là một con trâu nước Việt Nam hoạt hình 3D cute."
    )
    items = _split_explicit_image_items(command, max_images=6)
    assert items == (
        "chân dung một phụ nữ Á Đông trưởng thành 25 tuổi",
        "một con trâu nước Việt Nam chân thực ngoài đồng lúa",
        "một con trâu nước Việt Nam hoạt hình 3D cute",
    )


def test_split_numbered_images_is_generic_not_species_specific():
    command = "Ảnh 1 là chiếc xe đỏ; ảnh 2 là núi tuyết; ảnh 3 là ly cà phê."
    assert _split_explicit_image_items(command, max_images=6) == (
        "chiếc xe đỏ",
        "núi tuyết",
        "ly cà phê",
    )


def test_single_image_does_not_enter_batch_splitter():
    assert _split_explicit_image_items(
        "Tạo ảnh một con trâu đứng ngoài đồng",
        max_images=6,
    ) is None


def test_non_sequential_markers_fail_closed_to_natural_path():
    assert _split_explicit_image_items(
        "Ảnh thứ nhất là mèo; ảnh thứ ba là chó",
        max_images=6,
    ) is None
