from ai_agent.core.production_learning import (
    HttpProductionLearningStore,
    ProductionLesson,
    tuned_reference_scale,
)


def test_token_provider_refreshes_for_each_request_boundary():
    tokens = iter(("token-1", "token-2"))
    store = HttpProductionLearningStore(
        base_url="https://example.test/functions/v1/story-processor-api",
        token_provider=lambda: next(tokens),
    )

    assert store._token() == "token-1"
    assert store._token() == "token-2"


def test_static_token_remains_supported():
    store = HttpProductionLearningStore(
        base_url="https://example.test/functions/v1/story-processor-api",
        bearer_token="static-token",
    )

    assert store._token() == "static-token"


def test_reference_scale_ignores_lessons_from_other_image_stacks():
    lessons = (
        ProductionLesson(
            task_type="reference-motion-video",
            success=False,
            failure_kind="identity_drift",
            lesson="old sd15 lesson",
            config={
                "reference_scale": 0.85,
                "image_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "ip_adapter_weight": "ip-adapter-full-face_sd15.bin",
            },
        ),
        ProductionLesson(
            task_type="reference-motion-video",
            success=True,
            lesson="realvis verified",
            config={
                "reference_scale": 0.68,
                "image_model": "SG161222/RealVisXL_V4.0",
                "ip_adapter_weight": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
            },
        ),
    )

    scale = tuned_reference_scale(
        lessons,
        base=0.65,
        model="SG161222/RealVisXL_V4.0",
        adapter_weight="ip-adapter-plus-face_sdxl_vit-h.safetensors",
    )

    assert scale == 0.68


def test_reference_scale_ignores_legacy_unscoped_lessons_when_stack_is_known():
    lessons = (
        ProductionLesson(
            task_type="reference-motion-video",
            success=False,
            failure_kind="identity_drift",
            lesson="legacy lesson without model provenance",
            config={"reference_scale": 0.90},
        ),
    )

    scale = tuned_reference_scale(
        lessons,
        base=0.65,
        model="SG161222/RealVisXL_V4.0",
        adapter_weight="ip-adapter-plus-face_sdxl_vit-h.safetensors",
    )

    assert scale == 0.65
