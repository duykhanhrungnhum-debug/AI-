import pytest

from ai_agent.core.kaggle_production_image_batch import (
    KaggleProductionImageBatchProvider,
    ObjectCountGate,
    ProductionImageItem,
)


class DummyWorker:
    def status(self, slug):  # pragma: no cover - source-only test helper
        raise AssertionError("not used")


def provider():
    engines = {
        "animal_photo_premium": {
            "engine_id": "photoreal", "model": "SG161222/RealVisXL_V5.0", "variant": "fp16",
            "steps": 28, "guidance": 4.0, "scheduler": "dpm_karras", "candidate_count": 2,
        },
        "human_photo_premium": {
            "engine_id": "photoreal", "model": "SG161222/RealVisXL_V5.0", "variant": "fp16",
            "steps": 28, "guidance": 4.0, "scheduler": "dpm_karras", "candidate_count": 2,
        },
        "general_premium": {
            "engine_id": "photoreal", "model": "SG161222/RealVisXL_V5.0", "variant": "fp16",
            "steps": 28, "guidance": 4.0, "scheduler": "dpm_karras", "candidate_count": 2,
        },
        "mascot_premium": {
            "engine_id": "mascot3d", "model": "playgroundai/playground-v2.5-1024px-aesthetic", "variant": "fp16",
            "steps": 40, "guidance": 3.0, "scheduler": "edm_dpm", "candidate_count": 3,
        },
    }
    keys = set(engines)
    return KaggleProductionImageBatchProvider(
        worker=DummyWorker(),
        profile_engines=engines,
        profile_positive_constraints={key: "premium clean output" for key in keys},
        profile_negative_constraints={key: "bad anatomy, text, logo" for key in keys},
        profile_dimensions={key: (1024, 1024) for key in keys},
        poll_interval=0,
    )


def test_worker_source_compiles_and_has_bilingual_subject_contract_and_repair():
    source = provider()._build_worker_source((
        ProductionImageItem(
            "real", "Tạo ảnh đúng một con trâu nước Việt Nam thật", 1,
            hard_gates=(ObjectCountGate("horn", 2, "individual water buffalo horn"),),
        ),
        ProductionImageItem(
            "cute3d", "Tạo ảnh đúng một con trâu nước 3D cute", 2,
            hard_gates=(ObjectCountGate("horn", 2, "individual water buffalo horn"),),
        ),
    ))
    compile(source, "<production-image-worker>", "exec")

    assert "AI_AGENT_PROGRESS" in source
    assert 'progress("planner_loading"' in source
    assert 'progress("planner_repair"' in source
    assert '"planner_item_complete"' in source
    assert 'progress("engine_loading"' in source
    assert 'progress("candidate_generating"' in source
    assert 'progress("detector_loading"' in source
    assert 'progress("worker_complete"' in source
    assert "Qwen/Qwen3-1.7B" in source
    assert "Qwen/Qwen3-0.6B" not in source
    assert "Never generalize a specific species" in source
    assert "subject_matches_source" in source
    assert "english_subject_matches_prompt" in source
    assert "SUBJECT_ENGLISH" in source
    assert "exact English common name/translation" in source
    assert "PROMPT must be a complete ENGLISH generation prompt" in source
    assert "Do not transliterate or invent a near-sounding English word" in source
    assert "ORIGINAL" not in source  # original-request QA is host-side


def test_detector_count_is_trusted_contract_not_planner_output():
    source = provider()._build_worker_source((
        ProductionImageItem(
            "buffalo", "Tạo trâu nước", 3,
            hard_gates=(ObjectCountGate("horn", 2, "individual water buffalo horn"),),
        ),
    ))
    assert "AutoModelForZeroShotObjectDetection" in source
    assert "grounding-dino-tiny" in source
    assert '"expected_count": gate.expected_count' not in source
    assert 'item["hard_gates"]' in source
    assert "planner never invents them" in source
    assert "detected_count" in source


def test_object_count_gate_rejects_invalid_contract():
    with pytest.raises(ValueError):
        ObjectCountGate("", 2)
    with pytest.raises(ValueError):
        ObjectCountGate("horn", -1)
    with pytest.raises(ValueError):
        ObjectCountGate("horn", 2, "")
