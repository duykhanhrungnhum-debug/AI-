from pathlib import Path


def test_simple_image_pipeline_contract():
    source = Path("scripts/generate_simple_image_batch.py").read_text(encoding="utf-8")
    compile(source, "<generate_simple_image_batch.py>", "exec")

    assert "QUALITY_TARGET = 9.0" in source
    assert "Qwen/Qwen3-VL-8B-Instruct" in source
    assert '"candidate_count": 1' in source
    assert '"candidate_count": 2' in source
    assert "hard_gates=()" in source
    assert "GroundingDINO" not in source
    assert "grounding-dino" not in source
    assert '"detector_enabled": False' in source
    assert "ORIGINAL USER REQUEST (authoritative)" in source
    assert "one targeted retry" in source
    assert "score_floor(entry[1]) >= QUALITY_TARGET" in source


def test_buffalo_exam_uses_simple_pipeline_without_detector_contracts():
    workflow = Path(".github/workflows/buffalo-image-exam.yml").read_text(encoding="utf-8")

    assert "python scripts/generate_simple_image_batch.py" in workflow
    assert "hard_gates" not in workflow
    assert "GroundingDINO" not in workflow
