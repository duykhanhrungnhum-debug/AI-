import json

from ai_agent.core.model import ModelResponse
from ai_agent.core.narrative_pipeline import NarrativeProcessor


class QueueModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return ModelResponse(self.outputs.pop(0), "fake", "queue")


def analysis():
    return json.dumps({
        "characters": ["Lan", "Minh"],
        "events": ["Lan finds the letter", "Minh explains the secret"],
        "must_preserve": ["The letter belongs to Minh"],
    })


def test_narrative_pipeline_rewrites_and_passes_review():
    model = QueueModel([
        analysis(),
        "Lan tìm thấy lá thư. Minh giải thích bí mật.",
        json.dumps({"passed": True, "issues": []}),
    ])

    result = NarrativeProcessor(model).process("Lan finds a letter. Minh explains the secret.")

    assert result.review.passed is True
    assert result.revision_count == 0
    assert "Lan" in result.script
    assert model.prompts[0].startswith("NARRATIVE_ANALYSIS")
    assert model.prompts[1].startswith("NARRATIVE_REWRITE")
    assert model.prompts[2].startswith("NARRATIVE_REVIEW")


def test_narrative_pipeline_repairs_failed_review():
    model = QueueModel([
        analysis(),
        "Lan tìm thấy thư. Minh biến mất.",
        json.dumps({"passed": False, "issues": ["Minh's explanation is missing"]}),
        "Lan tìm thấy lá thư. Minh giải thích bí mật.",
        json.dumps({"passed": True, "issues": []}),
    ])

    result = NarrativeProcessor(model, max_revisions=2).process("Lan finds a letter. Minh explains the secret.")

    assert result.review.passed is True
    assert result.revision_count == 1
    assert any(p.startswith("NARRATIVE_REPAIR") for p in model.prompts)


def test_deterministic_duplicate_check_can_override_model_pass():
    duplicate = "Đây là một đoạn văn đủ dài để kiểm tra lặp chính xác."
    model = QueueModel([
        analysis(),
        duplicate + "\n\n" + duplicate,
        json.dumps({"passed": True, "issues": []}),
        "Đây là bản đã sửa không còn đoạn văn bị lặp chính xác.",
        json.dumps({"passed": True, "issues": []}),
    ])

    result = NarrativeProcessor(model, max_revisions=1).process("Lan finds a letter. Minh explains the secret.")

    assert result.review.passed is True
    assert result.revision_count == 1


def test_narrative_pipeline_stops_after_revision_limit():
    model = QueueModel([
        analysis(),
        "Sai bản đầu.",
        json.dumps({"passed": False, "issues": ["wrong meaning"]}),
        "Sai bản sửa.",
        json.dumps({"passed": False, "issues": ["still wrong"]}),
    ])

    result = NarrativeProcessor(model, max_revisions=1).process("Lan finds a letter. Minh explains the secret.")

    assert result.review.passed is False
    assert result.revision_count == 1
    assert result.review.issues == ("still wrong",)

def test_narrative_pipeline_detects_repair_loop():
    model = QueueModel([
        analysis(),
        "Lan tìm thấy thư. Minh biến mất.",
        json.dumps({"passed": False, "issues": ["Minh's explanation is missing"]}),
        "Lan tìm thấy thư. Minh biến mất.",
    ])

    result = NarrativeProcessor(model, max_revisions=3).process("Lan finds a letter. Minh explains the secret.")

    assert result.review.passed is False
    assert result.revision_count == 1
    assert "repair loop detected: repeated script" in result.review.issues


def test_narrative_pipeline_can_use_separate_reviewer():
    writer = QueueModel([
        analysis(),
        "Lan tìm thấy lá thư. Minh giải thích bí mật.",
    ])
    reviewer = QueueModel([
        json.dumps({"passed": True, "issues": []}),
    ])

    result = NarrativeProcessor(writer, review_provider=reviewer).process(
        "Lan finds a letter. Minh explains the secret."
    )

    assert result.review.passed is True
    assert len(writer.prompts) == 2
    assert len(reviewer.prompts) == 1
    assert reviewer.prompts[0].startswith("NARRATIVE_REVIEW")
