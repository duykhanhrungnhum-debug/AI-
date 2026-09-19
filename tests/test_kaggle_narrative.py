import hashlib
import json

from ai_agent.core.kaggle_narrative import KaggleNarrativeProcessor
from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission


class FakeWorker:
    def __init__(self, result):
        self.result = result
        self.submitted = None

    def submit_script(self, **kwargs):
        self.submitted = kwargs
        return KaggleKernelSubmission("testuser", kwargs["slug"], 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def download_output_file(self, slug, filename):
        assert filename == "narrative_result.json"
        return json.dumps(self.result, ensure_ascii=False).encode()


def make_result(source, script, *, passed=True, issues=None):
    return {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "gpu_name": "Tesla T4",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "brief": {
            "characters": ["Lan"],
            "events": ["Lan receives a letter"],
            "must_preserve": ["the letter has no sender"],
        },
        "script": script,
        "review": {
            "passed": passed,
            "issues": issues or [],
        },
        "revision_count": 1,
    }


def test_kaggle_narrative_runs_full_cycle_in_one_model_load():
    source = "Lan received an old letter with no sender."
    script = "Lan nhận được một lá thư cũ không có tên người gửi."
    worker = FakeWorker(make_result(source, script))
    processor = KaggleNarrativeProcessor(worker=worker, poll_interval=0)

    result = processor.process(source)

    assert result.review.passed is True
    assert result.script == script
    assert result.revision_count == 1
    assert "gpu:Tesla T4" in result.evidence
    assert worker.submitted["source"].count("AutoModelForCausalLM.from_pretrained") == 1
    assert "NARRATIVE_ANALYSIS" in worker.submitted["source"]
    assert "NARRATIVE_REWRITE" in worker.submitted["source"]
    assert "NARRATIVE_REVIEW" in worker.submitted["source"]
    assert "NARRATIVE_REPAIR" in worker.submitted["source"]


def test_kaggle_narrative_host_rejects_duplicate_paragraph_even_if_model_passes():
    source = "Story."
    paragraph = "Đây là một đoạn văn đủ dài để kiểm tra trùng lặp chính xác."
    script = paragraph + "\n\n" + paragraph
    worker = FakeWorker(make_result(source, script, passed=True))
    processor = KaggleNarrativeProcessor(worker=worker, poll_interval=0)

    result = processor.process(source)

    assert result.review.passed is False
    assert "exact duplicate paragraph detected" in result.review.issues
