import hashlib
import json

import pytest

from ai_agent.core.kaggle_model import KaggleModelProvider
from ai_agent.core.kaggle_worker import KaggleKernelStatus, KaggleKernelSubmission


class FakeWorker:
    def __init__(self, responses):
        self.responses = responses
        self.submissions = []

    def submit_script(self, **kwargs):
        self.submissions.append(kwargs)
        return KaggleKernelSubmission("testuser", kwargs["slug"], 1, 1, None)

    def status(self, slug):
        return KaggleKernelStatus("COMPLETE")

    def download_output_file(self, slug, filename):
        assert filename == "responses.json"
        prompts = []
        source = self.submissions[-1]["source"]
        for response in self.responses:
            prompts.append(response["prompt"])
        items = [{
            "text": item["text"],
            "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
            "response_sha256": hashlib.sha256(item["text"].encode()).hexdigest(),
        } for item in self.responses]
        return json.dumps({
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "gpu_name": "Tesla T4",
            "responses": items,
        }).encode()


def test_kaggle_model_provider_runs_open_model_and_verifies_hashes():
    worker = FakeWorker([{"prompt": "hello", "text": "xin chao"}])
    provider = KaggleModelProvider(worker=worker, poll_interval=0)

    response = provider.generate("hello")

    assert response.text == "xin chao"
    assert response.provider == "kaggle-gpu-open-model"
    assert response.model == "Qwen/Qwen2.5-3B-Instruct"
    assert worker.submissions[0]["enable_internet"] is True
    assert "AutoModelForCausalLM.from_pretrained" in worker.submissions[0]["source"]


def test_kaggle_model_provider_batches_prompts_in_one_model_load():
    worker = FakeWorker([
        {"prompt": "one", "text": "mot"},
        {"prompt": "two", "text": "hai"},
    ])
    provider = KaggleModelProvider(worker=worker, poll_interval=0)

    result = provider.generate_many(["one", "two"])

    assert [item.text for item in result.responses] == ["mot", "hai"]
    assert len(worker.submissions) == 1
    assert worker.submissions[0]["source"].count("AutoModelForCausalLM.from_pretrained") == 1
    assert "response_count:2" in result.evidence
    assert "gpu:Tesla T4" in result.evidence


def test_kaggle_model_provider_rejects_empty_prompt():
    provider = KaggleModelProvider(worker=FakeWorker([]), poll_interval=0)

    with pytest.raises(ValueError, match="non-empty"):
        provider.generate("   ")
