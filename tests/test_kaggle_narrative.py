import hashlib
import json

from ai_agent.core.editorial_lessons import DEFAULT_EDITORIAL_LESSONS
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


def make_result(source, script, *, passed=True, issues=None, strategies=None):
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
        "review": {"passed": passed, "issues": issues or []},
        "revision_count": 1,
        "strategy_history": strategies or ["targeted_repair"],
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
    assert f"editorial_lessons_applied:{len(DEFAULT_EDITORIAL_LESSONS)}" in result.evidence
    generated = worker.submitted["source"]
    assert generated.count("AutoModelForCausalLM.from_pretrained") == 1
    assert "BitsAndBytesConfig" in generated
    assert "import bitsandbytes" in generated
    assert "bitsandbytes>=0.46.1,<1" in generated
    assert "load_in_4bit=True" in generated
    assert 'bnb_4bit_quant_type="nf4"' in generated
    assert "PYTORCH_CUDA_ALLOC_CONF" in generated
    assert "NARRATIVE_FACT_REVIEW" in generated
    assert "NARRATIVE_FACT_ADJUDICATION" in generated
    assert "SentenceTransformer" in generated
    assert 'device="cpu"' in generated
    assert "semantic_adjudicate_fact" in generated
    assert "semantic_threshold" in generated
    assert "exact contiguous substring copied from SCRIPT" in generated
    assert "NARRATIVE_TARGETED_REPAIR" in generated
    assert "NARRATIVE_REBUILD_FROM_FACTS" in generated
    assert "FACT_CHECKLIST" in generated


def test_kaggle_narrative_host_rejects_duplicate_paragraph_even_if_model_passes():
    source = "Story."
    paragraph = "Đây là một đoạn văn đủ dài để kiểm tra trùng lặp chính xác."
    script = paragraph + "\n\n" + paragraph
    worker = FakeWorker(make_result(source, script, passed=True))
    processor = KaggleNarrativeProcessor(worker=worker, poll_interval=0)

    result = processor.process(source)

    assert result.review.passed is False
    assert "exact duplicate paragraph detected" in result.review.issues


def test_kaggle_narrative_generated_worker_source_compiles():
    processor = KaggleNarrativeProcessor(worker=FakeWorker({}), poll_interval=0)
    source = processor._build_worker_source(
        "Lan found a letter. Do not open the door.",
        "Vietnamese",
    )

    compile(source, "<generated-kaggle-worker>", "exec")
    assert source.startswith("from __future__ import annotations")
    assert "repair loop detected after strategy switch" in source
    assert "unknown fact ID" in source
    assert "for fact_id in sorted(omitted)" in source
    assert '"preserved": False' in source
    assert 'normalized_checks.append({' in source
    review_section = source.split("NARRATIVE_FACT_REVIEW", 1)[1].split("def review_script", 1)[0] if False else source
    assert "FACT_CHECKLIST, which was extracted from SOURCE" in source
    assert 'f"SOURCE:\\n{source}\\nSCRIPT:\\n{current}"' not in source
    assert "decoder.raw_decode" in source
    assert "def split_source" in source
    assert "Analyze SOURCE_CHUNK" in source
    assert "story-critical facts explicitly supported" in source
    assert "Write every extracted fact in" in source
    assert "exclude figures mentioned only in poems, cosmology, background history" in source
    assert 'plot_fact_text = " ".join(' in source
    assert "if character.casefold() in plot_fact_text" in source
    assert "torch.argsort(scores, descending=True)[:4]" in source
    assert 'candidate_evidence = "\\n".join(' in source
    assert "if not preserved and semantic_candidates:" in source
    assert 'if CONFIG["review_only"]:' in source
    assert '"repetition_penalty": 1.08' in source
    assert 'def adjudicate_fact(fact_id, current, candidate_evidence="")' in source


def test_default_editorial_lessons_capture_known_failure_modes():
    joined = " ".join(DEFAULT_EDITORIAL_LESSONS).casefold()

    assert "character names" in joined
    assert "source checklist" in joined
    assert "unchanged" in joined
    assert "rebuild" in joined


def test_kaggle_narrative_semantic_verifier_defaults_are_bounded():
    processor = KaggleNarrativeProcessor(worker=FakeWorker({}), poll_interval=0)

    assert processor.semantic_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert 0 < processor.semantic_threshold <= 1
