import hashlib
import json
import re

from ai_agent.core.kaggle_worker import KaggleKernelSubmission
from ai_agent.core.media_v2 import ImageRequestV2
from ai_agent.core.media_v2_stable import StableKaggleImageV2Provider
from ai_agent.executors.image import ImageExecutor


PNG = b"\x89PNG\r\n\x1a\n" + b"S" * 64


class StableFakeWorker:
    def __init__(self):
        self.source = ""
        self.report_reads = 0
        self.run_token = ""

    def submit_script(self, **kwargs):
        self.source = kwargs["source"]
        match = re.search(r"RUN_TOKEN = '([0-9a-f]+)'", self.source)
        assert match
        self.run_token = match.group(1)
        return KaggleKernelSubmission("user", kwargs["slug"], 2, 2, None)

    def status(self, slug):
        raise AssertionError("stable provider must not call kernels.get/status")

    def logs(self, slug):
        return "AIKA_IMAGE_RUN " + self.run_token

    def download_output_file(self, slug, filename):
        if filename == "image_v2_report.json":
            self.report_reads += 1
            token = "stale-old-run" if self.report_reads == 1 else self.run_token
            return json.dumps({
                "run_token": token,
                "gpu_name": "Tesla T4",
                "items": {
                    "x": {
                        "filename": "x.png",
                        "prompt": "A faithful portrait test prompt.",
                        "sha256": hashlib.sha256(PNG).hexdigest(),
                        "elapsed_seconds": 1.0,
                    }
                },
            }).encode()
        if filename == "x.png":
            return PNG
        raise FileNotFoundError(filename)


def test_stable_provider_ignores_stale_report_and_never_calls_status():
    worker = StableFakeWorker()
    provider = StableKaggleImageV2Provider(
        worker=worker,
        poll_interval=0,
        max_poll_attempts=3,
    )
    result = provider.generate_many((ImageRequestV2("x", "Tạo ảnh chân dung", 7),))[0]

    assert result.data == PNG
    assert worker.report_reads == 2
    assert 'report["run_token"] = RUN_TOKEN' in worker.source
    assert "AIKA_IMAGE_RUN" in worker.source


def test_stable_source_has_generic_subject_identity_lock_without_species_patch():
    worker = StableFakeWorker()
    provider = StableKaggleImageV2Provider(
        worker=worker,
        poll_interval=0,
        max_poll_attempts=3,
    )
    provider.generate_many((ImageRequestV2("x", "một con trâu nước hoạt hình 3D", 7),))

    source = worker.source
    assert "def contract_for(user_request):" in source
    assert "def extract_subject_locks(user_request):" in source
    assert "def prompt_has_subject_locks(prompt, locks):" in source
    assert "def enforce_semantic_integrity(" in source
    assert "shortest exact noun phrase copied verbatim" in source
    assert "Silently verify each translation before output" in source
    assert "subject integrity guard failed before image generation" in source
    assert 'if "áo dài" in user_request.casefold():' in source
    assert "+ contract_for(item[\"command\"]) + \" \"" in source
    assert 'repaired_prompt = enforce_semantic_integrity(item["command"], prompt)' in source
    assert '"integrity_checked"' in source
    assert "literal-faithful English image-generation description" in source
    assert "+ base_contract" not in source

    # Regression guard: the mechanism must stay generic, never become a species table.
    folded = source.casefold()
    assert "subject_class" not in folded
    assert "species table" not in folded
    assert "if species" not in folded
    assert "if animal" not in folded
    assert "crab" not in folded
    assert "frog" not in folded
    assert "water buffalo" not in folded


def test_subject_lock_repairs_only_on_mismatch_and_validates_before_flux():
    provider = StableKaggleImageV2Provider(
        worker=StableFakeWorker(),
        poll_interval=0,
        max_poll_attempts=3,
    )
    source = provider._harden_recaption_source(provider._build_worker_source((
        ImageRequestV2("x", "hai chủ thể hoạt hình 3D", 7),
    )))

    assert "if not locks or prompt_has_subject_locks(candidate_prompt, locks):" in source
    assert "These locked subjects are mandatory" in source
    assert "each English label must appear verbatim" in source
    assert "not prompt_has_subject_locks(repaired, locks)" in source


def test_stable_natural_batch_keeps_sibling_variants_separate():
    provider = StableKaggleImageV2Provider(
        worker=StableFakeWorker(),
        poll_interval=0,
        max_poll_attempts=3,
    )
    source = provider._harden_recaption_source(provider._build_command_worker_source(
        "Tạo hai ảnh riêng với hai chủ thể khác nhau",
        seed=7,
        max_images=2,
    ))

    assert "variant_scope=(len(parts) > 1)" in source
    assert "do not merge subjects from other requested images" in source


def test_production_executor_uses_stable_provider_and_is_bounded(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    provider = ImageExecutor()._provider()

    assert isinstance(provider, StableKaggleImageV2Provider)
    assert provider.max_poll_attempts == 180
    assert provider.poll_interval == 3
    assert provider.max_poll_attempts * provider.poll_interval == 540
