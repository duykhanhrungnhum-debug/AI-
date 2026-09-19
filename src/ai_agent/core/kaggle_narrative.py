"""Single-session narrative transform/review/repair on Kaggle GPU."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
import textwrap
import time

from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker
from .narrative_pipeline import (
    NarrativeBrief,
    NarrativeResult,
    ScriptQualityReport,
)


@dataclass
class KaggleNarrativeProcessor:
    """Load one open LLM once, then analyze, rewrite, review, and repair."""

    worker: KaggleGpuWorker
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    kernel_slug: str = "ai-agent-narrative-worker"
    poll_interval: float = 15.0
    max_poll_attempts: int = 120
    max_revisions: int = 2
    temperature: float = 0.0
    provider: str = "kaggle-gpu-open-model"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if not self.kernel_slug.strip() or "/" in self.kernel_slug:
            raise ValueError("kernel_slug must be a plain Kaggle slug")
        if self.poll_interval < 0 or self.max_poll_attempts <= 0:
            raise ValueError("poll configuration must be valid")
        if self.max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")

    def process(self, source_text: str, *, target_language: str = "Vietnamese") -> NarrativeResult:
        assert_core_invariants()
        source_text = source_text.strip()
        target_language = target_language.strip()
        if not source_text:
            raise ValueError("source_text must not be empty")
        if not target_language:
            raise ValueError("target_language must not be empty")

        source = self._build_worker_source(source_text, target_language)
        compile(source, "<kaggle-narrative-worker>", "exec")
        title = self.kernel_slug.replace("-", " ").title()
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=title,
            source=source,
            enable_internet=True,
            is_private=True,
        )
        self._wait()

        raw = self.worker.download_output_file(self.kernel_slug, "narrative_result.json")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Kaggle narrative result is invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Kaggle narrative result must be an object")
        if data.get("model") != self.model:
            raise ValueError("Kaggle narrative result model mismatch")
        if data.get("source_sha256") != sha256(source_text.encode("utf-8")).hexdigest():
            raise ValueError("Kaggle narrative source hash mismatch")
        gpu_name = str(data.get("gpu_name", "")).strip()
        if not gpu_name:
            raise ValueError("Kaggle narrative result lacks GPU evidence")

        brief_data = data.get("brief")
        review_data = data.get("review")
        script = data.get("script")
        if not isinstance(brief_data, dict) or not isinstance(review_data, dict):
            raise ValueError("Kaggle narrative result lacks brief/review")
        if not isinstance(script, str) or not script.strip():
            raise ValueError("Kaggle narrative result has empty script")
        script = script.strip()
        digest = sha256(script.encode("utf-8")).hexdigest()
        if data.get("script_sha256") != digest:
            raise ValueError("Kaggle narrative script hash mismatch")

        brief = NarrativeBrief(
            characters=self._string_tuple(brief_data.get("characters"), "characters"),
            events=self._string_tuple(brief_data.get("events"), "events"),
            must_preserve=self._string_tuple(brief_data.get("must_preserve"), "must_preserve"),
        )
        issues = list(self._string_tuple(review_data.get("issues"), "issues"))
        deterministic = self._deterministic_issues(script)
        for issue in deterministic:
            if issue not in issues:
                issues.append(issue)

        model_passed = review_data.get("passed")
        if not isinstance(model_passed, bool):
            raise ValueError("Kaggle narrative review passed must be boolean")
        review = ScriptQualityReport(
            passed=(model_passed and not issues),
            issues=tuple(issues),
            checks=(
                "deterministic:non_empty",
                "deterministic:no_exact_duplicate_paragraphs",
                "deterministic:no_placeholder_markers",
                "review:fidelity_and_coherence",
            ),
            reviewer=f"{self.provider}/{self.model}",
        )
        revisions = data.get("revision_count")
        if not isinstance(revisions, int) or revisions < 0:
            raise ValueError("Kaggle narrative revision_count is invalid")

        return NarrativeResult(
            brief=brief,
            script=script,
            review=review,
            revision_count=revisions,
            evidence=(
                f"kaggle_kernel:{submission.ref}",
                f"gpu:{gpu_name}",
                f"model:{self.model}",
                f"source_sha256:{data['source_sha256']}",
                f"script_sha256:{digest}",
                f"script_revisions:{revisions}",
            ),
        )

    def _wait(self) -> None:
        for _ in range(self.max_poll_attempts):
            status = self.worker.status(self.kernel_slug)
            if status.terminal:
                if not status.successful:
                    try:
                        logs = self.worker.logs(self.kernel_slug).strip()
                    except Exception as exc:
                        logs = f"<failed to retrieve Kaggle logs: {exc}>"
                    if len(logs) > 5000:
                        logs = logs[-5000:]
                    raise RuntimeError(
                        f"Kaggle narrative worker failed: {status.status} "
                        f"{status.failure_message}\nKaggle logs:\n{logs}"
                    )
                return
            if self.poll_interval:
                time.sleep(self.poll_interval)
        raise TimeoutError("Timed out waiting for Kaggle narrative worker")

    @staticmethod
    def _string_tuple(value: object, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be an array of strings")
        return tuple(item.strip() for item in value if item.strip())

    @staticmethod
    def _deterministic_issues(script: str) -> tuple[str, ...]:
        issues: list[str] = []
        if not script.strip():
            issues.append("script is empty")
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip().casefold()
            for paragraph in re.split(r"\n\s*\n", script)
            if len(paragraph.strip()) >= 20
        ]
        seen: set[str] = set()
        for paragraph in paragraphs:
            if paragraph in seen:
                issues.append("exact duplicate paragraph detected")
                break
            seen.add(paragraph)
        if re.search(r"\b(?:TODO|TBD)\b|\[(?:insert|placeholder)[^]]*\]", script, re.IGNORECASE):
            issues.append("placeholder marker detected")
        return tuple(issues)

    def _build_worker_source(self, source_text: str, target_language: str) -> str:
        config = {
            "model": self.model,
            "source": source_text,
            "target_language": target_language,
            "max_revisions": self.max_revisions,
            "temperature": self.temperature,
        }
        config_json = json.dumps(config, ensure_ascii=False)
        return textwrap.dedent(
            rf"""
            from __future__ import annotations

            from hashlib import sha256
            import json
            import re
            import subprocess
            import sys
            from pathlib import Path

            CONFIG = json.loads({config_json!r})

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers<5", "accelerate<2", "safetensors", "sentencepiece",
                ])
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model"],
                torch_dtype=torch.float16,
                device_map="auto",
            )

            def generate(prompt, max_new_tokens):
                messages = [{{"role": "user", "content": prompt}}]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
                kwargs = {{
                    "max_new_tokens": max_new_tokens,
                    "do_sample": float(CONFIG["temperature"]) > 0,
                }}
                if kwargs["do_sample"]:
                    kwargs["temperature"] = float(CONFIG["temperature"])
                generated = model.generate(**inputs, **kwargs)
                new_tokens = generated[:, inputs.input_ids.shape[1]:]
                text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                if not text:
                    raise RuntimeError("local model returned empty text")
                return text

            def parse_json(raw, label):
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{{label}} is not valid JSON: {{cleaned[:300]}}") from exc
                if not isinstance(data, dict):
                    raise RuntimeError(f"{{label}} must be a JSON object")
                return data

            def string_list(value, label):
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    raise RuntimeError(f"{{label}} must be an array of strings")
                return [item.strip() for item in value if item.strip()]

            def deterministic_issues(script):
                issues = []
                paragraphs = [
                    re.sub(r"\s+", " ", paragraph).strip().casefold()
                    for paragraph in re.split(r"\n\s*\n", script)
                    if len(paragraph.strip()) >= 20
                ]
                seen = set()
                for paragraph in paragraphs:
                    if paragraph in seen:
                        issues.append("exact duplicate paragraph detected")
                        break
                    seen.add(paragraph)
                if re.search(r"\b(?:TODO|TBD)\b|\[(?:insert|placeholder)[^]]*\]", script, re.IGNORECASE):
                    issues.append("placeholder marker detected")
                return issues

            source = CONFIG["source"]
            analysis_prompt = (
                "NARRATIVE_ANALYSIS\n"
                "Read the source carefully. Extract only information supported by the source. "
                "Return ONLY JSON with keys characters, events, must_preserve; each value must be an array of strings. "
                "Do not translate or rewrite yet.\nSOURCE:\n" + source
            )
            brief = parse_json(generate(analysis_prompt, 700), "narrative analysis")
            brief = {{
                "characters": string_list(brief.get("characters"), "characters"),
                "events": string_list(brief.get("events"), "events"),
                "must_preserve": string_list(brief.get("must_preserve"), "must_preserve"),
            }}
            brief_json = json.dumps(brief, ensure_ascii=False)

            draft_prompt = (
                "NARRATIVE_REWRITE\n"
                f"Rewrite the source naturally in {{CONFIG['target_language']}}. "
                "Preserve characters, event order, relationships, causality, and important facts. "
                "Do not invent new plot facts. Avoid repetitive wording and broken sentences. "
                "Return only the rewritten script.\n"
                f"BRIEF:\n{{brief_json}}\nSOURCE:\n{{source}}"
            )
            script = generate(draft_prompt, 1400).strip()

            def review_script(current):
                deterministic = deterministic_issues(current)
                review_prompt = (
                    "NARRATIVE_REVIEW\n"
                    "Compare SCRIPT against SOURCE and BRIEF. Check wrong meaning, changed character identity, "
                    "missing or reordered major events, invented facts, contradictions, nonsense, and excessive repetition. "
                    'Return ONLY JSON: {{"passed": true|false, "issues": [strings]}}. '
                    "passed may be true only when no material issue is found.\n"
                    f"BRIEF:\n{{brief_json}}\nSOURCE:\n{{source}}\nSCRIPT:\n{{current}}"
                )
                review = parse_json(generate(review_prompt, 700), "narrative review")
                passed = review.get("passed")
                if not isinstance(passed, bool):
                    raise RuntimeError("review passed must be boolean")
                issues = string_list(review.get("issues"), "issues")
                for issue in deterministic:
                    if issue not in issues:
                        issues.append(issue)
                return {{"passed": passed and not issues, "issues": issues}}

            review = review_script(script)
            revisions = 0
            seen = {{sha256(re.sub(r"\s+", " ", script).strip().casefold().encode("utf-8")).hexdigest()}}

            while not review["passed"] and revisions < int(CONFIG["max_revisions"]):
                issues_text = "\n".join(f"- {{item}}" for item in review["issues"]) or "- review did not pass"
                repair_prompt = (
                    "NARRATIVE_REPAIR\n"
                    f"Repair the script in {{CONFIG['target_language']}}. Correct every listed issue while preserving "
                    "the source facts and event order. Do not introduce unrelated changes or new plot facts. "
                    "Return only the repaired script.\n"
                    f"ISSUES:\n{{issues_text}}\nBRIEF:\n{{brief_json}}\nSOURCE:\n{{source}}\nCURRENT SCRIPT:\n{{script}}"
                )
                repaired = generate(repair_prompt, 1400).strip()
                revisions += 1
                fingerprint = sha256(
                    re.sub(r"\s+", " ", repaired).strip().casefold().encode("utf-8")
                ).hexdigest()
                script = repaired
                if fingerprint in seen:
                    if "repair loop detected: repeated script" not in review["issues"]:
                        review["issues"].append("repair loop detected: repeated script")
                    review["passed"] = False
                    break
                seen.add(fingerprint)
                review = review_script(script)

            result = {{
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "source_sha256": sha256(source.encode("utf-8")).hexdigest(),
                "script_sha256": sha256(script.encode("utf-8")).hexdigest(),
                "brief": brief,
                "script": script,
                "review": review,
                "revision_count": revisions,
            }}
            Path("/kaggle/working/narrative_result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print("AI_AGENT_NARRATIVE_OK")
            print(json.dumps({{
                "review_passed": review["passed"],
                "revision_count": revisions,
                "issues": review["issues"],
                "gpu_name": gpu_name,
                "model": CONFIG["model"],
            }}, ensure_ascii=False, indent=2))
            """
        ).strip() + "\n"
