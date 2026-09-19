"""Single-session narrative transform/review/repair on Kaggle GPU."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
import textwrap
import time

from .editorial_lessons import DEFAULT_EDITORIAL_LESSONS
from .invariants import assert_core_invariants
from .kaggle_worker import KaggleGpuWorker
from .narrative_pipeline import NarrativeBrief, NarrativeResult, ScriptQualityReport


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
    lessons: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EDITORIAL_LESSONS)

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
        if any(not lesson.strip() for lesson in self.lessons):
            raise ValueError("editorial lessons must be non-empty")

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
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=self.kernel_slug.replace("-", " ").title(),
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
        expected_source_hash = sha256(source_text.encode("utf-8")).hexdigest()
        if data.get("source_sha256") != expected_source_hash:
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
        for issue in self._deterministic_issues(script):
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
                "review:fact_id_checklist",
                "review:fact_bound_contradictions",
            ),
            reviewer=f"{self.provider}/{self.model}",
        )

        revisions = data.get("revision_count")
        if not isinstance(revisions, int) or revisions < 0:
            raise ValueError("Kaggle narrative revision_count is invalid")
        strategy_history = data.get("strategy_history")
        if not isinstance(strategy_history, list) or any(not isinstance(item, str) for item in strategy_history):
            raise ValueError("Kaggle narrative strategy_history is invalid")

        return NarrativeResult(
            brief=brief,
            script=script,
            review=review,
            revision_count=revisions,
            evidence=(
                f"kaggle_kernel:{submission.ref}",
                f"gpu:{gpu_name}",
                f"model:{self.model}",
                f"source_sha256:{expected_source_hash}",
                f"script_sha256:{digest}",
                f"script_revisions:{revisions}",
                f"editorial_lessons_applied:{len(self.lessons)}",
                f"repair_strategies:{','.join(strategy_history) if strategy_history else 'none'}",
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
            "lessons": list(self.lessons),
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
                    messages, tokenize=False, add_generation_prompt=True
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
                fence = chr(96) * 3
                if cleaned.startswith(fence):
                    lines = cleaned.splitlines()
                    if lines and lines[0].startswith(fence):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == fence:
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{{label}} is not valid JSON: {{cleaned[:400]}}") from exc
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

            def fingerprint(script):
                normalized = re.sub(r"\s+", " ", script).strip().casefold()
                return sha256(normalized.encode("utf-8")).hexdigest()

            source = CONFIG["source"]
            lessons_text = "\n".join(f"- {{item}}" for item in CONFIG["lessons"])
            analysis_prompt = (
                "NARRATIVE_ANALYSIS\n"
                "Read SOURCE carefully. Extract only facts explicitly supported by SOURCE. "
                "Return ONLY JSON with keys characters, events, must_preserve; each value is an array of short strings. "
                "characters contains stable names/identities only. events stays chronological. "
                "must_preserve contains concrete details whose loss changes meaning.\n"
                f"EDITORIAL_LESSONS:\n{{lessons_text}}\nSOURCE:\n{{source}}"
            )
            brief = parse_json(generate(analysis_prompt, 800), "narrative analysis")
            brief = {{
                "characters": string_list(brief.get("characters"), "characters"),
                "events": string_list(brief.get("events"), "events"),
                "must_preserve": string_list(brief.get("must_preserve"), "must_preserve"),
            }}

            facts = []
            for index, item in enumerate(brief["characters"], 1):
                facts.append({{"id": f"C{{index}}", "kind": "character", "fact": item}})
            for index, item in enumerate(brief["events"], 1):
                facts.append({{"id": f"E{{index}}", "kind": "event", "fact": item}})
            for index, item in enumerate(brief["must_preserve"], 1):
                facts.append({{"id": f"P{{index}}", "kind": "must_preserve", "fact": item}})
            if not facts:
                raise RuntimeError("narrative analysis produced an empty fact checklist")

            facts_json = json.dumps(facts, ensure_ascii=False)
            draft_prompt = (
                "NARRATIVE_REWRITE\n"
                f"Rewrite SOURCE naturally in {{CONFIG['target_language']}}. "
                "Every FACT_CHECKLIST item is mandatory and must remain true. Preserve character names exactly. "
                "Preserve event order. Do not add a new major plot event. Return only the rewritten script.\n"
                f"EDITORIAL_LESSONS:\n{{lessons_text}}\nFACT_CHECKLIST:\n{{facts_json}}\nSOURCE:\n{{source}}"
            )
            script = generate(draft_prompt, 1400).strip()

            def review_script(current):
                review_prompt = (
                    "NARRATIVE_FACT_REVIEW\n"
                    "Audit SCRIPT only against FACT_CHECKLIST and SOURCE. For EVERY fact ID return exactly one check. "
                    "A fact is preserved when SCRIPT expresses the same meaning in the target language; wording may differ. "
                    "Never reinterpret surrounding translated words as a renamed character. "
                    "For contradictions, cite a valid fact_id and an exact short script_claim. "
                    "Return ONLY JSON with keys checks and contradictions. Each check has fact_id, preserved, script_evidence. "
                    "Each contradiction has fact_id, script_claim, reason. contradictions must be empty unless SCRIPT "
                    "directly conflicts with a checklist fact.\n"
                    f"EDITORIAL_LESSONS:\n{{lessons_text}}\nFACT_CHECKLIST:\n{{facts_json}}\n"
                    f"SOURCE:\n{{source}}\nSCRIPT:\n{{current}}"
                )
                data = parse_json(generate(review_prompt, 1100), "narrative fact review")
                checks = data.get("checks")
                contradictions = data.get("contradictions")
                if not isinstance(checks, list) or not isinstance(contradictions, list):
                    raise RuntimeError("review checks and contradictions must be arrays")

                fact_by_id = {{item["id"]: item for item in facts}}
                seen_ids = set()
                failed_ids = []
                normalized_checks = []
                for item in checks:
                    if not isinstance(item, dict):
                        raise RuntimeError("review check must be an object")
                    fact_id = item.get("fact_id")
                    preserved = item.get("preserved")
                    evidence = item.get("script_evidence")
                    if fact_id not in fact_by_id:
                        raise RuntimeError(f"review used unknown fact ID: {{fact_id}}")
                    if fact_id in seen_ids:
                        raise RuntimeError(f"review duplicated fact ID: {{fact_id}}")
                    if not isinstance(preserved, bool) or not isinstance(evidence, str):
                        raise RuntimeError("review check has invalid fields")
                    seen_ids.add(fact_id)
                    if fact_by_id[fact_id]["kind"] == "character":
                        character = fact_by_id[fact_id]["fact"].strip()
                        if character and character.casefold() not in current.casefold():
                            preserved = False
                            evidence = ""
                    if not preserved:
                        failed_ids.append(fact_id)
                    normalized_checks.append({{
                        "fact_id": fact_id,
                        "preserved": preserved,
                        "script_evidence": evidence.strip(),
                    }})

                expected_ids = set(fact_by_id)
                omitted = expected_ids - seen_ids
                for fact_id in sorted(omitted):
                    failed_ids.append(fact_id)
                    normalized_checks.append({{
                        "fact_id": fact_id,
                        "preserved": False,
                        "script_evidence": "",
                    }})

                normalized_contradictions = []
                contradiction_ids = []
                for item in contradictions:
                    if not isinstance(item, dict):
                        raise RuntimeError("contradiction must be an object")
                    fact_id = item.get("fact_id")
                    claim = item.get("script_claim")
                    reason = item.get("reason")
                    if fact_id not in fact_by_id:
                        raise RuntimeError(f"contradiction used unknown fact ID: {{fact_id}}")
                    if not isinstance(claim, str) or not claim.strip():
                        raise RuntimeError("contradiction script_claim must be text")
                    if claim.strip().casefold() not in current.casefold():
                        continue
                    if not isinstance(reason, str) or not reason.strip():
                        raise RuntimeError("contradiction reason must be text")
                    contradiction_ids.append(fact_id)
                    normalized_contradictions.append({{
                        "fact_id": fact_id,
                        "script_claim": claim.strip(),
                        "reason": reason.strip(),
                    }})

                issues = list(deterministic_issues(current))
                for fact_id in failed_ids:
                    issues.append(f"missing fact {{fact_id}}: {{fact_by_id[fact_id]['fact']}}")
                for item in normalized_contradictions:
                    issues.append(f"contradiction {{item['fact_id']}}: {{item['reason']}}")
                return {{
                    "passed": not issues,
                    "issues": issues,
                    "checks": normalized_checks,
                    "failed_fact_ids": failed_ids,
                    "contradictions": normalized_contradictions,
                    "contradiction_fact_ids": contradiction_ids,
                }}

            review = review_script(script)
            revisions = 0
            seen = {{fingerprint(script)}}
            strategy_history = []

            while not review["passed"] and revisions < int(CONFIG["max_revisions"]):
                fact_by_id = {{item["id"]: item for item in facts}}
                repair_payload = json.dumps({{
                    "missing_or_unpreserved": [
                        fact_by_id[fact_id] for fact_id in review["failed_fact_ids"]
                        if fact_id in fact_by_id
                    ],
                    "contradictions": review["contradictions"],
                }}, ensure_ascii=False)
                repair_prompt = (
                    "NARRATIVE_TARGETED_REPAIR\n"
                    f"Repair CURRENT_SCRIPT in {{CONFIG['target_language']}}. Change only what is necessary to resolve "
                    "REPAIR_TARGETS. Preserve all checklist facts that already pass and preserve event order. "
                    "Character names must remain exact. Return only the full repaired script.\n"
                    f"EDITORIAL_LESSONS:\n{{lessons_text}}\nFACT_CHECKLIST:\n{{facts_json}}\n"
                    f"REPAIR_TARGETS:\n{{repair_payload}}\nSOURCE:\n{{source}}\nCURRENT_SCRIPT:\n{{script}}"
                )
                repaired = generate(repair_prompt, 1400).strip()
                revisions += 1
                strategy_history.append("targeted_repair")
                repaired_fingerprint = fingerprint(repaired)

                if repaired_fingerprint in seen:
                    rebuild_prompt = (
                        "NARRATIVE_REBUILD_FROM_FACTS\n"
                        f"Create a fresh {{CONFIG['target_language']}} script from SOURCE and FACT_CHECKLIST. "
                        "Do not copy CURRENT_SCRIPT sentence structure. Include every checklist fact, preserve event order "
                        "and exact character names, and add no new major plot event. Return only the rebuilt script.\n"
                        f"EDITORIAL_LESSONS:\n{{lessons_text}}\nFACT_CHECKLIST:\n{{facts_json}}\n"
                        f"SOURCE:\n{{source}}\nCURRENT_SCRIPT_TO_AVOID_COPYING:\n{{script}}"
                    )
                    rebuilt = generate(rebuild_prompt, 1400).strip()
                    strategy_history.append("fact_rebuild")
                    rebuilt_fingerprint = fingerprint(rebuilt)
                    if rebuilt_fingerprint in seen:
                        review["passed"] = False
                        if "repair loop detected after strategy switch" not in review["issues"]:
                            review["issues"].append("repair loop detected after strategy switch")
                        break
                    repaired = rebuilt
                    repaired_fingerprint = rebuilt_fingerprint

                seen.add(repaired_fingerprint)
                script = repaired
                review = review_script(script)

            result = {{
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "source_sha256": sha256(source.encode("utf-8")).hexdigest(),
                "script_sha256": sha256(script.encode("utf-8")).hexdigest(),
                "brief": brief,
                "fact_checklist": facts,
                "script": script,
                "review": review,
                "revision_count": revisions,
                "strategy_history": strategy_history,
                "lessons_applied": CONFIG["lessons"],
            }}
            Path("/kaggle/working/narrative_result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print("AI_AGENT_NARRATIVE_OK")
            print(json.dumps({{
                "review_passed": review["passed"],
                "revision_count": revisions,
                "failed_fact_ids": review["failed_fact_ids"],
                "contradiction_fact_ids": review["contradiction_fact_ids"],
                "strategy_history": strategy_history,
                "gpu_name": gpu_name,
                "model": CONFIG["model"],
            }}, ensure_ascii=False, indent=2))
            """
        ).strip() + "\n"
