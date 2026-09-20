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
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    semantic_threshold: float = 0.62
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
        if not self.semantic_model.strip():
            raise ValueError("semantic_model is required")
        if not 0 < self.semantic_threshold <= 1:
            raise ValueError("semantic_threshold must be in (0, 1]")
        if any(not lesson.strip() for lesson in self.lessons):
            raise ValueError("editorial lessons must be non-empty")

    def process(
        self,
        source_text: str,
        *,
        target_language: str = "Vietnamese",
        existing_script: str | None = None,
    ) -> NarrativeResult:
        assert_core_invariants()
        source_text = source_text.strip()
        target_language = target_language.strip()
        if not source_text:
            raise ValueError("source_text must not be empty")
        if not target_language:
            raise ValueError("target_language must not be empty")
        if existing_script is not None:
            existing_script = existing_script.strip()
            if not existing_script:
                raise ValueError("existing_script must not be empty when provided")

        source = self._build_worker_source(
            source_text,
            target_language,
            existing_script=existing_script,
        )
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
        adjudicated_fact_ids = review_data.get("adjudicated_fact_ids", [])
        if not isinstance(adjudicated_fact_ids, list) or any(
            not isinstance(item, str) for item in adjudicated_fact_ids
        ):
            raise ValueError("Kaggle narrative adjudicated_fact_ids is invalid")
        semantic_scores = review_data.get("semantic_scores", {})
        if not isinstance(semantic_scores, dict) or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            for key, value in semantic_scores.items()
        ):
            raise ValueError("Kaggle narrative semantic_scores is invalid")
        semantic_score_evidence = ",".join(
            f"{key}={float(value):.3f}"
            for key, value in sorted(semantic_scores.items())
        ) or "none"
        review = ScriptQualityReport(
            passed=(model_passed and not issues),
            issues=tuple(issues),
            checks=(
                "deterministic:non_empty",
                "deterministic:no_exact_duplicate_paragraphs",
                "deterministic:no_placeholder_markers",
                "review:fact_id_checklist",
                "review:multilingual_semantic_fact_adjudication",
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
                f"review_adjudicated_fact_ids:{','.join(adjudicated_fact_ids) if adjudicated_fact_ids else 'none'}",
                f"review_semantic_scores:{semantic_score_evidence}",
                f"semantic_model:{self.semantic_model}",
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

    def _build_worker_source(
        self,
        source_text: str,
        target_language: str,
        *,
        existing_script: str | None = None,
    ) -> str:
        config = {
            "model": self.model,
            "source": source_text,
            "target_language": target_language,
            "existing_script": existing_script,
            "review_only": existing_script is not None,
            "max_revisions": self.max_revisions,
            "temperature": self.temperature,
            "semantic_model": self.semantic_model,
            "semantic_threshold": self.semantic_threshold,
            "lessons": list(self.lessons),
        }
        config_json = json.dumps(config, ensure_ascii=False)
        return textwrap.dedent(
            rf"""
            from __future__ import annotations

            from hashlib import sha256
            import json
            import os
            import re
            import subprocess
            import sys
            from pathlib import Path

            CONFIG = json.loads({config_json!r})
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            try:
                import bitsandbytes
                import torch
                from sentence_transformers import SentenceTransformer
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            except ImportError:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
                    "transformers<5", "accelerate<2", "safetensors", "sentencepiece",
                    "sentence-transformers>=3,<4", "bitsandbytes>=0.46.1,<1",
                ])
                import bitsandbytes
                import torch
                from sentence_transformers import SentenceTransformer
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")

            gpu_name = torch.cuda.get_device_name(0)
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model"],
                quantization_config=quantization,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            model.eval()
            semantic_model = SentenceTransformer(
                CONFIG["semantic_model"],
                device="cpu",
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
                    "repetition_penalty": 1.08,
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
                decoder = json.JSONDecoder()
                candidates = [cleaned]
                first_object = cleaned.find("{{")
                if first_object > 0:
                    candidates.append(cleaned[first_object:])
                last_error = None
                for candidate in candidates:
                    try:
                        data, _ = decoder.raw_decode(candidate.lstrip())
                    except json.JSONDecodeError as exc:
                        last_error = exc
                        continue
                    if not isinstance(data, dict):
                        raise RuntimeError(f"{{label}} must be a JSON object")
                    return data
                raise RuntimeError(
                    f"{{label}} is not valid JSON: {{cleaned[:400]}}"
                ) from last_error

            def string_list(value, label, limit=None):
                if value is None:
                    return []
                if isinstance(value, str):
                    value = [value]
                elif isinstance(value, dict):
                    nested = None
                    for key in ("items", "values", label):
                        candidate = value.get(key)
                        if isinstance(candidate, list):
                            nested = candidate
                            break
                    value = nested or []
                elif not isinstance(value, list):
                    return []

                result = []
                seen = set()
                for item in value:
                    if isinstance(item, dict):
                        text_value = ""
                        for key in ("name", "fact", "text", "value", "label"):
                            candidate = item.get(key)
                            if isinstance(candidate, str) and candidate.strip():
                                text_value = candidate
                                break
                        item = text_value
                    elif not isinstance(item, str):
                        continue
                    item = re.sub(r"\s+", " ", item).strip()
                    key = item.casefold()
                    if not item or key in seen:
                        continue
                    seen.add(key)
                    result.append(item)
                    if limit is not None and len(result) >= limit:
                        break
                return result

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

            def split_source(text, max_chars=1400):
                paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
                chunks = []
                current = ""
                for paragraph in paragraphs:
                    if current and len(current) + len(paragraph) + 2 > max_chars:
                        chunks.append(current)
                        current = paragraph
                    else:
                        current = paragraph if not current else current + "\n\n" + paragraph
                if current:
                    chunks.append(current)
                return chunks or [text]

            def analyze_piece(piece, index, total):
                prompt = (
                    "NARRATIVE_ANALYSIS\n"
                    f"Analyze SOURCE_CHUNK {{index}}/{{total}} only. Extract only story-critical facts explicitly supported by it. "
                    f"Write every extracted fact in {{CONFIG['target_language']}} so it can be checked against the target-language script. "
                    "Return one JSON object only, with keys characters, events, must_preserve. "
                    "characters: at most 3 story-active named people or sentient beings who materially act in this episode; "
                    "exclude figures mentioned only in poems, cosmology, background history, examples, or quoted references. "
                    "events: at most 4 chronological plot events that materially advance the episode. "
                    "must_preserve: at most 3 concrete plot details whose loss would change the story outcome. "
                    "Exclude ornamental scenery, repeated restatements, and facts already represented by an event. "
                    "No duplicates, no commentary, no markdown fences.\n"
                    f"SOURCE_CHUNK:\n{{piece}}"
                )
                return parse_json(generate(prompt, 360), f"narrative analysis chunk {{index}}")

            if CONFIG["review_only"]:
                chunks = split_source(source)
                merged = {{"characters": [], "events": [], "must_preserve": []}}
                for index, piece in enumerate(chunks, 1):
                    piece_brief = analyze_piece(piece, index, len(chunks))
                    for key in merged:
                        merged[key].extend(
                            string_list(piece_brief.get(key, []), key)
                        )
                brief = {{
                    "characters": string_list(merged["characters"], "characters", 6),
                    "events": string_list(merged["events"], "events", 12),
                    "must_preserve": string_list(merged["must_preserve"], "must_preserve", 8),
                }}
                brief["characters"] = []

                def drop_event_duplicate_preserve(events, preserves):
                    if not events or not preserves:
                        return preserves
                    embeddings = semantic_model.encode(
                        [*events, *preserves],
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    event_vectors = embeddings[:len(events)]
                    preserve_vectors = embeddings[len(events):]
                    kept = []
                    for item, vector in zip(preserves, preserve_vectors):
                        best_event_score = float(
                            torch.max(torch.matmul(event_vectors, vector)).item()
                        )
                        if best_event_score >= float(CONFIG["semantic_threshold"]):
                            continue
                        kept.append(item)
                    return kept

                brief["must_preserve"] = drop_event_duplicate_preserve(
                    brief["events"],
                    brief["must_preserve"],
                )
            else:
                analysis_prompt = (
                    "NARRATIVE_ANALYSIS\n"
                    "Read SOURCE carefully. Extract only facts explicitly supported by SOURCE. "
                    "Return ONLY JSON with keys characters, events, must_preserve; each value is an array of short strings. "
                    "characters contains stable names/identities only, never places. events stays chronological. "
                    "must_preserve contains concrete details whose loss changes meaning. No duplicates.\n"
                    f"EDITORIAL_LESSONS:\n{{lessons_text}}\nSOURCE:\n{{source}}"
                )
                brief_raw = parse_json(generate(analysis_prompt, 900), "narrative analysis")
                brief = {{
                    "characters": string_list(brief_raw.get("characters"), "characters", 12),
                    "events": string_list(brief_raw.get("events"), "events", 24),
                    "must_preserve": string_list(brief_raw.get("must_preserve"), "must_preserve", 24),
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
            fact_by_id = {{item["id"]: item for item in facts}}
            draft_prompt = (
                "NARRATIVE_REWRITE\n"
                f"Rewrite SOURCE naturally in {{CONFIG['target_language']}}. "
                "Every FACT_CHECKLIST item is mandatory and must remain true. Preserve character names exactly. "
                "Preserve event order. Do not add a new major plot event. Return only the rewritten script.\n"
                f"EDITORIAL_LESSONS:\n{{lessons_text}}\nFACT_CHECKLIST:\n{{facts_json}}\nSOURCE:\n{{source}}"
            )
            if CONFIG["review_only"]:
                script = str(CONFIG["existing_script"]).strip()
            else:
                script = generate(draft_prompt, 1400).strip()

            def semantic_adjudicate_fact(fact_id, current):
                fact = fact_by_id[fact_id]
                segments = [
                    item.strip()
                    for item in re.split(r"(?<=[.!?])\s+|\n+", current)
                    if item.strip()
                ]
                if not segments:
                    return False, "", 0.0, ""
                embeddings = semantic_model.encode(
                    [fact["fact"], *segments],
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                scores = torch.matmul(embeddings[1:], embeddings[0])
                ranked = torch.argsort(scores, descending=True)[:4].tolist()
                best_index = int(ranked[0])
                best_score = float(scores[best_index].item())
                evidence = segments[best_index]
                candidate_evidence = "\n".join(
                    f"- {{segments[int(index)]}}" for index in ranked
                )
                return (
                    best_score >= float(CONFIG["semantic_threshold"]),
                    evidence,
                    best_score,
                    candidate_evidence,
                )

            def adjudicate_fact(fact_id, current, candidate_evidence=""):
                fact = fact_by_id[fact_id]
                script_scope = candidate_evidence.strip() or current
                adjudication_prompt = (
                    "NARRATIVE_FACT_ADJUDICATION\n"
                    "Decide only whether SCRIPT_EVIDENCE expresses the same concrete meaning as FACT in the target language. "
                    "Do not judge style and do not infer unrelated omissions. If preserved, script_evidence must be an "
                    "exact contiguous substring copied from SCRIPT_EVIDENCE. Return ONLY JSON with keys preserved and "
                    "script_evidence. No commentary or markdown.\n"
                    f"FACT_ID: {{fact_id}}\nFACT: {{fact['fact']}}\nSCRIPT_EVIDENCE:\n{{script_scope}}"
                )
                try:
                    data = parse_json(
                        generate(adjudication_prompt, 350),
                        "narrative fact adjudication",
                    )
                except RuntimeError:
                    return False, ""
                preserved = data.get("preserved")
                if isinstance(preserved, str):
                    normalized = preserved.strip().casefold()
                    if normalized in {"true", "yes", "preserved", "1"}:
                        preserved = True
                    elif normalized in {"false", "no", "missing", "0"}:
                        preserved = False
                evidence = data.get("script_evidence", data.get("evidence", ""))
                if not isinstance(preserved, bool):
                    return False, ""
                if not isinstance(evidence, str):
                    evidence = ""
                evidence = evidence.strip()
                if preserved and (
                    not evidence
                    or evidence.casefold() not in current.casefold()
                ):
                    preserved = False
                return preserved, evidence

            def review_script(current):
                if CONFIG["review_only"]:
                    normalized_checks = []
                    failed_ids = []
                    adjudicated_fact_ids = []
                    semantic_scores = {{}}
                    for fact_id in fact_by_id:
                        (
                            semantic_preserved,
                            semantic_evidence,
                            semantic_score,
                            semantic_candidates,
                        ) = semantic_adjudicate_fact(fact_id, current)
                        semantic_scores[fact_id] = round(semantic_score, 6)
                        preserved = semantic_preserved
                        evidence = semantic_evidence if semantic_preserved else ""
                        if not semantic_preserved:
                            adjudicated_fact_ids.append(fact_id)
                            preserved, evidence = adjudicate_fact(
                                fact_id,
                                current,
                                semantic_candidates,
                            )
                            if not preserved and semantic_candidates:
                                preserved, evidence = adjudicate_fact(
                                    fact_id,
                                    current,
                                )
                        normalized_checks.append({{
                            "fact_id": fact_id,
                            "preserved": bool(preserved),
                            "script_evidence": evidence.strip() if isinstance(evidence, str) else "",
                        }})
                        if not preserved:
                            failed_ids.append(fact_id)

                    issues = list(deterministic_issues(current))
                    for fact_id in failed_ids:
                        issues.append(f"missing fact {{fact_id}}: {{fact_by_id[fact_id]['fact']}}")
                    return {{
                        "passed": not issues,
                        "issues": issues,
                        "checks": normalized_checks,
                        "adjudicated_fact_ids": adjudicated_fact_ids,
                        "semantic_scores": semantic_scores,
                        "failed_fact_ids": failed_ids,
                        "contradictions": [],
                        "contradiction_fact_ids": [],
                    }}

                review_prompt = (
                    "NARRATIVE_FACT_REVIEW\n"
                    "Audit SCRIPT against FACT_CHECKLIST, which was extracted from SOURCE and is the source-of-truth checklist. "
                    "For EVERY fact ID return exactly one check. A fact is preserved when SCRIPT expresses the same meaning "
                    "in the target language; wording may differ. "
                    "Never reinterpret surrounding translated words as a renamed character. "
                    "For contradictions, cite a valid fact_id and an exact short script_claim. "
                    "Return ONLY JSON with keys checks and contradictions. Each check has fact_id, preserved, script_evidence. "
                    "Each contradiction has fact_id, script_claim, reason. contradictions must be empty unless SCRIPT "
                    "directly conflicts with a checklist fact.\n"
                    f"EDITORIAL_LESSONS:\n{{lessons_text}}\nFACT_CHECKLIST:\n{{facts_json}}\n"
                    f"SCRIPT:\n{{current}}"
                )
                data = parse_json(generate(review_prompt, 1100), "narrative fact review")
                checks = data.get("checks")
                contradictions = data.get("contradictions")
                if not isinstance(checks, list) or not isinstance(contradictions, list):
                    raise RuntimeError("review checks and contradictions must be arrays")

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
                    if fact_by_id[fact_id]["kind"] == "character" and not CONFIG["review_only"]:
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

                adjudicated_fact_ids = []
                semantic_scores = {{}}
                if failed_ids:
                    confirmed_failed_ids = []
                    for fact_id in failed_ids:
                        (
                            semantic_preserved,
                            semantic_evidence,
                            semantic_score,
                            semantic_candidates,
                        ) = semantic_adjudicate_fact(fact_id, current)
                        semantic_scores[fact_id] = round(semantic_score, 6)
                        adjudicated_fact_ids.append(fact_id)
                        if semantic_preserved:
                            preserved = True
                            evidence = semantic_evidence
                        else:
                            preserved, evidence = adjudicate_fact(
                                fact_id,
                                current,
                                semantic_candidates,
                            )
                        if preserved:
                            for check in normalized_checks:
                                if check["fact_id"] == fact_id:
                                    check["preserved"] = True
                                    check["script_evidence"] = evidence
                                    break
                        else:
                            confirmed_failed_ids.append(fact_id)
                    failed_ids = confirmed_failed_ids

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
                    "adjudicated_fact_ids": adjudicated_fact_ids,
                    "semantic_scores": semantic_scores,
                    "failed_fact_ids": failed_ids,
                    "contradictions": normalized_contradictions,
                    "contradiction_fact_ids": contradiction_ids,
                }}

            review = review_script(script)
            revisions = 0
            seen = {{fingerprint(script)}}
            strategy_history = []

            while (
                not CONFIG["review_only"]
                and not review["passed"]
                and revisions < int(CONFIG["max_revisions"])
            ):
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
                "adjudicated_fact_ids": review["adjudicated_fact_ids"],
                "semantic_scores": review["semantic_scores"],
                "contradiction_fact_ids": review["contradiction_fact_ids"],
                "strategy_history": strategy_history,
                "gpu_name": gpu_name,
                "model": CONFIG["model"],
            }}, ensure_ascii=False, indent=2))
            """
        ).strip() + "\n"
