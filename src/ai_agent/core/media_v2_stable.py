"""Stable production wrapper for Image V2.

The normal Kaggle status endpoint is not reliable for the current private-token
setup (``kernels.get`` can return 403 even while a submitted kernel runs). This
provider therefore treats the output artifact itself as the source of truth.

Each run injects a unique token into the worker log and final report, so an old
report from the previous kernel version can never be mistaken for the current
run. Polling is bounded to <10 minutes and logs are used only for early failure
diagnostics, never as a required control-plane dependency.
"""
from __future__ import annotations

from hashlib import sha256
import json
import time
from uuid import uuid4

from .media_v2 import KaggleImageV2Provider


class StableKaggleImageV2Provider(KaggleImageV2Provider):
    """Image V2 provider that verifies completion from the current output report."""

    max_wait_seconds: float = 540.0

    def _harden_recaption_source(self, source: str) -> str:
        """Install a generic subject-identity lock before FLUX.

        There is no species table, blacklist, or per-word patch. The already-loaded
        Qwen first extracts the concrete requested subjects into a tiny bilingual
        lock list. If the normal recaption contains every locked English subject,
        it proceeds immediately. Otherwise Qwen gets exactly one text-only repair
        pass and the repaired prompt is validated again. A second mismatch fails
        before FLUX, so semantic drift never wastes an image-generation run.
        """
        old_contract = '''base_contract = (\n    "Preserve exactly the requested subject or species, number of subjects, visual style, setting, framing, "\n    "important attributes, and explicit exclusions. Preserve culturally specific names and untranslated proper "\n    "terms verbatim instead of substituting an item from another culture. For named garments, foods, places, or "\n    "art forms, keep the original name and optionally add a short English gloss. For a culturally specific named "\n    "garment, add its canonical silhouette and construction details when known confidently. In particular, Vietnamese "\n    "'áo dài' must remain 'Vietnamese áo dài' and must be described as a fitted high-collared long-sleeved tunic with "\n    "long front and back panels, high side slits, worn over separate loose full-length trousers; it is not a one-piece "\n    "dress, hanbok, qipao, or cheongsam. Do not generalize a named subject. Do not invent body parts, objects, text, "\n    "logos, or requirements that the user did not request."\n)\n'''
        new_contract = '''def contract_for(user_request):\n    contract = (\n        "Treat the original user request as semantic authority. Preserve exactly every requested concrete subject, "\n        "entity, species, breed, count, visual style, setting, framing, important attribute, and explicit exclusion. "\n        "Translate named subjects literally and precisely; never replace one subject/species/entity with a related, "\n        "similar, generic, culturally adjacent, or more visually convenient substitute. Never invent a new subject. "\n        "For a non-English named subject/species/entity, retain its exact source-language term in parentheses after the "\n        "precise English translation when useful as a fidelity anchor. Preserve culturally specific names and proper "\n        "terms instead of substituting an item from another culture. Do not introduce garments, props, body parts, "\n        "text, logos, or other requirements the user did not ask for. Do not generalize a named subject."\n    )\n    if "áo dài" in user_request.casefold():\n        contract += (\n            " Vietnamese 'áo dài' must remain 'Vietnamese áo dài' and be described as a fitted high-collared "\n            "long-sleeved tunic with long front and back panels, high side slits, worn over separate loose full-length "\n            "trousers; it is not a one-piece dress, hanbok, qipao, or cheongsam."\n        )\n    return contract\n\n\ndef extract_subject_locks(user_request):\n    instruction = (\n        "Extract only the concrete primary visual subjects explicitly requested in USER REQUEST. Return ONLY a strict "\n        "JSON array. Use one object per separate subject with exactly two keys: source and english. source must be the "\n        "shortest exact noun phrase copied verbatim from USER REQUEST. english must be the precise common English name "\n        "of that same subject. Split coordinated subjects into separate objects. Do not include styles, actions, settings, "\n        "camera terms, colors, quantities, or generic words such as image/photo/cartoon. Never replace a species/entity "\n        "with a related or visually similar one. Silently verify each translation before output. If there is no concrete "\n        "visual subject, output []. No markdown or explanation.\\nUSER REQUEST: " + user_request\n    )\n    raw = render_recaption(tokenizer, recaptioner, instruction).strip()\n    if raw.startswith("```"):\n        raw = raw.strip("`").strip()\n        if raw.casefold().startswith("json"):\n            raw = raw[4:].strip()\n    try:\n        data = json.loads(raw)\n    except Exception as exc:\n        raise RuntimeError("subject lock extraction returned invalid JSON") from exc\n    if not isinstance(data, list):\n        raise RuntimeError("subject lock extraction must return a JSON array")\n    locks = []\n    folded_request = user_request.casefold()\n    for entry in data:\n        if not isinstance(entry, dict):\n            raise RuntimeError("subject lock entry is invalid")\n        source_term = str(entry.get("source") or "").strip()\n        english_term = str(entry.get("english") or "").strip()\n        if not source_term or not english_term:\n            raise RuntimeError("subject lock entry is incomplete")\n        if source_term.casefold() not in folded_request:\n            raise RuntimeError("subject lock source is not copied from the user request")\n        if len(english_term) > 80 or "\\n" in english_term:\n            raise RuntimeError("subject lock English label is invalid")\n        locks.append({"source": source_term, "english": english_term})\n    return locks\n\n\ndef prompt_has_subject_locks(prompt, locks):\n    folded = " ".join(prompt.casefold().split())\n    return all(" ".join(lock["english"].casefold().split()) in folded for lock in locks)\n\n\ndef enforce_semantic_integrity(user_request, candidate_prompt, *, variant_scope=False):\n    if not variant_scope:\n        locks = extract_subject_locks(user_request)\n        if not locks or prompt_has_subject_locks(candidate_prompt, locks):\n            return candidate_prompt\n        required = "; ".join(\n            lock["english"] + " (source: " + lock["source"] + ")" for lock in locks\n        )\n        instruction = (\n            "Repair CANDIDATE PROMPT so it follows USER REQUEST exactly. These locked subjects are mandatory and each "\n            "English label must appear verbatim in the repaired prompt: " + required + ". Do not substitute, merge, "\n            "generalize, or remove any locked subject. Preserve the requested style and other valid details. Output only "\n            "one concise English image-generation description; no JSON, labels, or explanation.\\nUSER REQUEST: "\n            + user_request + "\\nCANDIDATE PROMPT: " + candidate_prompt\n        )\n        repaired = render_recaption(tokenizer, recaptioner, instruction).strip().strip('"')\n        if len(repaired) < 12 or not prompt_has_subject_locks(repaired, locks):\n            raise RuntimeError("subject integrity guard failed before image generation")\n        return repaired\n\n    # Multi-image natural-language planning does not expose each original item as a\n    # separate source span. Keep the existing scope-aware semantic repair here so\n    # subjects from sibling variants are never forced into one image.\n    instruction = (\n        "You are AIKA's final semantic-integrity gate for one image from a multi-image request. Independently reread "\n        "the ORIGINAL USER REQUEST; do not trust the candidate when they disagree. Repair only the subjects and "\n        "attributes relevant to this candidate; do not merge subjects from other requested images. Correct any subject, "\n        "species/entity, count, style, or attribute drift. Return exactly one concise English image-generation "\n        "description and nothing else.\\nORIGINAL USER REQUEST: " + user_request + "\\nCANDIDATE PROMPT: "\n        + candidate_prompt\n    )\n    repaired = render_recaption(tokenizer, recaptioner, instruction).strip().strip('"')\n    if len(repaired) < 12:\n        raise RuntimeError("semantic integrity gate returned an empty/invalid description")\n    return repaired\n'''
        if old_contract not in source:
            raise RuntimeError("Image V2 recaption contract changed; stable semantic patch is unsafe")
        source = source.replace(old_contract, new_contract, 1)

        natural_marker = '+ base_contract + " "'
        if source.count(natural_marker) != 2:
            raise RuntimeError("Image V2 recaption call sites changed; stable semantic patch is unsafe")
        source = source.replace(
            natural_marker,
            '+ contract_for(CONFIG["command"]) + " "',
            1,
        )
        source = source.replace(
            natural_marker,
            '+ contract_for(item["command"]) + " "',
            1,
        )

        source = source.replace(
            '"Rewrite the USER REQUEST as one concise, vivid English image-generation description. "',
            '"Rewrite the USER REQUEST as one concise, literal-faithful English image-generation description. "',
            1,
        )

        natural_assignment = '        prompts[item_id] = prompt\n        progress("recaptioned", item_id=item_id, item_index=index + 1, item_total=len(parts))'
        natural_replacement = '''        repaired_prompt = enforce_semantic_integrity(\n            CONFIG["command"], prompt, variant_scope=(len(parts) > 1)\n        )\n        prompts[item_id] = repaired_prompt\n        progress(\n            "integrity_checked",\n            item_id=item_id,\n            item_index=index + 1,\n            item_total=len(parts),\n            changed=(repaired_prompt != prompt),\n        )'''
        if natural_assignment not in source:
            raise RuntimeError("Image V2 natural-batch prompt assignment changed; semantic gate is unsafe")
        source = source.replace(natural_assignment, natural_replacement, 1)

        explicit_assignment = '        prompts[item["id"]] = prompt\n        progress("recaptioned", item_id=item["id"], item_index=index + 1, item_total=len(items))'
        explicit_replacement = '''        repaired_prompt = enforce_semantic_integrity(item["command"], prompt)\n        prompts[item["id"]] = repaired_prompt\n        progress(\n            "integrity_checked",\n            item_id=item["id"],\n            item_index=index + 1,\n            item_total=len(items),\n            changed=(repaired_prompt != prompt),\n        )'''
        if explicit_assignment not in source:
            raise RuntimeError("Image V2 explicit prompt assignment changed; semantic gate is unsafe")
        source = source.replace(explicit_assignment, explicit_replacement, 1)
        return source

    def _instrument_source(self, source: str, run_token: str) -> str:
        future = "from __future__ import annotations\n"
        if future not in source:
            raise RuntimeError("Image V2 source no longer has the expected future import")
        source = source.replace(
            future,
            future
            + f'\nRUN_TOKEN = {run_token!r}\n'
            + 'print("AIKA_IMAGE_RUN " + RUN_TOKEN, flush=True)\n',
            1,
        )
        report_write = 'Path("/kaggle/working/image_v2_report.json").write_text('
        if report_write not in source:
            raise RuntimeError("Image V2 source no longer has the expected report writer")
        return source.replace(
            report_write,
            'report["run_token"] = RUN_TOKEN\n' + report_write,
            1,
        )

    def _run_source(self, source: str):
        run_token = uuid4().hex
        source = self._harden_recaption_source(source)
        source = self._instrument_source(source, run_token)
        kernel_title = (
            "AI Agent Image V2"
            if self.kernel_slug == "ai-agent-image-v2"
            else f"AI Agent Image V2 {sha256(self.kernel_slug.encode('utf-8')).hexdigest()[:8]}"
        )
        submission = self.worker.submit_script(
            slug=self.kernel_slug,
            title=kernel_title,
            source=source,
            enable_internet=True,
            enable_gpu=True,
            is_private=True,
        )

        poll_interval = max(1.0, float(self.poll_interval or 1.0))
        configured_window = max(1, int(self.max_poll_attempts)) * poll_interval
        deadline = time.monotonic() + min(self.max_wait_seconds, configured_window)
        last_logs = ""

        while time.monotonic() < deadline:
            try:
                raw = self.worker.download_output_file(self.kernel_slug, "image_v2_report.json")
                report = json.loads(raw.decode("utf-8"))
                if isinstance(report, dict) and report.get("run_token") == run_token:
                    items = report.get("items")
                    if not isinstance(items, dict):
                        raise ValueError("image V2 report is invalid")
                    gpu_name = str(report.get("gpu_name") or "").strip()
                    if not gpu_name:
                        raise ValueError("image V2 report has no GPU evidence")
                    return submission, report, gpu_name
            except FileNotFoundError:
                pass
            except RuntimeError as exc:
                # Output can legitimately be absent while the current kernel is running.
                # Authentication/permission failures are not transient and should fail fast.
                text = str(exc)
                if "HTTP 404" not in text and "HTTP 409" not in text:
                    raise
            except json.JSONDecodeError:
                pass

            try:
                logs = self.worker.logs(self.kernel_slug)
                if logs:
                    last_logs = logs[-12000:]
                    current = last_logs.split("AIKA_IMAGE_RUN " + run_token, 1)
                    if len(current) == 2:
                        tail = current[1]
                        if "Traceback (most recent call last)" in tail:
                            raise RuntimeError("image V2 worker failed: " + tail[-8000:])
            except RuntimeError as exc:
                # Logs are diagnostic only. A permission error here must not block output polling.
                if "image V2 worker failed:" in str(exc):
                    raise
            except Exception:
                pass

            time.sleep(poll_interval)

        detail = ""
        if last_logs:
            detail = ": " + last_logs[-4000:]
        raise TimeoutError(
            f"image V2 run {run_token[:8]} exceeded {min(self.max_wait_seconds, configured_window):.0f}s{detail}"
        )
