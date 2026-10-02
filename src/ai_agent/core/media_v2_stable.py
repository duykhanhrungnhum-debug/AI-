"""Stable production wrapper for Image V2.

The normal Kaggle status endpoint is not reliable for the current private-token
setup (``kernels.get`` can return 403 even while a submitted kernel runs). This
provider therefore treats the output artifact itself as the source of truth.

Each run injects a unique token into the worker log and final report, so an old
report from the previous kernel version can never be mistaken for the current
run. Polling is bounded and logs are diagnostic only.
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
        """Install one generic semantic review/repair gate before FLUX.

        This intentionally has no species table, entity dictionary, blacklist or
        per-word patch. Qwen recaptions normally, then compares that candidate with
        the original request. It returns either ``OK`` or one corrected prompt.
        A correction is checked once more; if it still does not pass, the job fails
        before FLUX instead of spending GPU on a semantically wrong image.
        """
        old_contract = '''base_contract = (\n    "Preserve exactly the requested subject or species, number of subjects, visual style, setting, framing, "\n    "important attributes, and explicit exclusions. Preserve culturally specific names and untranslated proper "\n    "terms verbatim instead of substituting an item from another culture. For named garments, foods, places, or "\n    "art forms, keep the original name and optionally add a short English gloss. For a culturally specific named "\n    "garment, add its canonical silhouette and construction details when known confidently. In particular, Vietnamese "\n    "'áo dài' must remain 'Vietnamese áo dài' and must be described as a fitted high-collared long-sleeved tunic with "\n    "long front and back panels, high side slits, worn over separate loose full-length trousers; it is not a one-piece "\n    "dress, hanbok, qipao, or cheongsam. Do not generalize a named subject. Do not invent body parts, objects, text, "\n    "logos, or requirements that the user did not request."\n)\n'''
        new_contract = '''def contract_for(user_request):\n    contract = (\n        "Treat the original user request as semantic authority. Preserve every concrete subject/entity/species/breed, "\n        "the requested count, style, setting, framing, important attributes and explicit exclusions. Translate named "\n        "subjects literally and precisely. Never replace a requested subject with a related, similar, generic, culturally "\n        "adjacent or visually convenient substitute. Never invent a new subject. Do not introduce garments, props, text, "\n        "logos or requirements the user did not request. Do not generalize a named subject."\n    )\n    if "áo dài" in user_request.casefold():\n        contract += (\n            " Vietnamese 'áo dài' must remain 'Vietnamese áo dài' and be described as a fitted high-collared "\n            "long-sleeved tunic with long front and back panels, high side slits, worn over separate loose full-length "\n            "trousers; it is not a one-piece dress, hanbok, qipao or cheongsam."\n        )\n    return contract\n\n\ndef _clean_verdict(raw):\n    text = str(raw or "").strip().strip("`").strip()\n    if text.casefold().startswith("json\\n"):\n        text = text[5:].strip()\n    return text\n\n\ndef _review_instruction(user_request, candidate_prompt, *, variant_scope=False, final_check=False):\n    scope = (\n        "This candidate represents one image from a multi-image request. Judge only the subjects and attributes that "\n        "belong to this candidate; never merge subjects from sibling images. "\n        if variant_scope else ""\n    )\n    decision = (\n        "If the candidate is fully faithful, output exactly OK. If anything is still wrong, output exactly FAIL."\n        if final_check else\n        "If the candidate is fully faithful, output exactly OK. If anything is wrong, output FIX: followed by one "\n        "corrected concise English image-generation prompt. Do not output explanation or markdown."\n    )\n    return (\n        "You are AIKA's semantic fidelity checker. Independently compare ORIGINAL USER REQUEST with CANDIDATE PROMPT. "\n        + scope +\n        "Check every concrete subject/entity/species/breed, subject count, requested style and explicit visual attribute. "\n        "A requested subject must stay the same exact subject; a related or visually similar substitute is incorrect. "\n        "Do not trust the candidate's translation when it conflicts with the original-language request. " + decision +\n        "\\nORIGINAL USER REQUEST: " + user_request +\n        "\\nCANDIDATE PROMPT: " + candidate_prompt\n    )\n\n\ndef enforce_semantic_integrity(user_request, candidate_prompt, *, variant_scope=False):\n    verdict = _clean_verdict(render_recaption(\n        tokenizer, recaptioner,\n        _review_instruction(user_request, candidate_prompt, variant_scope=variant_scope, final_check=False),\n    ))\n    normalized = verdict.casefold().rstrip(".! ")\n    if normalized == "ok":\n        return candidate_prompt\n    if not verdict.casefold().startswith("fix:"):\n        raise RuntimeError("semantic integrity verifier returned an invalid verdict before image generation")\n\n    repaired = verdict.split(":", 1)[1].strip().strip('"')\n    if len(repaired) < 12:\n        raise RuntimeError("semantic integrity repair returned an empty/invalid prompt")\n\n    final_verdict = _clean_verdict(render_recaption(\n        tokenizer, recaptioner,\n        _review_instruction(user_request, repaired, variant_scope=variant_scope, final_check=True),\n    ))\n    if final_verdict.casefold().rstrip(".! ") != "ok":\n        raise RuntimeError("semantic integrity guard failed final verification before image generation")\n    return repaired\n'''
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
