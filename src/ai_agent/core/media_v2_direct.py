"""Direct production Image V2 provider.

Production preserves the user's original-language image prompt verbatim and lets
FLUX.2 Klein's built-in Qwen3 text encoder interpret it.  This removes the extra
recaption LLM that previously had authority to substitute concrete subjects.
"""
from __future__ import annotations

from .media_v2_stable import StableKaggleImageV2Provider


class DirectStableKaggleImageV2Provider(StableKaggleImageV2Provider):
    """Stable artifact-polling provider with no external recaption stage."""

    max_wait_seconds: float = 720.0

    def _harden_recaption_source(self, source: str) -> str:
        # Production always submits explicit items.  Keep their original command
        # verbatim as the FLUX prompt, so no intermediate LLM can rename, merge,
        # generalize or invent a subject.
        load_block = '''progress("recaption_loading", model=CONFIG["recaption_model"])
tokenizer = AutoTokenizer.from_pretrained(CONFIG["recaption_model"])
recaptioner = AutoModelForCausalLM.from_pretrained(
    CONFIG["recaption_model"],
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
recaptioner.eval()
'''
        if load_block not in source:
            raise RuntimeError("Image V2 recaption load block changed; direct production patch is unsafe")
        source = source.replace(load_block, 'progress("prompt_mode", mode="direct_original_language")\n', 1)

        explicit_block = '''else:
    items = list(CONFIG["items"])
    for index, item in enumerate(items):
        instruction = (
            "Rewrite the USER REQUEST as one concise, vivid English image-generation description. "
            + base_contract + " "
            "Output only the final English description, with no labels, JSON, explanation, scoring, or commentary.\\n"
            "USER REQUEST: " + item["command"]
        )
        prompt = render_recaption(tokenizer, recaptioner, instruction)
        if len(prompt) < 12:
            raise RuntimeError("recaption returned an empty/invalid description for " + item["id"])
        prompts[item["id"]] = prompt
        progress("recaptioned", item_id=item["id"], item_index=index + 1, item_total=len(items))
'''
        direct_block = '''else:
    items = list(CONFIG["items"])
    for index, item in enumerate(items):
        prompt = str(item["command"]).strip()
        if len(prompt) < 3:
            raise RuntimeError("original image request is empty/invalid for " + item["id"])
        prompts[item["id"]] = prompt
        progress("prompt_preserved", item_id=item["id"], item_index=index + 1, item_total=len(items))
'''
        if explicit_block not in source:
            raise RuntimeError("Image V2 explicit recaption block changed; direct production patch is unsafe")
        source = source.replace(explicit_block, direct_block, 1)

        cleanup = 'del recaptioner, tokenizer\ngc.collect()\ntorch.cuda.empty_cache()'
        if cleanup not in source:
            raise RuntimeError("Image V2 recaption cleanup changed; direct production patch is unsafe")
        source = source.replace(cleanup, 'gc.collect()\ntorch.cuda.empty_cache()', 1)

        # Avoid importing/loading an extra transformers model. Undefined names in
        # the unused natural_batch branch are harmless because production never
        # submits that mode through this provider.
        source = source.replace('    from transformers import AutoModelForCausalLM, AutoTokenizer\n', '', 2)
        return source
