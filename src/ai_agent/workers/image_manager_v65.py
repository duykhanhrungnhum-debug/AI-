"""AIKA Warm Image Worker v6.5 stability manager.

V6.5 keeps the verified V6 FLUX execution path and isolates recaption from CUDA.
Recent deployed canaries showed that repeatedly handing the Kaggle GPU between
Qwen and FLUX can leave generation stalled at ``generate:1/1``. Qwen therefore
stays on CPU while FLUX remains the only CUDA owner.

The change is intentionally narrow: V6 routing, batching, leases, stage
watchdogs, FLUX model/offload configuration, and result upload are unchanged.
"""
from __future__ import annotations

from .image_manager_v6 import WarmImageWorkerManagerV6


class WarmImageWorkerManagerV65(WarmImageWorkerManagerV6):
    """V6 warm-image manager with CPU-only Qwen recaption."""

    def _worker_source(
        self,
        *,
        base_url: str,
        worker_token: str,
        session_id: str = "test-session",
    ) -> str:
        source = super()._worker_source(
            base_url=base_url,
            worker_token=worker_token,
            session_id=session_id,
        )

        load_old = '''recaptioner = AutoModelForCausalLM.from_pretrained(\n    CONFIG["recaption_model"],\n    torch_dtype=torch.float16,\n    low_cpu_mem_usage=True,\n)\n'''
        load_new = '''recaptioner = AutoModelForCausalLM.from_pretrained(\n    CONFIG["recaption_model"],\n    torch_dtype=torch.float32,\n    low_cpu_mem_usage=True,\n)\n'''
        if load_old not in source:
            raise RuntimeError("V6 Qwen load block changed; CPU-isolation patch is unsafe")
        source = source.replace(load_old, load_new, 1)

        handoff_old = '''def acquire_qwen_gpu():\n    if not hasattr(pipe, "maybe_free_model_hooks"):\n        raise RuntimeError("diffusers pipeline lacks maybe_free_model_hooks")\n    pipe.maybe_free_model_hooks()\n    gc.collect()\n    torch.cuda.empty_cache()\n    recaptioner.to("cuda")\n\n\ndef release_qwen_gpu():\n    recaptioner.to("cpu")\n    gc.collect()\n    torch.cuda.empty_cache()\n'''
        handoff_new = '''def acquire_qwen_gpu():\n    # Qwen stays on CPU. FLUX is the sole CUDA owner for the worker session.\n    return None\n\n\ndef release_qwen_gpu():\n    # Kept as a compatibility no-op so the verified V6 job flow stays unchanged.\n    return None\n'''
        if handoff_old not in source:
            raise RuntimeError("V6 GPU handoff block changed; CPU-isolation patch is unsafe")
        return source.replace(handoff_old, handoff_new, 1)
