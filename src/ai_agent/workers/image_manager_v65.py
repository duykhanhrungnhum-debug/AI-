"""AIKA Warm Image Worker v6.5 stability manager.

V6.5 isolates recaption from CUDA and selects the simplest safe FLUX execution
mode from the actual Kaggle GPU. Qwen stays on CPU. FLUX runs fully on CUDA when
the device has enough VRAM for the ~13 GB Klein 4B footprint; smaller devices
retain V6 model CPU offload.

The worker also reports GPU/memory/mode in the normal generate heartbeat so a
future hardware-specific failure is diagnosable without Kaggle kernels.get.
"""
from __future__ import annotations

from .image_manager_v6 import WarmImageWorkerManagerV6


class WarmImageWorkerManagerV65(WarmImageWorkerManagerV6):
    """V6 manager with CPU Qwen and VRAM-aware FLUX placement."""

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

        pipe_old = '''pipe = Flux2KleinPipeline.from_pretrained(\n    CONFIG["image_model"],\n    torch_dtype=torch.float16,\n)\npipe.enable_model_cpu_offload()\n'''
        pipe_new = '''pipe = Flux2KleinPipeline.from_pretrained(\n    CONFIG["image_model"],\n    torch_dtype=torch.float16,\n)\ngpu_name = torch.cuda.get_device_name(0)\ngpu_total_bytes = int(torch.cuda.get_device_properties(0).total_memory)\n# BFL documents Klein 4B at about 13 GB VRAM. Require extra headroom before\n# keeping the full pipeline on CUDA; otherwise retain the verified offload path.\nif gpu_total_bytes >= 15 * 1024 ** 3:\n    pipe.to("cuda")\n    flux_execution = "cuda"\nelse:\n    pipe.enable_model_cpu_offload()\n    flux_execution = "model_cpu_offload"\n'''
        if pipe_old not in source:
            raise RuntimeError("V6 FLUX load block changed; placement patch is unsafe")
        source = source.replace(pipe_old, pipe_new, 1)

        handoff_old = '''def acquire_qwen_gpu():\n    if not hasattr(pipe, "maybe_free_model_hooks"):\n        raise RuntimeError("diffusers pipeline lacks maybe_free_model_hooks")\n    pipe.maybe_free_model_hooks()\n    gc.collect()\n    torch.cuda.empty_cache()\n    recaptioner.to("cuda")\n\n\ndef release_qwen_gpu():\n    recaptioner.to("cpu")\n    gc.collect()\n    torch.cuda.empty_cache()\n'''
        handoff_new = '''def acquire_qwen_gpu():\n    # Qwen stays on CPU. FLUX is the sole CUDA owner for the worker session.\n    return None\n\n\ndef release_qwen_gpu():\n    # Compatibility no-op: keep the verified V6 job flow unchanged.\n    return None\n'''
        if handoff_old not in source:
            raise RuntimeError("V6 GPU handoff block changed; CPU-isolation patch is unsafe")
        source = source.replace(handoff_old, handoff_new, 1)

        stage_old = '''                stage = f"busy:generate:{index + 1}/{total}"\n                set_state(stage, job_id)\n                signal(stage, job_id)\n'''
        stage_new = '''                free_bytes, total_bytes = torch.cuda.mem_get_info()\n                gpu_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", gpu_name).strip("_") or "unknown"\n                stage = (\n                    f"busy:generate:{index + 1}/{total}"\n                    f":gpu={gpu_slug}:mode={flux_execution}"\n                    f":free_mb={int(free_bytes // (1024 ** 2))}"\n                    f":total_mb={int(total_bytes // (1024 ** 2))}"\n                )\n                set_state(stage, job_id)\n                signal(stage, job_id)\n'''
        if stage_old not in source:
            raise RuntimeError("V6 generate stage changed; GPU preflight patch is unsafe")
        return source.replace(stage_old, stage_new, 1)
