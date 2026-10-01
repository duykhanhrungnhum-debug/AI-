"""AIKA Warm Image Worker v6.5.

V6.5 keeps the verified V6 lifecycle/model path and adds an in-pipeline FLUX
heartbeat. Python background threads can be starved while a CUDA pipeline call
holds execution, so the denoising callback reports progress from the inference
path itself. The stage changes on every denoising step, which means the V6
watchdog still catches a genuinely stuck step without killing a healthy render.
"""
from __future__ import annotations

from .image_manager_v6 import WarmImageWorkerManagerV6


class WarmImageWorkerManagerV65(WarmImageWorkerManagerV6):
    """V6 manager with denoise-step heartbeats embedded in the worker source."""

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
        needle = '''                image = pipe(\n                    prompt=prompt,\n                    height=height,\n                    width=width,\n                    guidance_scale=float(CONFIG["guidance"]),\n                    num_inference_steps=int(CONFIG["steps"]),\n                    generator=generator,\n                ).images[0]\n'''
        replacement = '''                def _step_heartbeat(_pipeline, step_index, _timestep, callback_kwargs):\n                    progress_stage = (\n                        stage + f":step{int(step_index) + 1}/{int(CONFIG['steps'])}"\n                    )\n                    set_state(progress_stage, job_id)\n                    signal(progress_stage, job_id)\n                    return callback_kwargs\n\n                image = pipe(\n                    prompt=prompt,\n                    height=height,\n                    width=width,\n                    guidance_scale=float(CONFIG["guidance"]),\n                    num_inference_steps=int(CONFIG["steps"]),\n                    generator=generator,\n                    callback_on_step_end=_step_heartbeat,\n                ).images[0]\n'''
        if needle not in source:
            raise RuntimeError("V6 worker source changed; cannot inject FLUX step heartbeat safely")
        return source.replace(needle, replacement, 1)
