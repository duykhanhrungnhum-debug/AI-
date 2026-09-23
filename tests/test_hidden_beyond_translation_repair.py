from __future__ import annotations

import ast
from pathlib import Path


WORKER = Path("hidden_beyond/longform_audio_worker_v3.py")


def _extract_function(name: str):
    source = WORKER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict[str, object] = {}
    exec(compile(module, str(WORKER), "exec"), ns)
    return ns[name]


def test_structural_repair_drops_poisoned_previous_output_and_context():
    policy = _extract_function("repair_prompt_policy")
    for reason in (
        "repair segment 1630 has repetition loop",
        "repair segment 224 still contains CJK",
        "repair segment 5 empty",
    ):
        result = policy(reason)
        assert result["structural"] is True
        assert result["include_previous"] is False
        assert result["include_context"] is False


def test_soft_review_can_keep_context():
    policy = _extract_function("repair_prompt_policy")
    result = policy("")
    assert result["structural"] is False
    assert result["include_previous"] is True
    assert result["include_context"] is True


def test_worker_has_bounded_source_only_structural_rescue():
    source = WORKER.read_text(encoding="utf-8")
    assert "Structural rescue pass:" in source
    assert "rescue_prompt_for" in source
    assert "no_repeat_ngram_size=3" in source
    assert "repetition_penalty=1.20" in source
    assert "max_new_tokens=48" in source
    assert "structural_rescue_segments" in source
    assert "after bounded structural rescue" in source

def test_worker_has_translation_checkpoint_resume_contract():
    source = WORKER.read_text(encoding="utf-8")
    assert "translation_segment_signature" in source
    assert "/translation-checkpoint-get" in source
    assert "/translation-checkpoint-save" in source
    assert "Resuming translation checkpoint" in source
    assert "cursor=resume_cursor" in source
    assert "cursor-last_checkpoint_cursor>=100" in source
    assert "\"translation_complete\"" in source



def test_worker_splits_gpu_translation_and_cpu_tts():
    source = WORKER.read_text(encoding="utf-8")
    assert 'WORKER_PHASE=str(JOB.get("phase") or "full")' in source
    assert 'WORKER_PHASE=="translation_only"' in source
    assert 'WORKER_PHASE=="tts_only"' in source
    assert '"segments_data"' in source
    assert '"translation_complete"' in source
    assert '"compute_stage":"cpu_tts_mix_upload"' in source
    assert 'enable_gpu' not in source  # worker itself never decides Kaggle accelerator


def test_completed_checkpoint_reuses_same_video_same_segment_count():
    source = WORKER.read_text(encoding="utf-8")
    assert "HB_CHECKPOINT_COMPATIBLE same_video_same_count" in source
    assert 'stored_phase=="translation_complete"' in source
    assert 'stored_cursor==int(segment_count)' in source
    assert 'payload["cursor"]=stored_cursor' in source
    assert 'HB_CHECKPOINT_RESUME cursor' in source
    assert 'len(translated)==int(segment_count)' in source
    assert 'all(str(i) in translated for i in range(1,int(segment_count)+1))' in source


def test_worker_persists_and_reuses_asr_checkpoint():
    source = WORKER.read_text(encoding="utf-8")
    assert '"asr_complete"' in source
    assert '"ASR checkpoint saved:' in source
    assert '"ASR checkpoint reused:' in source
    assert 'checkpoint_segments=list(pre_payload.get("segments_data") or [])' in source
    assert 'if not pre_segments:' in source
    assert 'if not pre_translation_complete:' in source
    assert '"transcript_meta"' in source
    assert 'candidate=clean(s.get("vi","")) or translated_cache.get(s["index"],"")' in source


def test_asr_checkpoint_cursor_does_not_become_translation_cursor():
    source = WORKER.read_text(encoding="utf-8")
    assert 'payload["cursor"]=stored_cursor if stored_phase in {"translating","translation_complete"} else 0' in source


def test_worker_prefers_cpu_acquired_captions():
    source = WORKER.read_text(encoding="utf-8")
    assert 'mounted.parent.glob("source-caption*.json3")' in source
    assert '"CPU-acquired captions ready:' in source
    assert '"CPU source captions reused:' in source
    assert 'transcript_source="cpu_source_caption_json3"' in source
    assert 'whisper_name="skipped-cpu-caption"' in source


def test_translation_batch_prefers_gpu_throughput_without_quality_change():
    source = WORKER.read_text(encoding="utf-8")
    assert "batch_size=12" in source
    assert "if batch_size<=4:" in source
    assert "batch_size=max(4,batch_size//2)" in source
    assert "do_sample=False" in source
    assert "cursor-last_checkpoint_cursor>=100" in source


def test_overlong_dialogue_cues_are_split_not_fatal():
    source = WORKER.read_text(encoding="utf-8")
    assert "def enforce_max_cue_duration(" in source
    assert "Normalized {overlong_before} overlong dialogue cues before checkpointing" in source
    assert "segments=enforce_max_cue_duration(segments,max_seconds=6.0)" in source
    assert "dialogue segmentation has {too_long} cues longer than 6.2s" in source
