#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ai_agent.core.kaggle_model import KaggleModelProvider
from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.tts_model import PiperTTSProvider

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
REPEAT_RE = re.compile(r"\b([\wÀ-ỹ]+)(?:\s+\1){2,}\b", re.IGNORECASE)


def parse_lines(raw: str, expected_indexes: list[int]) -> tuple[str, dict[int, str]]:
    title = ""
    items: dict[int, str] = {}
    for original in raw.splitlines():
        line = original.strip()
        if not line or line.startswith("```"):
            continue
        if line.upper().startswith("TITLE\t"):
            title = line.split("\t", 1)[1].strip()
            continue
        match = re.match(r"^(\d+)\t(.+)$", line)
        if not match:
            continue
        idx = int(match.group(1))
        if idx in items:
            raise ValueError(f"duplicate translated segment index {idx}")
        items[idx] = match.group(2).strip()
    if not title:
        raise ValueError("AI response has no TITLE line")
    if sorted(items) != sorted(expected_indexes):
        raise ValueError(f"AI returned indexes {sorted(items)}; expected {sorted(expected_indexes)}")
    return title, items


def collapse_runaway_repetition(text: str) -> str:
    previous = None
    value = re.sub(r"\s+", " ", text).strip()
    while previous != value:
        previous = value
        value = REPEAT_RE.sub(r"\1", value)
    return value


def validate_translation(text: str, *, field: str) -> str:
    value = collapse_runaway_repetition(text)
    if not value:
        raise ValueError(f"{field} is empty")
    if CJK_RE.search(value):
        raise ValueError(f"{field} contains CJK characters: {value}")
    if REPEAT_RE.search(value):
        raise ValueError(f"{field} contains repeated words: {value}")
    return value


def main() -> None:
    request_path = Path(os.environ.get("HB_REQUEST", "request/request.json"))
    result_dir = Path(os.environ.get("HB_RESULT_DIR", "result"))
    result_dir.mkdir(parents=True, exist_ok=True)
    req = json.loads(request_path.read_text(encoding="utf-8"))
    segments = req["segments"]

    compact = [
        {
            "index": int(s["index"]),
            "start": round(float(s["start"]), 2),
            "end": round(float(s["end"]), 2),
            "seconds": round(float(s["end"]) - float(s["start"]), 2),
            "text": str(s["text"]),
        }
        for s in segments
    ]
    prompt = (
        "You are the Vietnamese dubbing editor for Hidden Beyond. "
        "Translate the ENTIRE English dialogue below with full episode context, not line-by-line in isolation. "
        "Return plain tab-separated lines only, no JSON, no markdown and no commentary. "
        "First line must be TITLE<TAB>Vietnamese title. Then one line per segment as INDEX<TAB>Vietnamese dialogue. "
        "Rules: output natural spoken Vietnamese only; never output Chinese/Japanese/Korean characters; "
        "do not repeat a word or filler unnecessarily; preserve meaning; keep each line concise enough for its seconds value; "
        "preserve proper names exactly: Pepper, Carrot, Saffron, Morevna, Synfig, RabbiDuck, DragonCow; "
        "translate 'Pepper & Carrot' as 'Pepper & Carrot', not as vegetables; "
        "keep every input index exactly once and in order. "
        "Silently review the full translation for consistency, names, Vietnamese-only output and repetition before returning final lines.\n"
        f"TITLE: {req['title']}\n"
        "SEGMENTS_JSON:\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    )

    worker = KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        submission_retry_attempts=3,
        submission_retry_delay_seconds=10.0,
    )
    llm = KaggleModelProvider(
        worker=worker,
        model="Qwen/Qwen2.5-3B-Instruct",
        kernel_slug="hidden-beyond-on-demand",
        poll_interval=5,
        max_poll_attempts=360,
        max_new_tokens=1400,
        temperature=0.0,
    )
    response = llm.generate(prompt)
    expected = [int(s["index"]) for s in segments]
    raw_title, raw_by_index = parse_lines(response.text, expected)
    title = validate_translation(raw_title, field="title")
    by_index = {
        idx: validate_translation(text, field=f"segment {idx}")
        for idx, text in raw_by_index.items()
    }

    texts = [by_index[int(s["index"])] for s in segments]
    voice = PiperTTSProvider(
        model_path=f"piper-voices/{os.environ['PIPER_VOICE']}.onnx",
        binary="piper",
        timeout=300,
        min_duration_seconds=0.05,
    )
    audios = voice.synthesize_many(texts)

    out = []
    for src, vi, audio in zip(segments, texts, audios, strict=True):
        name = f"segment-{int(src['index']):05d}.wav"
        (result_dir / name).write_bytes(audio.data)
        out.append({
            "index": src["index"],
            "start": src["start"],
            "end": src["end"],
            "translation": vi,
            "audio_file": name,
            "audio_evidence": audio.evidence,
        })

    report = {
        "request_id": req["request_id"],
        "translated_title": title,
        "segments": out,
        "llm_evidence": [
            f"provider:{response.provider}",
            f"model:{response.model}",
            "translation_mode:single_context_request",
            "quality_gate:vietnamese_only+no_runaway_repetition+name_preservation_prompt",
            "tts_mode:single_piper_batch_process",
        ],
    }
    (result_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"HIDDEN_BEYOND_AI_OK segments={len(out)} translation_calls=1 piper_processes=1")


if __name__ == "__main__":
    main()
