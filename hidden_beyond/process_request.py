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


def parse_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("AI response has no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("AI response must be a JSON object")
    return obj


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
        "Return one JSON object only, no markdown and no commentary. "
        "Schema: {\"title\":\"Vietnamese title\",\"segments\":[{\"index\":1,\"vi\":\"...\"}]}. "
        "Rules: output natural spoken Vietnamese only; never output Chinese/Japanese/Korean characters; "
        "do not repeat a word or filler unnecessarily; preserve meaning; keep each line concise enough for its seconds value; "
        "preserve proper names exactly: Pepper, Carrot, Saffron, Morevna, Synfig, RabbiDuck, DragonCow; "
        "translate 'Pepper & Carrot' as 'Pepper & Carrot', not as vegetables; "
        "keep every input index exactly once and in order. "
        "Silently review the full translation for consistency, names, Vietnamese-only output and repetition before returning final JSON.\n"
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
        max_new_tokens=1800,
        temperature=0.0,
    )
    response = llm.generate(prompt)
    data = parse_object(response.text)
    title = validate_translation(str(data.get("title", "")), field="title")
    translated = data.get("segments")
    if not isinstance(translated, list) or len(translated) != len(segments):
        got = len(translated) if isinstance(translated, list) else "invalid"
        raise ValueError(f"AI returned {got} segments; expected {len(segments)}")

    by_index: dict[int, str] = {}
    for item in translated:
        if not isinstance(item, dict):
            raise ValueError("translation segment must be an object")
        idx = int(item.get("index"))
        if idx in by_index:
            raise ValueError(f"duplicate translated segment index {idx}")
        by_index[idx] = validate_translation(str(item.get("vi", "")), field=f"segment {idx}")

    expected = [int(s["index"]) for s in segments]
    if sorted(by_index) != sorted(expected):
        raise ValueError("translated segment indexes do not match request")

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
