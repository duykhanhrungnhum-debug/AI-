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
REPEAT_RE = re.compile(r"\b([\wÀ-ỹ]+)(?:\s+\1)+\b", re.IGNORECASE)


def parse_lines(raw: str, expected_indexes: list[int]) -> tuple[str | None, dict[int, str]]:
    title: str | None = None
    items: dict[int, str] = {}
    for original in raw.splitlines():
        line = original.strip().strip(chr(96))
        if not line:
            continue
        title_match = re.match(r"^(?:TITLE|TIÊU\s*ĐỀ)\s*(?:\t|:|-)+\s*(.+)$", line, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            continue
        match = re.match(r"^(\d+)\s*(?:\t|:|\.|\)|-)\s*(.+)$", line)
        if not match:
            continue
        idx = int(match.group(1))
        if idx in items:
            raise ValueError(f"duplicate translated segment index {idx}")
        items[idx] = match.group(2).strip()
    if sorted(items) != sorted(expected_indexes):
        print("RAW_AI_RESPONSE_START")
        print(raw)
        print("RAW_AI_RESPONSE_END")
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

    proper_names = ("Pepper", "Carrot", "Saffron", "Morevna", "Synfig", "RabbiDuck", "DragonCow")

    def make_prompt(index: int) -> str:
        current = segments[index]
        before = str(segments[index - 1]["text"]) if index > 0 else "(none)"
        after = str(segments[index + 1]["text"]) if index + 1 < len(segments) else "(none)"
        seconds = max(0.1, float(current["end"]) - float(current["start"]))
        return (
            "Translate CURRENT into concise, natural spoken Vietnamese for dubbing. Return ONLY the Vietnamese translation. "
            "Use BEFORE and AFTER only as context; do not translate them. Never output Chinese/Japanese/Korean characters. "
            "Never repeat a word or filler unnecessarily. Preserve all proper names exactly, including Pepper, Carrot, Saffron, "
            "Morevna, Synfig, RabbiDuck and DragonCow. Pepper and Carrot are character names, not vegetables. "
            "For fantasy dialogue, potion means thuốc phép. Keep the line short enough for the time window. "
            "Silently check Vietnamese fluency, names and repetition before answering.\n"
            f"SECONDS: {seconds:.2f}\nBEFORE: {before}\nCURRENT: {current['text']}\nAFTER: {after}"
        )

    prompts = [make_prompt(i) for i in range(len(segments))]
    prompts.append(
        "Translate this video title naturally into Vietnamese. Return ONLY the title. "
        "Preserve proper names exactly; Pepper & Carrot must remain Pepper & Carrot. "
        "Never output Chinese/Japanese/Korean characters and do not add commentary.\n" + str(req["title"])
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
        max_new_tokens=128,
        temperature=0.0,
    )
    batch = llm.generate_many(prompts)
    responses = batch.responses
    if len(responses) != len(segments) + 1:
        raise ValueError("AI response count does not match request")

    def keep_names(source: str, translated: str, *, field: str) -> None:
        for name in proper_names:
            if name.casefold() in source.casefold() and name not in translated:
                raise ValueError(f"{field} did not preserve proper name {name}: {translated}")

    texts = []
    for src, response in zip(segments, responses[:-1], strict=True):
        vi = validate_translation(response.text, field=f"segment {src['index']}")
        keep_names(str(src["text"]), vi, field=f"segment {src['index']}")
        texts.append(vi)
    title = validate_translation(responses[-1].text, field="title")
    keep_names(str(req["title"]), title, field="title")

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
    print(f"HIDDEN_BEYOND_AI_OK segments={len(out)} kaggle_sessions=1 prompts={len(prompts)} piper_processes=1")


if __name__ == "__main__":
    main()
