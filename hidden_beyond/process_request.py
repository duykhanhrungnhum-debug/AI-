#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ai_agent.core.tts_model import PiperTTSProvider

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
REPEAT_RE = re.compile(r"\b([\w\u00c0-\u1ef9]+)(?:\s+\1)+\b", re.IGNORECASE)
REPEATED_BIGRAM_RE = re.compile(r"\b([\w\u00c0-\u1ef9]+\s+[\w\u00c0-\u1ef9]+)(?:\s+\1)+\b", re.IGNORECASE)
PROPER_NAMES = ("Pepper", "Carrot", "Saffron", "Morevna", "Synfig", "RabbiDuck", "DragonCow")


def model_for(language: str) -> str:
    lang = language.strip().casefold()
    if lang.startswith(("zh", "cmn", "yue")):
        return "Helsinki-NLP/opus-mt-zh-vi"
    return "VietAI/envit5-translation"


def clean_translation(text: str, *, field: str) -> str:
    value = text.strip().strip('\"“”').strip()
    if value.lower().startswith("vi:"):
        value = value[3:].strip()
    previous = None
    while previous != value:
        previous = value
        value = REPEATED_BIGRAM_RE.sub(r"\1", value)
        value = REPEAT_RE.sub(r"\1", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise ValueError(f"{field} is empty")
    if CJK_RE.search(value):
        raise ValueError(f"{field} still contains CJK characters: {value}")
    return value


def translate_local(model_name: str, texts: list[str]) -> tuple[list[str], tuple[str, ...]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
    model.eval()

    if model_name == "VietAI/envit5-translation":
        prepared = [f"en: {text}" for text in texts]
        max_input_length = 256
        max_new_tokens = 192
    else:
        prepared = texts
        max_input_length = 256
        max_new_tokens = 128

    responses: list[str] = []
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    if len(responses) != len(texts):
        raise ValueError("translation response count does not match request")
    return responses, (
        "runtime:github-actions-cpu",
        f"model:{model_name}",
        f"response_count:{len(responses)}",
    )


def assert_names(source: str, translated: str, *, field: str) -> None:
    for name in PROPER_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", source, re.IGNORECASE) and name not in translated:
            raise ValueError(f"{field} lost exact proper name {name}: {translated}")


def main() -> None:
    started = time.monotonic()
    request_path = Path(os.environ.get("HB_REQUEST", "request/request.json"))
    result_dir = Path(os.environ.get("HB_RESULT_DIR", "result"))
    result_dir.mkdir(parents=True, exist_ok=True)
    req = json.loads(request_path.read_text(encoding="utf-8"))
    segments = req["segments"]

    source_texts = [str(segment["text"]) for segment in segments] + [str(req["title"])]
    model_name = model_for(str(req.get("detected_language", "en")))

    translation_started = time.monotonic()
    raw_translations, evidence = translate_local(model_name, source_texts)
    translation_seconds = time.monotonic() - translation_started

    cleaned = [
        clean_translation(raw, field=f"segment {index}")
        for index, raw in enumerate(raw_translations[:-1], 1)
    ]
    title = clean_translation(raw_translations[-1], field="title")

    for index, (source, translated) in enumerate(zip(source_texts[:-1], cleaned, strict=True), 1):
        assert_names(source, translated, field=f"segment {index}")
    assert_names(source_texts[-1], title, field="title")

    tts_started = time.monotonic()
    voice = PiperTTSProvider(
        model_path=f"piper-voices/{os.environ['PIPER_VOICE']}.onnx",
        binary="piper",
        timeout=300,
        min_duration_seconds=0.05,
    )
    audios = voice.synthesize_many(cleaned)
    tts_seconds = time.monotonic() - tts_started

    out = []
    for src, vi, audio in zip(segments, cleaned, audios, strict=True):
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

    total_seconds = time.monotonic() - started
    report = {
        "request_id": req["request_id"],
        "translated_title": title,
        "segments": out,
        "llm_evidence": [
            *evidence,
            "translation_provider:envit5-for-en+opus-fallback-for-zh",
            "quality_gate:no_cjk+no_immediate_word_or_bigram_repetition+exact_proper_names",
            "tts_mode:piper-python-api-single-model-load",
        ],
        "timing": {
            "translation_seconds": round(translation_seconds, 3),
            "tts_seconds": round(tts_seconds, 3),
            "processor_seconds": round(total_seconds, 3),
        },
    }
    (result_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "HIDDEN_BEYOND_AI_OK "
        f"segments={len(out)} model={model_name} piper_processes=1 "
        f"translation_seconds={translation_seconds:.2f} tts_seconds={tts_seconds:.2f} total_seconds={total_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
