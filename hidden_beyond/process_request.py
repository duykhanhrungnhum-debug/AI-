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
PROPER_NAMES = ("Pepper", "Carrot", "Saffron", "Morevna", "Synfig", "RabbiDuck", "DragonCow")


def model_for(language: str) -> str:
    lang = language.strip().casefold()
    if lang.startswith(("zh", "cmn", "yue")):
        return "Helsinki-NLP/opus-mt-zh-vi"
    return "Helsinki-NLP/opus-mt-en-vi"


def split_names(text: str) -> list[tuple[str, bool]]:
    """Split text into translatable chunks and exact proper-name literals."""
    pattern = "(" + "|".join(rf"\b{re.escape(name)}\b" for name in PROPER_NAMES) + ")"
    pieces = re.split(pattern, text, flags=re.IGNORECASE)
    result: list[tuple[str, bool]] = []
    for piece in pieces:
        if not piece:
            continue
        canonical = next((name for name in PROPER_NAMES if piece.casefold() == name.casefold()), None)
        if canonical is not None:
            result.append((canonical, True))
        else:
            result.append((piece, False))
    return result


def build_translation_units(texts: list[str]) -> tuple[list[str], list[list[tuple[int | None, str | None]]]]:
    """Create one flat batch of only non-name chunks; keep names outside the model."""
    units: list[str] = []
    layouts: list[list[tuple[int | None, str | None]]] = []
    for text in texts:
        layout: list[tuple[int | None, str | None]] = []
        for piece, is_name in split_names(text):
            if is_name:
                layout.append((None, piece))
                continue
            if piece.strip():
                idx = len(units)
                units.append(piece)
                layout.append((idx, None))
            else:
                layout.append((None, piece))
        layouts.append(layout)
    return units, layouts


def rebuild_translation(
    translated_units: list[str],
    layout: list[tuple[int | None, str | None]],
) -> str:
    parts: list[str] = []
    for idx, literal in layout:
        parts.append(literal if idx is None else translated_units[idx])
    value = "".join(parts)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([,.;:!?])(?=[A-Za-zÀ-ỹ])", r"\1 ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_translation(text: str, *, field: str) -> str:
    value = text.strip().strip('\"“”').strip()
    previous = None
    while previous != value:
        previous = value
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
    responses: list[str] = []
    batch_size = 16
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    if len(responses) != len(texts):
        raise ValueError("Marian translation response count does not match request")
    return responses, (
        "runtime:github-actions-cpu",
        f"model:{model_name}",
        f"response_count:{len(responses)}",
    )


def main() -> None:
    started = time.monotonic()
    request_path = Path(os.environ.get("HB_REQUEST", "request/request.json"))
    result_dir = Path(os.environ.get("HB_RESULT_DIR", "result"))
    result_dir.mkdir(parents=True, exist_ok=True)
    req = json.loads(request_path.read_text(encoding="utf-8"))
    segments = req["segments"]

    source_texts = [str(segment["text"]) for segment in segments] + [str(req["title"])]
    translation_units, layouts = build_translation_units(source_texts)

    model_name = model_for(str(req.get("detected_language", "en")))
    translation_started = time.monotonic()
    raw_units, evidence = translate_local(model_name, translation_units)
    translation_seconds = time.monotonic() - translation_started

    rebuilt = [
        rebuild_translation(raw_units, layout)
        for layout in layouts
    ]
    translations = [
        clean_translation(value, field=f"segment {index}")
        for index, value in enumerate(rebuilt[:-1], 1)
    ]
    title = clean_translation(rebuilt[-1], field="title")

    for index, (source, translated) in enumerate(zip(source_texts[:-1], translations, strict=True), 1):
        for name in PROPER_NAMES:
            if re.search(rf"\b{re.escape(name)}\b", source, re.IGNORECASE) and name not in translated:
                raise ValueError(f"segment {index} lost exact proper name {name}: {translated}")
    for name in PROPER_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", source_texts[-1], re.IGNORECASE) and name not in title:
            raise ValueError(f"title lost exact proper name {name}: {title}")

    tts_started = time.monotonic()
    voice = PiperTTSProvider(
        model_path=f"piper-voices/{os.environ['PIPER_VOICE']}.onnx",
        binary="piper",
        timeout=300,
        min_duration_seconds=0.05,
    )
    audios = voice.synthesize_many(translations)
    tts_seconds = time.monotonic() - tts_started

    out = []
    for src, vi, audio in zip(segments, translations, audios, strict=True):
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
            "translation_provider:marian-direct-translation",
            "quality_gate:no_cjk+collapse_immediate_repetition+names_never_sent_to_model",
            "tts_mode:single_piper_batch_process",
        ],
        "timing": {
            "translation_seconds": round(translation_seconds, 3),
            "tts_seconds": round(tts_seconds, 3),
            "processor_seconds": round(total_seconds, 3),
        },
    }
    (result_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "HIDDEN_BEYOND_AI_OK "
        f"segments={len(out)} marian_runtime=github_cpu piper_processes=1 "
        f"translation_seconds={translation_seconds:.2f} tts_seconds={tts_seconds:.2f} total_seconds={total_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
