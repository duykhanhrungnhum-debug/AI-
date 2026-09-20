#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import textwrap
import time
from pathlib import Path

from ai_agent.core.kaggle_worker import KaggleGpuWorker
from ai_agent.core.tts_model import PiperTTSProvider

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
REPEAT_RE = re.compile(r"\b([\w\u00c0-\u1ef9]+)(?:\s+\1)+\b", re.IGNORECASE)
PROPER_NAMES = ("Pepper", "Carrot", "Saffron", "Morevna", "Synfig", "RabbiDuck", "DragonCow")


def model_for(language: str) -> str:
    lang = language.strip().casefold()
    if lang.startswith(("zh", "cmn", "yue")):
        return "Helsinki-NLP/opus-mt-zh-vi"
    return "Helsinki-NLP/opus-mt-en-vi"


def protect_names(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    protected = text
    used: list[tuple[str, str]] = []
    for index, name in enumerate(PROPER_NAMES):
        if re.search(rf"\b{re.escape(name)}\b", protected, re.IGNORECASE):
            marker = f"ZXQNAME{index}QXZ"
            protected = re.sub(rf"\b{re.escape(name)}\b", marker, protected, flags=re.IGNORECASE)
            used.append((marker, name))
    return protected, tuple(used)


def restore_names(text: str, names: tuple[tuple[str, str], ...]) -> str:
    value = text
    for marker, name in names:
        flexible = r"\s*".join(re.escape(ch) for ch in marker)
        value = re.sub(flexible, name, value, flags=re.IGNORECASE)
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


def translate_on_kaggle(worker: KaggleGpuWorker, model: str, texts: list[str]) -> tuple[list[str], tuple[str, ...]]:
    config = {"model": model, "texts": texts}
    config_json = json.dumps(config, ensure_ascii=False)
    source = textwrap.dedent(
        f"""
        from __future__ import annotations
        import json
        import subprocess
        import sys
        from pathlib import Path

        CONFIG = json.loads({config_json!r})
        try:
            import torch
            import sentencepiece
            import sacremoses
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet",
                "transformers<5", "sentencepiece", "sacremoses",
            ])
            import torch
            import sentencepiece
            import sacremoses
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
        kwargs = {{"device_map": "auto"}} if torch.cuda.is_available() else {{}}
        model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model"], **kwargs)
        model.eval()

        responses = []
        batch_size = 16
        for start in range(0, len(CONFIG["texts"]), batch_size):
            batch = CONFIG["texts"][start:start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            if not kwargs:
                inputs = {{key: value.to(device) for key, value in inputs.items()}}
                model = model.to(device)
            else:
                inputs = {{key: value.to(model.device) for key, value in inputs.items()}}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=4,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

        Path("/kaggle/working/translations.json").write_text(
            json.dumps({{
                "model": CONFIG["model"],
                "gpu_name": gpu_name,
                "responses": responses,
            }}, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        print("HIDDEN_BEYOND_MARIAN_OK", len(responses), CONFIG["model"], gpu_name)
        """
    ).strip() + "\n"

    submission = worker.submit_script(
        slug="hidden-beyond-marian",
        title="Hidden Beyond Marian Translation",
        source=source,
        enable_internet=True,
        is_private=True,
    )
    for _ in range(180):
        status = worker.status("hidden-beyond-marian")
        if status.terminal:
            if not status.successful:
                logs = worker.logs("hidden-beyond-marian")
                raise RuntimeError(f"Marian Kaggle worker failed: {status.status} {status.failure_message}\n{logs[-5000:]}")
            break
        time.sleep(5)
    else:
        raise TimeoutError("Timed out waiting for Marian Kaggle worker")

    raw = worker.download_output_file("hidden-beyond-marian", "translations.json")
    report = json.loads(raw.decode("utf-8"))
    responses = report.get("responses")
    if report.get("model") != model or not isinstance(responses, list) or len(responses) != len(texts):
        raise ValueError("Marian translation report does not match request")
    gpu_name = str(report.get("gpu_name", "")).strip()
    if not gpu_name:
        raise ValueError("Marian translation report lacks runtime evidence")
    return [str(item) for item in responses], (
        f"kaggle_kernel:{submission.ref}",
        f"gpu:{gpu_name}",
        f"model:{model}",
        f"response_count:{len(responses)}",
    )


def main() -> None:
    started = time.monotonic()
    request_path = Path(os.environ.get("HB_REQUEST", "request/request.json"))
    result_dir = Path(os.environ.get("HB_RESULT_DIR", "result"))
    result_dir.mkdir(parents=True, exist_ok=True)
    req = json.loads(request_path.read_text(encoding="utf-8"))
    segments = req["segments"]

    protected_texts: list[str] = []
    restore_maps: list[tuple[tuple[str, str], ...]] = []
    for segment in segments:
        protected, names = protect_names(str(segment["text"]))
        protected_texts.append(protected)
        restore_maps.append(names)
    protected_title, title_names = protect_names(str(req["title"]))
    protected_texts.append(protected_title)
    restore_maps.append(title_names)

    worker = KaggleGpuWorker(
        api_token=os.environ["KAGGLE_API_TOKEN"],
        username=os.environ["KAGGLE_USERNAME"],
        submission_retry_attempts=3,
        submission_retry_delay_seconds=10.0,
    )
    model = model_for(str(req.get("detected_language", "en")))
    translation_started = time.monotonic()
    raw_translations, evidence = translate_on_kaggle(worker, model, protected_texts)
    translation_seconds = time.monotonic() - translation_started

    translations: list[str] = []
    for index, (raw, names) in enumerate(zip(raw_translations[:-1], restore_maps[:-1], strict=True), 1):
        restored = restore_names(raw, names)
        value = clean_translation(restored, field=f"segment {index}")
        for _, name in names:
            if name not in value:
                raise ValueError(f"segment {index} lost protected proper name {name}: {value}")
        translations.append(value)
    title = clean_translation(restore_names(raw_translations[-1], restore_maps[-1]), field="title")
    for _, name in title_names:
        if name not in title:
            raise ValueError(f"title lost protected proper name {name}: {title}")

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
            "quality_gate:no_cjk+collapse_immediate_repetition+protected_names",
            "tts_mode:single_piper_batch_process",
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
        f"segments={len(out)} marian_sessions=1 piper_processes=1 "
        f"translation_seconds={translation_seconds:.2f} tts_seconds={tts_seconds:.2f} total_seconds={total_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
