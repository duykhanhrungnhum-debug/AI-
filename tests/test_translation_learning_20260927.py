from __future__ import annotations

import re


def vietnamese_cps(text: str, seconds: float) -> float:
    visible = len(re.sub(r"\s+", " ", text.strip()))
    return visible / max(0.001, seconds)


def classify_repetition(source: str, target: str) -> str:
    # A repeated Vietnamese phrase is only an error when the source does not
    # carry comparable repetition. This prevents emotional stutters/calls
    # from being "fixed" away.
    target_loop = re.search(r"\b(.{2,24}?)\b(?:[ ,.!?…-]+\1){1,}", target, re.I)
    if not target_loop:
        return "none"
    src_repeat = bool(re.search(r"(.{1,8}?)(?:[，,。！？!?、…-]*\1){1,}", source))
    return "source_supported" if src_repeat else "accidental_loop"


def asr_hallucination_insertions(reference_tokens: list[str], hypothesis_tokens: list[str]) -> int:
    # Lightweight insertion proxy for the CPU regression gate. Runtime ASR
    # benchmark can later report HER beside WER.
    from difflib import SequenceMatcher
    inserted = 0
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(a=reference_tokens, b=hypothesis_tokens).get_opcodes():
        if tag == "insert":
            inserted += j2 - j1
        elif tag == "replace":
            inserted += max(0, (j2 - j1) - (_i2 - _i1))
    return inserted


def test_vietnamese_reading_speed_reference_gate():
    assert vietnamese_cps("Ngươi mau rời khỏi nơi này.", 2.0) <= 17.0
    assert vietnamese_cps("Ngươi lập tức rời khỏi nơi này ngay, đừng quay đầu lại.", 2.0) > 17.0


def test_expressive_repetition_is_not_accidental_loop():
    assert classify_repetition("快快快！", "Mau, mau, mau!") == "source_supported"


def test_target_only_repetition_is_flagged():
    assert classify_repetition("快走！", "Mau đi, mau đi, mau đi!") == "accidental_loop"


def test_asr_insertion_metric_separates_hallucination_signal():
    ref = ["师兄", "小心"]
    hyp = ["师兄", "小心", "欢迎", "订阅", "频道"]
    assert asr_hallucination_insertions(ref, hyp) == 3


def test_blind_xianxia_challenger_not_in_existing_glossary():
    # Deliberately not a glossary assertion. The runtime challenger must infer
    # relationship/register from context rather than memorize a fixed term.
    dialogue = [
        ("师叔，掌教师兄让我来请您。", "junior addresses senior generation"),
        ("他既是你师父的师弟，你怎可直呼其名？", "relationship constrains address"),
        ("弟子知错。", "speaker self-position changes register"),
    ]
    assert len(dialogue) == 3
