from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Iterable, Mapping, Sequence

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
VI_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ0-9]+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

NEGATION_ZH = ("不","没","沒有","没有","未","無","无","别","別","莫")
NEGATION_VI = ("không","chẳng","chưa","đừng","khỏi","không có","chớ")
QUESTION_ZH = ("吗","嗎","呢","？","?")
QUESTION_VI = ("không","à","ư","sao","gì","nào","chứ")
META_HALLUCINATION_VI = (
    "không thể thực hiện yêu cầu",
    "không thể đáp ứng yêu cầu",
    "vi phạm bản quyền",
    "vấn đề bản quyền",
    "chính sách nội dung",
    "đội làm phim",
    "đội phim",
    "đoàn phim",
    "quay phim",
    "quá trình quay",
)
META_SOURCE_ZH = ("版权","著作权","拍摄","摄制","剧组","影片","电影","政策","请求","要求")

PROFESSIONAL_ZH_VI_PHRASES = {
    "自寻死路": "tự tìm đường chết",
    "别给脸不要脸": "đừng có không biết điều",
    "一成": "một thành",
    "筑基丹": "đan dược Trúc Cơ",
    "不要命": "liều mạng",
    "竟敢": "lại dám",
    "你还知道回来": "ngươi còn biết đường về",
    "不是你想的那样": "không phải như ngươi nghĩ",
    "你不帮就算了": "ngươi không giúp thì thôi",
    "天道不公": "Thiên đạo bất công",
    "筑基之前": "Trước khi Trúc Cơ",
    "开辟丹田": "khai mở đan điền",
}


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


@dataclass(frozen=True)
class TranslationIssue:
    code: str
    severity: Severity
    detail: str = ""


@dataclass(frozen=True)
class TranslationBrief:
    source_language: str = "Chinese"
    target_language: str = "Vietnamese"
    domain: str = "general"
    audience: str = "native Vietnamese audience"
    register: str = "natural spoken Vietnamese"
    style: str = "faithful, fluent, concise"
    glossary: Mapping[str, str] = field(default_factory=dict)
    forbidden_additions: tuple[str, ...] = ()
    preserve_numbers: bool = True
    preserve_intent: bool = True
    no_explanation: bool = True


@dataclass(frozen=True)
class ProfessionalAssessment:
    source: str
    target: str
    issues: tuple[TranslationIssue, ...]

    @property
    def publishable(self) -> bool:
        return not any(i.severity in (Severity.CRITICAL, Severity.MAJOR) for i in self.issues)

    @property
    def critical_count(self) -> int:
        return sum(i.severity == Severity.CRITICAL for i in self.issues)

    @property
    def major_count(self) -> int:
        return sum(i.severity == Severity.MAJOR for i in self.issues)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _word_count(value: str) -> int:
    return len(VI_WORD_RE.findall(_norm(value)))


def _cjk_count(value: str) -> int:
    return len(CJK_RE.findall(value))


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    low = haystack.casefold()
    return any(str(x).casefold() in low for x in needles)


def _has_vi_question_marker(value: str) -> bool:
    low = _norm(value).casefold()
    if "?" in low:
        return True
    words = set(re.findall(r"[A-Za-zÀ-ỹ]+", low))
    return any(marker in words for marker in QUESTION_VI)


def build_professional_prompt(
    source_text: str,
    brief: TranslationBrief,
    *,
    background: str = "",
    previous_target: Sequence[str] = (),
) -> str:
    """Build a translation brief modeled on professional human workflow.

    The prompt separates background from target text, locks terminology,
    explicitly prioritizes fidelity over fluency, and makes omissions/additions
    unacceptable. It is compatible with Hy-MT2's documented context,
    terminology and style instruction patterns.
    """
    source = _norm(source_text)
    if not source:
        raise ValueError("source_text is required")

    parts: list[str] = []
    merged_glossary = dict(PROFESSIONAL_ZH_VI_PHRASES)
    merged_glossary.update(brief.glossary)
    glossary = [(k, v) for k, v in merged_glossary.items() if k in source]
    if glossary:
        parts.append("参考下面的固定术语翻译：")
        parts.extend(f"{src} 翻译成 {dst}" for src, dst in glossary)

    if background:
        parts.extend(("〖背景信息〗", _norm(background)))

    if previous_target:
        recent = " / ".join(_norm(x) for x in previous_target[-3:] if _norm(x))
        if recent:
            parts.extend(("〖前文越南语译文，仅用于保持人物称呼和语气一致〗", recent))

    constraints = [
        "忠实传达原意，不得添加原文没有的信息，不得遗漏关键含义。",
        "先保证准确，再保证越南语自然；不得为了顺口改变事实、人物关系、否定、数字或语气。",
        f"译文风格必须符合：{brief.style}；语域：{brief.register}；领域：{brief.domain}。",
        "人物称呼、专有名词、境界、功法和术语在全文中保持一致。",
        "疑问句、否定句、命令、讽刺、情绪强度必须保留。",
    ]
    domain_hint = f"{brief.domain} {brief.register} {brief.style}".casefold()
    if any(x in domain_hint for x in ("xianxia", "cultivation", "tu tiên", "tiên hiệp")):
        constraints.extend((
            "修仙/仙侠古风对白默认保持古风人物称谓：ta/ngươi/nàng/hắn/bổn tọa；除非背景明确是现代语境，不得擅自改成 tôi/anh/bạn。",
            "越南语对白应简洁、像影视台词；原文简短时不要改写成解释性长句。",
            "辱骂、挑衅、责备等语气强度要用自然越南语保留，不可弱化成中性表达。",
        ))
    if brief.no_explanation:
        constraints.append("只输出译文，不要解释、注释、免责声明或元话语。")
    if brief.preserve_numbers:
        constraints.append("数字、数量、等级、时间和单位不得丢失或擅自修改。")

    parts.extend(("〖翻译要求〗", *[f"{i+1}. {x}" for i, x in enumerate(constraints)]))
    parts.extend((
        "〖待翻译文本〗",
        source,
        f"请将〖待翻译文本〗准确翻译为{brief.target_language}。",
    ))
    return "\n".join(parts)


def assess_professional_translation(
    source_text: str,
    target_text: str,
    brief: TranslationBrief,
) -> ProfessionalAssessment:
    source = _norm(source_text)
    target = _norm(target_text)
    issues: list[TranslationIssue] = []

    if not source:
        issues.append(TranslationIssue("empty_source", Severity.CRITICAL))
    if not target:
        issues.append(TranslationIssue("empty_target", Severity.CRITICAL))
        return ProfessionalAssessment(source, target, tuple(issues))

    if CJK_RE.search(target):
        issues.append(TranslationIssue("target_contains_source_script", Severity.MAJOR))

    src_cjk = _cjk_count(source)
    tgt_words = _word_count(target)
    if src_cjk and tgt_words > max(12, math.ceil(src_cjk * 2.8) + 4):
        issues.append(TranslationIssue("extreme_expansion", Severity.MAJOR))
    if src_cjk >= 10 and tgt_words < max(2, math.floor(src_cjk / 6)):
        issues.append(TranslationIssue("extreme_omission", Severity.MAJOR))

    low = target.casefold()
    source_has_meta = any(term in source for term in META_SOURCE_ZH)
    if not source_has_meta and any(term in low for term in META_HALLUCINATION_VI):
        issues.append(TranslationIssue("meta_hallucination", Severity.CRITICAL))

    if brief.preserve_numbers:
        src_numbers = NUMBER_RE.findall(source)
        for number in src_numbers:
            if number not in target:
                issues.append(TranslationIssue("lost_number", Severity.CRITICAL, number))

    for src, dst in brief.glossary.items():
        if src in source and dst.casefold() not in low:
            issues.append(TranslationIssue("terminology_violation", Severity.MAJOR, f"{src}->{dst}"))

    for phrase in brief.forbidden_additions:
        if phrase.casefold() in low:
            issues.append(TranslationIssue("forbidden_addition", Severity.CRITICAL, phrase))

    if brief.preserve_intent:
        source_negative = any(x in source for x in NEGATION_ZH)
        target_negative = _contains_any(target, NEGATION_VI)
        if source_negative and not target_negative:
            issues.append(TranslationIssue("negation_lost", Severity.CRITICAL))
        source_question = any(x in source for x in QUESTION_ZH)
        target_question = _has_vi_question_marker(target)
        if source_question and not target_question:
            issues.append(TranslationIssue("question_intent_lost", Severity.MAJOR))

    # Duplicate model boilerplate is almost always an MT failure in dialogue.
    if re.search(r"\b(.{8,40})\b(?:\s+\1){1,}", target, re.IGNORECASE):
        issues.append(TranslationIssue("repetition", Severity.MAJOR))

    dedup: dict[tuple[str, str], TranslationIssue] = {}
    for issue in issues:
        dedup[(issue.code, issue.detail)] = issue
    return ProfessionalAssessment(source, target, tuple(dedup.values()))


@dataclass
class TranslationMemory:
    """Document-level memory for terminology and already approved phrasing."""

    terms: dict[str, str] = field(default_factory=dict)
    approved_pairs: list[tuple[str, str]] = field(default_factory=list)
    max_pairs: int = 2000

    def remember_term(self, source: str, target: str) -> None:
        source = _norm(source)
        target = _norm(target)
        if source and target:
            self.terms[source] = target

    def remember_pair(self, source: str, target: str) -> None:
        source = _norm(source)
        target = _norm(target)
        if not source or not target:
            return
        self.approved_pairs.append((source, target))
        if len(self.approved_pairs) > self.max_pairs:
            self.approved_pairs = self.approved_pairs[-self.max_pairs:]

    def glossary_for(self, source_text: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
        merged = dict(base or {})
        merged.update(self.terms)
        return {src: dst for src, dst in merged.items() if src in source_text}


def select_repair_targets(
    rows: Sequence[tuple[str, str]],
    brief: TranslationBrief,
) -> list[int]:
    """Return only rows that a professional pipeline must retranslate/review."""
    out: list[int] = []
    for i, (source, target) in enumerate(rows):
        assessment = assess_professional_translation(source, target, brief)
        if not assessment.publishable:
            out.append(i)
    return out
