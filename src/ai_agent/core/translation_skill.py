from __future__ import annotations

from dataclasses import dataclass
import math
import re

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
VI_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ0-9]+")

# These phrases are legitimate only when the Chinese source is actually talking
# about copyright, filming, policy, requests, etc. They are strong signals of
# model refusal/meta hallucination in dialogue translation.
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
META_SOURCE_ZH = (
    "版权", "著作权", "拍摄", "摄制", "剧组", "影片", "电影", "政策", "请求", "要求",
)

FANTASY_GLOSSARY = {
    "修仙": "tu tiên",
    "修士": "tu sĩ",
    "灵气": "linh khí",
    "靈氣": "linh khí",
    "灵力": "linh lực",
    "靈力": "linh lực",
    "灵根": "linh căn",
    "靈根": "linh căn",
    "炼气": "Luyện Khí",
    "練氣": "Luyện Khí",
    "筑基": "Trúc Cơ",
    "築基": "Trúc Cơ",
    "金丹": "Kim Đan",
    "元婴": "Nguyên Anh",
    "元嬰": "Nguyên Anh",
    "渡劫": "Độ Kiếp",
    "宗门": "tông môn",
    "宗門": "tông môn",
    "师尊": "sư tôn",
    "師尊": "sư tôn",
    "师兄": "sư huynh",
    "師兄": "sư huynh",
    "师姐": "sư tỷ",
    "師姐": "sư tỷ",
    "师弟": "sư đệ",
    "師弟": "sư đệ",
    "师妹": "sư muội",
    "師妹": "sư muội",
    "掌门": "chưởng môn",
    "掌門": "chưởng môn",
    "长老": "trưởng lão",
    "長老": "trưởng lão",
    "道友": "đạo hữu",
    "法宝": "pháp bảo",
    "法寶": "pháp bảo",
    "丹药": "đan dược",
    "丹藥": "đan dược",
    "功法": "công pháp",
    "秘境": "bí cảnh",
    "洞府": "động phủ",
    "魔修": "ma tu",
    "正道": "chính đạo",
    "天道": "thiên đạo",
    "飞升": "phi thăng",
    "飛升": "phi thăng",
    "境界": "cảnh giới",
}

@dataclass(frozen=True)
class TranslationAssessment:
    source_zh: str
    target_vi: str
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.reasons


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def cjk_count(value: str) -> int:
    return len(CJK_RE.findall(value))


def vi_word_count(value: str) -> int:
    return len(VI_WORD_RE.findall(_norm(value)))


def assess_translation_pair(source_zh: str, target_vi: str) -> TranslationAssessment:
    source = _norm(source_zh)
    target = _norm(target_vi)
    reasons: list[str] = []

    if not source:
        reasons.append("empty_source")
    if not target:
        reasons.append("empty_target")
        return TranslationAssessment(source, target, tuple(reasons))
    if CJK_RE.search(target):
        reasons.append("target_contains_cjk")

    src_cjk = cjk_count(source)
    tgt_words = vi_word_count(target)

    # Catch extreme expansion such as 看不懂 -> a long copyright refusal.
    if src_cjk and tgt_words > max(10, math.ceil(src_cjk * 2.6) + 3):
        reasons.append("extreme_expansion")

    # Catch extreme omission. This is deliberately loose to avoid punishing
    # concise natural Vietnamese.
    if src_cjk >= 10 and tgt_words < max(2, math.floor(src_cjk / 6)):
        reasons.append("extreme_omission")

    target_lower = target.casefold()
    source_has_meta = any(term in source for term in META_SOURCE_ZH)
    if not source_has_meta and any(term in target_lower for term in META_HALLUCINATION_VI):
        reasons.append("meta_hallucination")

    for zh, vi in FANTASY_GLOSSARY.items():
        if zh in source and vi.casefold() not in target_lower:
            reasons.append(f"missing_glossary:{zh}")

    return TranslationAssessment(source, target, tuple(dict.fromkeys(reasons)))


# Real regressions observed in Hidden Beyond episode 1 plus domain sanity cases.
TRANSLATION_REGRESSION_CASES = (
    {
        "source": "看不懂",
        "must_any": ("không hiểu", "đọc không hiểu"),
        "forbid": ("bản quyền", "yêu cầu", "đội phim", "quay phim"),
    },
    {
        "source": "这不就是太阳底下修炼吗",
        "must_all": ("tu luyện",),
        "must_any": ("mặt trời", "ánh nắng", "dưới nắng"),
        "forbid": ("bản quyền", "đội phim", "quay phim"),
    },
    {
        "source": "我果然是天选之人",
        "must_any": ("trời chọn", "thiên tuyển"),
        "forbid": ("máu trời", "bản quyền", "đội phim"),
    },
    {"source": "你已经练气三层了", "must_all": ("luyện khí",), "must_any": ("ba", "3")},
    {"source": "师兄小心", "must_all": ("sư huynh",), "must_any": ("cẩn thận", "coi chừng")},
    {"source": "灵气太稀薄了", "must_all": ("linh khí",), "must_any": ("loãng", "mỏng", "ít")},
    {"source": "我们去宗门", "must_all": ("tông môn",)},
    {"source": "这是筑基丹", "must_all": ("trúc cơ",), "must_any": ("đan", "đan dược")},
    {"source": "他突破到金丹境了", "must_all": ("kim đan",), "must_any": ("đột phá", "cảnh")},
    {"source": "我没有灵根", "must_all": ("linh căn",), "must_any": ("không", "chẳng")},
    {"source": "你想飞升吗", "must_all": ("phi thăng",)},
    {"source": "魔修来了", "must_all": ("ma tu",), "must_any": ("đến", "tới")},
    {"source": "这是我的法宝", "must_all": ("pháp bảo",)},
    {"source": "洞府里有秘境", "must_all": ("động phủ", "bí cảnh")},
    {"source": "天道不公", "must_all": ("thiên đạo",), "must_any": ("bất công", "không công bằng")},
)


def regression_case_passes(case: dict, translated: str) -> tuple[bool, tuple[str, ...]]:
    value = _norm(translated).casefold()
    reasons = list(assess_translation_pair(str(case["source"]), translated).reasons)
    for term in case.get("must_all", ()):
        if str(term).casefold() not in value:
            reasons.append(f"missing_required:{term}")
    must_any = tuple(str(x).casefold() for x in case.get("must_any", ()))
    if must_any and not any(x in value for x in must_any):
        reasons.append("missing_any_required")
    for term in case.get("forbid", ()):
        if str(term).casefold() in value:
            reasons.append(f"forbidden:{term}")
    reasons = list(dict.fromkeys(reasons))
    return not reasons, tuple(reasons)
