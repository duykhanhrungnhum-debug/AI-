from ai_agent.core.translation_skill import (
    TRANSLATION_REGRESSION_CASES,
    assess_translation_pair,
    regression_case_passes,
)


def test_rejects_real_episode_1_copyright_hallucination():
    result=assess_translation_pair(
        "看不懂",
        "Tôi không thể thực hiện yêu cầu này vì nó vi phạm quyền tác giả.",
    )
    assert not result.ok
    assert "meta_hallucination" in result.reasons
    assert "extreme_expansion" in result.reasons


def test_accepts_concise_correct_translation():
    result=assess_translation_pair("看不懂","Không hiểu.")
    assert result.ok


def test_rejects_missing_fantasy_glossary():
    result=assess_translation_pair("我没有灵根","Tôi không có căn cơ.")
    assert not result.ok
    assert "missing_glossary:灵根" in result.reasons


def test_accepts_domain_translation():
    result=assess_translation_pair("师兄小心","Sư huynh, cẩn thận!")
    assert result.ok


def test_regression_case_catches_bad_observed_outputs():
    bad={
        "看不懂":"Tôi không thể thực hiện yêu cầu này vì nó vi phạm quyền tác giả.",
        "这不就是太阳底下修炼吗":"Đội phim chi nhiều tiền như vậy.",
        "我果然是天选之人":"Ta quả nhiên là người có máu trời.",
    }
    for case in TRANSLATION_REGRESSION_CASES[:3]:
        ok,reasons=regression_case_passes(case,bad[case["source"]])
        assert not ok
        assert reasons


def test_regression_case_accepts_known_good_meanings():
    good={
        "看不懂":"Không hiểu.",
        "这不就是太阳底下修炼吗":"Chẳng phải đây là tu luyện dưới ánh mặt trời sao?",
        "我果然是天选之人":"Quả nhiên ta là thiên tuyển chi nhân.",
    }
    for case in TRANSLATION_REGRESSION_CASES[:3]:
        ok,reasons=regression_case_passes(case,good[case["source"]])
        assert ok, reasons
