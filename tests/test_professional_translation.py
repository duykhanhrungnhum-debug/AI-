from ai_agent.core.professional_translation import (
    ProfessionalAssessment,
    Severity,
    TranslationBrief,
    TranslationMemory,
    assess_professional_translation,
    build_professional_prompt,
    select_repair_targets,
)

GLOSSARY={
    "修仙":"tu tiên",
    "师兄":"sư huynh",
    "灵根":"linh căn",
    "筑基":"Trúc Cơ",
    "飞升":"phi thăng",
    "法宝":"pháp bảo",
}


def brief()->TranslationBrief:
    return TranslationBrief(
        domain="xianxia / cultivation drama",
        register="natural Vietnamese dialogue",
        style="faithful, idiomatic, cinematic",
        glossary=GLOSSARY,
        forbidden_additions=("vi phạm bản quyền","đội phim","quay phim"),
    )


def test_professional_prompt_contains_context_glossary_and_fidelity_rules():
    prompt=build_professional_prompt(
        "师兄，你想飞升吗？",
        brief(),
        background="两人在宗门山门前交谈。",
        previous_target=("Sư huynh, chúng ta đi thôi.",),
    )
    assert "师兄 翻译成 sư huynh" in prompt
    assert "飞升 翻译成 phi thăng" in prompt
    assert "不得添加原文没有的信息" in prompt
    assert "不得遗漏关键含义" in prompt
    assert "〖背景信息〗" in prompt
    assert "前文越南语译文" in prompt


def test_rejects_hallucinated_policy_refusal():
    a=assess_professional_translation(
        "看不懂",
        "Tôi không thể thực hiện yêu cầu này vì nó vi phạm bản quyền.",
        brief(),
    )
    assert not a.publishable
    assert any(i.code=="meta_hallucination" and i.severity==Severity.CRITICAL for i in a.issues)


def test_rejects_lost_negation():
    a=assess_professional_translation("我没有灵根","Tôi có linh căn.",brief())
    assert not a.publishable
    assert any(i.code=="negation_lost" for i in a.issues)


def test_rejects_lost_question_intent():
    a=assess_professional_translation("你想飞升吗","Ngươi muốn phi thăng.",brief())
    assert not a.publishable
    assert any(i.code=="question_intent_lost" for i in a.issues)


def test_rejects_terminology_violation():
    a=assess_professional_translation("这是我的法宝","Đây là bảo vật của ta.",brief())
    assert not a.publishable
    assert any(i.code=="terminology_violation" for i in a.issues)


def test_accepts_professional_domain_translation():
    a=assess_professional_translation(
        "师兄，你想飞升吗？",
        "Sư huynh, huynh muốn phi thăng không?",
        brief(),
    )
    assert a.publishable, a.issues


def test_translation_memory_adds_dynamic_terms():
    memory=TranslationMemory()
    memory.remember_term("青云宗","Thanh Vân Tông")
    glossary=memory.glossary_for("我们回青云宗吧",GLOSSARY)
    assert glossary["青云宗"]=="Thanh Vân Tông"
    assert "师兄" not in glossary


def test_select_repair_targets_only_returns_major_or_critical_rows():
    rows=[
        ("看不懂","Không hiểu."),
        ("我没有灵根","Tôi có linh căn."),
        ("这是我的法宝","Đây là pháp bảo của ta."),
    ]
    assert select_repair_targets(rows,brief())==[1]
