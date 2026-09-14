from ai_agent.core.experience import Experience, ExperienceStore
from ai_agent.core.knowledge import KnowledgeItem, KnowledgeSource
from ai_agent.core.knowledge_retrieval import KnowledgeRetriever


def source(uri="https://example.com/source"):
    return KnowledgeSource(uri=uri, content_hash="hash")


def test_retriever_prefers_verified_relevant_knowledge():
    verified = KnowledgeItem("Python uses indentation for blocks", sources=[source()])
    verified.verify(["source confirms indentation"])
    proposed = KnowledgeItem("Python uses braces for blocks", sources=[source("https://example.com/2")])
    retriever = KnowledgeRetriever([proposed, verified])

    context = retriever.retrieve("Python blocks")

    assert context.knowledge[0] is verified
    assert verified.status.value == "verified"
    assert proposed.status.value == "proposed"


def test_retriever_preserves_conflicted_status_and_does_not_present_it_as_verified():
    conflicted = KnowledgeItem("A system has conflicting behavior", sources=[source()])
    conflicted.mark_conflicted("sources disagree")
    retriever = KnowledgeRetriever([conflicted])

    context = retriever.retrieve("system conflicting behavior")
    formatted = context.format_for_prompt()

    assert context.knowledge[0].status.value == "conflicted"
    assert "[conflicted]" in formatted
    assert "status is authoritative" in formatted


def test_retriever_surfaces_relevant_failure_experience_as_lesson():
    store = ExperienceStore()
    store.record(Experience(
        goal="research Python blocks",
        outcome="fetch failed",
        lesson="Retry with another source when the first fetch fails.",
        success=False,
    ))
    retriever = KnowledgeRetriever(experience_store=store)

    context = retriever.retrieve("research Python blocks")

    assert len(context.experiences) == 1
    assert context.experiences[0].success is False
    assert "Retry" in context.format_for_prompt()


def test_retrieval_is_bounded():
    items = [KnowledgeItem(f"Python blocks lesson {i}", tags=["python"]) for i in range(10)]
    retriever = KnowledgeRetriever(items, max_knowledge=3, max_experiences=2)

    context = retriever.retrieve("Python blocks")

    assert len(context.knowledge) == 3
