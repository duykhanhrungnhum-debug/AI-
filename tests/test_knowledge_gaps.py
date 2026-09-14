from ai_agent.core.knowledge import KnowledgeItem
from ai_agent.core.knowledge_gaps import GapStatus, KnowledgeGapDetector
from ai_agent.core.learning_planner import LearningPlanner


def test_missing_task_creates_high_priority_gap():
    gaps = KnowledgeGapDetector().detect("learn quantum routing", [])
    assert len(gaps) == 1
    assert gaps[0].priority == 100
    assert gaps[0].status is GapStatus.OPEN


def test_unverified_related_knowledge_creates_gap():
    item = KnowledgeItem(statement="Quantum routing uses a test procedure")
    gaps = KnowledgeGapDetector().detect("quantum routing", [item])
    assert len(gaps) == 1
    assert item.id in gaps[0].related_knowledge_ids


def test_verified_related_knowledge_does_not_create_gap():
    item = KnowledgeItem(statement="Quantum routing uses a test procedure")
    item.verify(["direct evidence"])
    assert KnowledgeGapDetector().detect("quantum routing", [item]) == []


def test_planner_marks_gap_planned_and_bounds_tasks():
    planner = LearningPlanner(max_tasks=1)
    tasks = planner.plan("learn autonomous planning", [])
    assert len(tasks) == 1
    assert tasks[0].max_attempts == 3
    assert tasks[0].objective.startswith("Research and verify:")
