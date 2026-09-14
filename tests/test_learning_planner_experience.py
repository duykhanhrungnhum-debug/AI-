from ai_agent.core.experience import Experience, ExperienceStore
from ai_agent.core.learning_planner import LearningPlanner


def test_previous_failure_changes_next_learning_plan():
    store = ExperienceStore()
    store.record(Experience(
        goal="internet research reliability",
        outcome="failed",
        lesson="network unavailable; use another source instead of retrying the same endpoint",
        success=False,
    ))

    planner = LearningPlanner(experience_store=store)
    tasks = planner.plan("internet research reliability", knowledge=[])

    assert len(tasks) == 1
    assert "Prior failure lessons to avoid repeating" in tasks[0].objective
    assert "network unavailable" in tasks[0].objective
