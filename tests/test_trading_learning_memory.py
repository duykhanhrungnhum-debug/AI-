from ai_agent.core.trading_learning_memory import (
    DEFAULT_COOLDOWN_CYCLES,
    normalize_error,
    record_learning_memory,
    source_on_cooldown,
)


def test_normalize_fetch_403_uses_host_not_volatile_path():
    a = normalize_error("fetch:https://www.iea.org/reports/oil-market-report: HTTP Error 403: Forbidden")
    b = normalize_error("fetch:https://www.iea.org/other/path: HTTP Error 403: Forbidden")
    assert a[0] == b[0] == "fetch:iea.org:http_403"
    assert a[1] == "iea.org"


def test_repeated_source_failure_becomes_lesson_and_cooldown():
    state = {}
    error = "fetch:https://www.iea.org/reports/oil-market-report: HTTP Error 403: Forbidden"
    assert record_learning_memory(state, [error], cycle=1) == []
    lessons = record_learning_memory(state, [error], cycle=2)
    assert len(lessons) == 1
    assert state["error_memory"]["fetch:iea.org:http_403"]["count"] == 2
    assert source_on_cooldown(state, "https://www.iea.org/reports/oil-market-report", 3)
    assert not source_on_cooldown(
        state,
        "https://www.iea.org/reports/oil-market-report",
        2 + DEFAULT_COOLDOWN_CYCLES + 1,
    )


def test_lesson_is_not_duplicated_on_third_failure():
    state = {}
    error = "fetch:https://www.cmegroup.com/markets/energy: HTTP Error 403: Forbidden"
    record_learning_memory(state, [error], cycle=1)
    record_learning_memory(state, [error], cycle=2)
    new_lessons = record_learning_memory(state, [error], cycle=3)
    assert new_lessons == []
    assert len(state["lessons"]) == 1
