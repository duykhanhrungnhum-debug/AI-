from ai_agent.core.runtime import AgentRuntime


def test_runtime_retries_then_stops_after_consecutive_failures():
    calls = []

    def work():
        calls.append(1)
        return False

    result = AgentRuntime(work, interval_seconds=0, max_cycles=10, max_consecutive_failures=2).run()
    assert result.cycles == 2
    assert result.failures == 2
    assert len(calls) == 2


def test_runtime_runs_for_configured_cycles():
    result = AgentRuntime(lambda: True, interval_seconds=0, max_cycles=3).run()
    assert result.cycles == 3
    assert result.failures == 0
