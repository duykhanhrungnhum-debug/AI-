from ai_agent.core.failure_report import FailureReport


def test_failure_report_stops_blind_retry_for_template_errors(tmp_path):
    report = FailureReport.from_exception(
        stage="narrative-quality",
        error=ValueError("Invalid format specifier in generated worker"),
    )

    assert report.status == "BLOCKED"
    assert report.retryable is False
    assert "do not rerun unchanged code" in report.next_action
    target = report.write(tmp_path / "failure.json")
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["stage"] == "narrative-quality"
    assert data["status"] == "BLOCKED"


def test_failure_report_marks_kaggle_capacity_as_retryable():
    report = FailureReport.from_exception(
        stage="image-generation",
        error=RuntimeError("Maximum batch GPU session count of 2 reached."),
    )

    assert report.retryable is True
    assert "GPU slot" in report.next_action
