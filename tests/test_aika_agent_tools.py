from pathlib import Path

import ai_agent.chat_api as chat_api
from ai_agent.agent_broker import AIKAAgentBroker, parse_tool_call
from ai_agent.media_broker import MediaToolBroker


def test_parse_tool_call_is_structured_not_keyword_routing():
    assert parse_tool_call('{"tool":"image","input":"Một con mèo màu trắng"}') == (
        "image",
        "Một con mèo màu trắng",
    )
    assert parse_tool_call('{"tool":"video","input":"A lantern sways in the wind"}') == (
        "video",
        "A lantern sways in the wind",
    )
    assert parse_tool_call("Giải thích cách tạo video") is None


def test_agent_job_is_always_brain_first(monkeypatch):
    broker = AIKAAgentBroker()
    monkeypatch.setattr(broker, "ensure_worker", lambda: None)
    job = broker.create_job("AIKA tạo ảnh một con mèo")
    assert job.status == "pending"
    pulled = broker.pull_job()
    assert pulled == {"job_id": job.job_id, "prompt": "AIKA tạo ảnh một con mèo"}
    assert broker.get_job(job.job_id)["kind"] == "agent"


def test_aika_tool_decision_dispatches_same_public_job_id(monkeypatch):
    broker = AIKAAgentBroker()
    monkeypatch.setattr(broker, "ensure_worker", lambda: None)
    job = broker.create_job("tạo video một con mèo chạy")

    calls = []

    def fake_create_job(kind, command, *, job_id=None):
        calls.append((kind, command, job_id))
        return object()

    monkeypatch.setattr("ai_agent.media_broker.MEDIA_BROKER.create_job", fake_create_job)
    broker.finish_job({
        "job_id": job.job_id,
        "text": '{"tool":"video","input":"Một con mèo chạy tự nhiên trên bãi cỏ"}',
        "provider": "test-brain",
        "model": "test-model",
    })

    assert calls == [("video", "Một con mèo chạy tự nhiên trên bãi cỏ", job.job_id)]
    state = broker.get_job(job.job_id)
    assert state["status"] == "delegated"
    assert state["decision"] == "tool"
    assert state["tool"] == "video"


def test_normal_answer_does_not_call_media_tool(monkeypatch):
    broker = AIKAAgentBroker()
    monkeypatch.setattr(broker, "ensure_worker", lambda: None)
    job = broker.create_job("AIKA là ai?")

    def should_not_run(*args, **kwargs):
        raise AssertionError("media tool must not run for a normal answer")

    monkeypatch.setattr("ai_agent.media_broker.MEDIA_BROKER.create_job", should_not_run)
    broker.finish_job({"job_id": job.job_id, "text": "Tôi là AIKA."})
    state = broker.get_job(job.job_id)
    assert state["status"] == "done"
    assert state["decision"] == "respond"
    assert state["text"] == "Tôi là AIKA."


def test_media_broker_accepts_aika_owned_job_id(monkeypatch):
    broker = MediaToolBroker()
    monkeypatch.setattr("threading.Thread.start", lambda self: None)
    job = broker.create_job("image", "một con mèo", job_id="aika-job-123")
    assert job.job_id == "aika-job-123"
    assert broker.get_job("aika-job-123")["kind"] == "image"


def test_production_api_has_no_keyword_router_or_legacy_chat_session():
    source = Path(chat_api.__file__).read_text(encoding="utf-8")
    assert "route_skill" not in source
    assert "skill_hint" not in source
    assert "chat_session" not in source
    assert "AGENT_BROKER.create_job(command)" in source
