from ai_agent.core.kaggle_worker import KaggleGpuWorker


def test_submit_script_normalizes_owner_prefixed_ref_to_bare_slug(monkeypatch):
    worker = KaggleGpuWorker(api_token="test-token", username="duykhanhta")

    def fake_request(method, path, *, payload=None):
        assert method == "POST"
        assert path == "/kernels/push"
        # Regression: Kaggle may return an owner-prefixed/duplicated ref while
        # status/output APIs still require only the final bare slug.
        return {
            "ref": "duykhanhta/duykhanhta/ai-agent-video-smoke-2ef149d224",
            "versionNumber": 1,
        }

    monkeypatch.setattr(worker, "_request_json", fake_request)
    submission = worker.submit_script(
        slug="ai-agent-video-smoke-2ef149d224",
        title="ai-agent-video-smoke-2ef149d224",
        source="print('ok')",
        enable_gpu=False,
    )

    assert submission.owner == "duykhanhta"
    assert submission.slug == "ai-agent-video-smoke-2ef149d224"
    assert "/" not in submission.slug
    assert submission.ref == "duykhanhta/ai-agent-video-smoke-2ef149d224"
