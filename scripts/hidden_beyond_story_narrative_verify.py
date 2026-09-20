#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

from ai_agent.core.kaggle_narrative import KaggleNarrativeProcessor
from ai_agent.core.kaggle_worker import KaggleGpuWorker


API = os.environ.get(
    "STORY_PROCESSOR_API",
    "https://rlqqcuuphjmwksanbfml.supabase.co/functions/v1/story-processor-api",
).rstrip("/")
AUDIENCE = "hidden-beyond-story-processor"
REPAIR_SIGNATURE = "qwen25-3b-semantic-narrative-review-v1"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "hidden-beyond-story-narrative-output"))


def http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict | None = None,
    timeout: float = 120,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:4000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return parsed


def get_oidc() -> str:
    request_url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    separator = "&" if "?" in request_url else "?"
    url = request_url + separator + "audience=" + urllib.parse.quote(AUDIENCE)
    token = os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
    data = http_json(
        url,
        headers={"Authorization": f"bearer {token}"},
        timeout=30,
    )
    value = str(data.get("value") or "").strip()
    if not value:
        raise RuntimeError("GitHub OIDC response did not contain a token")
    return value


def api_post(route: str, payload: dict) -> dict:
    return http_json(
        f"{API}/{route}",
        method="POST",
        headers={
            "Authorization": f"Bearer {get_oidc()}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=120,
    )


def classify_failure(error: BaseException) -> str:
    text = str(error).casefold()
    if (
        "maximum batch gpu session count" in text
        or "cuda out of memory" in text
        or "torch.outofmemoryerror" in text
    ):
        return "resource_capacity"
    return "verification"


def record_failure(episode_id: int, error: BaseException) -> dict | None:
    try:
        response = api_post(
            "verify-fail",
            {
                "episode_id": episode_id,
                "error": str(error)[:2000],
                "failure_kind": classify_failure(error),
                "repair_signature": REPAIR_SIGNATURE,
            },
        )
        print("NARRATIVE_FAILURE_RECORDED", json.dumps(response, ensure_ascii=False))
        return response
    except Exception as report_error:
        print(f"Could not record narrative verification failure: {report_error}", file=sys.stderr)
        return None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    if not username:
        raise RuntimeError("KAGGLE_USERNAME is required")

    claim = api_post("narrative-verify-claim", {})
    (OUTPUT_DIR / "claim.json").write_text(
        json.dumps(claim, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if claim.get("stage") == "idle":
        print("HIDDEN_BEYOND_NARRATIVE_VERIFY_IDLE")
        print(json.dumps(claim, ensure_ascii=False))
        return 0
    if claim.get("stage") != "narrative_verify_claimed":
        raise RuntimeError(f"Unexpected narrative verify claim: {claim}")

    job = claim["job"]
    episode_id = int(job["episode_id"])
    source_text = str(job.get("source_text") or "").strip()
    script_vi = str(job.get("script_vi") or "").strip()
    if not source_text or not script_vi:
        error = RuntimeError("Narrative verification job is missing source or script")
        record_failure(episode_id, error)
        raise error

    try:
        worker = KaggleGpuWorker(
            api_token=token,
            username=username,
            timeout=120,
        )
        processor = KaggleNarrativeProcessor(
            worker=worker,
            model="Qwen/Qwen2.5-3B-Instruct",
            kernel_slug=f"hidden-beyond-narrative-verify-{episode_id}",
            poll_interval=15,
            max_poll_attempts=120,
            max_revisions=0,
            temperature=0.0,
        )
        result = processor.process(
            source_text,
            target_language="Vietnamese",
            existing_script=script_vi,
        )
        if result.script != script_vi:
            raise RuntimeError("Review-only narrative verifier modified the production script")

        evidence = {
            "episode_id": episode_id,
            "series_title": job.get("series_title"),
            "episode_no": job.get("episode_no"),
            "source_chars": len(source_text),
            "script_chars": len(script_vi),
            "review_passed": result.review.passed,
            "issues": list(result.review.issues),
            "reviewer": result.review.reviewer,
            "checks": list(result.review.checks),
            "evidence": list(result.evidence),
            "repair_signature": REPAIR_SIGNATURE,
        }

        if not result.review.passed:
            error = RuntimeError(
                "Narrative verification failed: "
                + ("; ".join(result.review.issues) or "unknown issue")
            )
            failure_response = record_failure(episode_id, error)
            evidence["verify_fail_response"] = failure_response
            (OUTPUT_DIR / "result.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise error

        passed = api_post("verify-pass", {"episode_id": episode_id})
        if passed.get("stage") != "verified":
            raise RuntimeError(f"verify-pass did not verify episode: {passed}")
        evidence["verify_pass_response"] = passed
        (OUTPUT_DIR / "result.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("HIDDEN_BEYOND_NARRATIVE_VERIFY_OK")
        print(json.dumps(evidence, ensure_ascii=False))
        return 0
    except BaseException as exc:
        result_path = OUTPUT_DIR / "result.json"
        if not result_path.exists():
            record_failure(episode_id, exc)
            result_path.write_text(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "stage": "failed",
                        "error": str(exc),
                        "repair_signature": REPAIR_SIGNATURE,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
