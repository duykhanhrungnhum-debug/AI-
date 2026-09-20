#!/usr/bin/env python3
"""Run one verification-first background learning cycle for crude-oil trading."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ai_agent.core.researcher import InternetResearcher
from ai_agent.core.search import DuckDuckGoSearchProvider
from ai_agent.core.trading_skill import (
    TradingEvidence,
    TradingEvidenceVerifier,
    TradingResearchEpisode,
    research_plan,
    source_hubs,
)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"schema": 1, "cycles": 0, "verified_cycles": 0, "history": [], "source_success": {}, "coverage": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {"schema": 1, "cycles": 0, "verified_cycles": 0, "history": [], "source_success": {}, "coverage": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def add_retrieval(episode: TradingResearchEpisode, researcher: InternetResearcher, *,
                  category: str, title: str, url: str, seen_urls: set[str]) -> None:
    if url in seen_urls:
        return
    seen_urls.add(url)
    try:
        document = researcher.fetch(url)
        episode.evidence.append(TradingEvidence.from_retrieval(
            category=category,
            title=title,
            url=url,
            content_hash=document.content_hash,
            retrieved_at=document.retrieved_at,
        ))
    except Exception as exc:
        episode.errors.append(f"fetch:{url}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="trading_training_episode.json")
    parser.add_argument("--state", default=".agent_state/trading/state.json")
    parser.add_argument("--results-per-query", type=int, default=2)
    args = parser.parse_args()

    provider = DuckDuckGoSearchProvider(timeout=15.0)
    researcher = InternetResearcher(timeout=15.0, max_bytes=750_000)
    episode = TradingResearchEpisode()
    seen_urls: set[str] = set()

    # First fetch stable primary-source hubs. This is the resilient baseline and
    # makes the learning cycle independent of any single search engine layout.
    for category, title, url in source_hubs():
        add_retrieval(
            episode, researcher, category=category, title=title, url=url,
            seen_urls=seen_urls,
        )

    # Then enrich with current discovered pages/news. Search failure is recorded
    # but cannot erase successfully retrieved primary-source evidence.
    for category, query in research_plan():
        try:
            results = provider.search(query, limit=args.results_per_query)
            if not results:
                episode.errors.append(f"search-empty:{category}:{query}")
        except Exception as exc:
            episode.errors.append(f"search:{category}:{query}: {exc}")
            continue

        for result in results:
            add_retrieval(
                episode, researcher, category=category,
                title=result.title or query, url=result.url, seen_urls=seen_urls,
            )

    episode.verification = TradingEvidenceVerifier().verify(episode.evidence)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(episode.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    state_path = Path(args.state)
    state = load_state(state_path)
    state["schema"] = 1
    state["cycles"] = int(state.get("cycles", 0)) + 1
    if episode.verification.passed:
        state["verified_cycles"] = int(state.get("verified_cycles", 0)) + 1

    source_success = Counter(state.get("source_success", {}))
    coverage = Counter(state.get("coverage", {}))
    for item in episode.evidence:
        source_success[item.source_host] += 1
        coverage[item.category] += 1
    state["source_success"] = dict(source_success.most_common(100))
    state["coverage"] = dict(coverage)

    summary = {
        "created_at": episode.created_at,
        "verified": episode.verification.passed,
        "evidence_count": len(episode.evidence),
        "independent_domains": episode.verification.independent_domains,
        "primary_sources": episode.verification.primary_sources,
        "categories": list(episode.verification.categories),
        "error_count": len(episode.errors),
    }
    history = list(state.get("history", []))
    history.append(summary)
    state["history"] = history[-200:]
    state["last_episode"] = summary
    save_state(state_path, state)

    print(json.dumps(summary, ensure_ascii=False))
    for error in episode.errors:
        print("WARN " + error)
    if episode.verification.passed:
        return 0
    print("Trading learning cycle NOT VERIFIED: " + "; ".join(episode.verification.reasons))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
