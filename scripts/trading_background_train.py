#!/usr/bin/env python3
"""Run one verification-first background learning cycle for crude-oil trading."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ai_agent.core.news_rss import GoogleNewsRSSProvider
from ai_agent.core.researcher import InternetResearcher
from ai_agent.core.trading_events import enforce_as_of_cutoff, normalize_news_signals
from ai_agent.core.trading_learning_memory import record_learning_memory, source_on_cooldown
from ai_agent.core.trading_prices import fetch_oil_price_history, price_history_summary
from ai_agent.core.trading_reactions import study_event_price_reactions
from ai_agent.core.trading_skill import TradingEvidence, TradingEvidenceVerifier, TradingResearchEpisode, research_plan, source_hubs


def default_state() -> dict:
    return {"schema": 4, "cycles": 0, "verified_cycles": 0, "history": [], "source_success": {}, "coverage": {}, "news_headlines": [], "error_memory": {}, "source_cooldowns": {}, "lessons": []}


def load_state(path: Path) -> dict:
    if not path.exists(): return default_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict): return default_state()
        base = default_state(); base.update(payload); return base
    except Exception: return default_state()


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def add_retrieval(episode, researcher, *, category, title, url, seen_urls, state, cycle, skipped_sources) -> None:
    if url in seen_urls: return
    seen_urls.add(url)
    if source_on_cooldown(state, url, cycle):
        skipped_sources.append({"category": category, "title": title, "url": url, "reason": "learning-memory-cooldown"}); return
    try:
        document = researcher.fetch(url)
        episode.evidence.append(TradingEvidence.from_retrieval(category=category, title=title, url=url, content_hash=document.content_hash, retrieved_at=document.retrieved_at))
    except Exception as exc: episode.errors.append(f"fetch:{url}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="trading_training_episode.json"); parser.add_argument("--state", default=".agent_state/trading/state.json"); parser.add_argument("--news-per-query", type=int, default=3); args = parser.parse_args()
    state_path = Path(args.state); state = load_state(state_path); cycle = int(state.get("cycles", 0)) + 1
    news = GoogleNewsRSSProvider(timeout=15.0); researcher = InternetResearcher(timeout=15.0, max_bytes=750_000); price_researcher = InternetResearcher(timeout=45.0, max_bytes=2_000_000)
    episode = TradingResearchEpisode(); seen_urls=set(); news_signals=[]; skipped_sources=[]
    for category, title, url in source_hubs(): add_retrieval(episode, researcher, category=category, title=title, url=url, seen_urls=seen_urls, state=state, cycle=cycle, skipped_sources=skipped_sources)
    for category, query in research_plan():
        try:
            signals = news.search(query, category=category, limit=args.news_per_query)
            if not signals: episode.errors.append(f"news-empty:{category}:{query}")
            news_signals.extend(signal.to_dict() for signal in signals)
        except Exception as exc: episode.errors.append(f"news:{category}:{query}: {exc}")
    normalized_events = normalize_news_signals(news_signals); as_of_gate = enforce_as_of_cutoff(normalized_events, datetime.now(timezone.utc).isoformat())
    price_history = {}
    try: price_history = fetch_oil_price_history(price_researcher)
    except Exception as exc: episode.errors.append(f"price-history:{exc}")
    price_summary = price_history_summary(price_history)
    reaction_study = study_event_price_reactions(as_of_gate.get("eligible_events", normalized_events), price_history)
    episode.verification = TradingEvidenceVerifier().verify(episode.evidence); new_lessons = record_learning_memory(state, episode.errors, cycle=cycle)
    payload = episode.to_dict(); payload.update({"news_signals":news_signals,"normalized_events":normalized_events,"as_of_gate":as_of_gate,"price_history":price_history,"price_history_summary":price_summary,"event_price_reaction_study":reaction_study,"skipped_sources":skipped_sources,"new_lessons":new_lessons})
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    state["schema"]=4; state["cycles"]=cycle
    if episode.verification.passed: state["verified_cycles"]=int(state.get("verified_cycles",0))+1
    source_success=Counter(state.get("source_success",{})); coverage=Counter(state.get("coverage",{}))
    for item in episode.evidence: source_success[item.source_host]+=1; coverage[item.category]+=1
    state["source_success"]=dict(source_success.most_common(100)); state["coverage"]=dict(coverage)
    recent=list(state.get("news_headlines",[])); recent.extend(news_signals); state["news_headlines"]=recent[-150:]
    recurring_errors=sum(int(item.get("count",0))>=2 for item in state.get("error_memory",{}).values() if isinstance(item,dict))
    summary={"created_at":episode.created_at,"cycle":cycle,"verified":episode.verification.passed,"evidence_count":len(episode.evidence),"independent_domains":episode.verification.independent_domains,"primary_sources":episode.verification.primary_sources,"categories":list(episode.verification.categories),"news_signal_count":len(news_signals),"normalized_event_count":len(normalized_events),"as_of_cutoff":as_of_gate["as_of"],"as_of_eligible_event_count":as_of_gate["eligible_count"],"lookahead_rejected_event_count":as_of_gate["rejected_count"],"no_lookahead_enforced":True,"price_history_verified":price_summary["verified"],"wti_observation_count":price_summary.get("assets",{}).get("WTI",{}).get("observation_count",0),"brent_observation_count":price_summary.get("assets",{}).get("BRENT",{}).get("observation_count",0),"event_price_reaction_verified":reaction_study["verified"],"event_price_reaction_count":reaction_study["reaction_count"],"reaction_horizons_trading_days":reaction_study["horizons_trading_days"],"error_count":len(episode.errors),"skipped_source_count":len(skipped_sources),"new_lesson_count":len(new_lessons),"recurring_error_count":recurring_errors}
    history=list(state.get("history",[])); history.append(summary); state["history"]=history[-200:]; state["last_episode"]=summary; save_state(state_path,state)
    print(json.dumps(summary,ensure_ascii=False))
    for lesson in new_lessons: print("LESSON "+lesson["lesson"])
    for skipped in skipped_sources: print("LEARNED-SKIP "+skipped["url"])
    for error in episode.errors: print("WARN "+error)
    if not episode.verification.passed or not price_summary["verified"] or not reaction_study["verified"]:
        print("Trading cycle NOT VERIFIED: evidence, price history, and event/price reaction gates are required"); return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
