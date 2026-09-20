# Trading Skill v1 — verification-first background learning

## Purpose

Train the Agent to research real-world drivers of crude-oil markets instead of
reasoning from chart data alone. Version 1 is deliberately **research/paper
training only** and contains no broker-order execution.

## Required evidence layers

1. Market structure and positioning: WTI/Brent, futures curve/spreads, CFTC.
2. Physical supply/demand: EIA, OPEC, IEA, production, inventories, refining.
3. External events: sanctions, shipping, pipelines/refineries, Gulf weather.
4. Macro: rates, USD/financial conditions, demand-sensitive macro releases.

Search results are discovery only. A source is accepted into a learning episode
only after the Agent retrieves it and records a content hash and timestamp.

## Source hierarchy

- Tier 1: official/primary sources (EIA, OPEC, IEA, CFTC, CME/ICE, Fed/FRED,
  NOAA/NHC, Treasury).
- Tier 2: high-quality news used for fast-moving events (Reuters, AP,
  Bloomberg, FT, WSJ).
- Tier 3: secondary sources.
- Unknown/unverifiable sources never satisfy the authoritative-source gate.

## Verification gate

A cycle is VERIFIED only when it contains at least 3 independent source
domains, at least one primary source, supply/demand evidence, market evidence,
and at least one external-driver category (macro, geopolitics, or weather).

Failure is explicit: the workflow exits non-zero rather than pretending the
cycle learned valid facts.

## Background operation

GitHub Actions runs the research cycle hourly on cloud infrastructure. State is
restored/saved through a bounded GitHub Actions cache and each run uploads its
episode and state as an artifact. The phone is not part of the runtime.

Persistent state records cycle count, verified-cycle count, category coverage,
source-domain retrieval history, and up to 200 recent cycle summaries.

## Next curriculum stages

- Extract normalized facts/events from fetched documents.
- Timestamp events and classify expected supply/demand direction and horizon.
- Join those events to historical WTI/Brent price reactions.
- Backtest hypotheses and keep rejected hypotheses as failure lessons.
- Add paper-trade scenario generation with strict risk limits.
- Only after measured evidence should any separate live-execution capability be
  considered; it is intentionally absent from v1.
