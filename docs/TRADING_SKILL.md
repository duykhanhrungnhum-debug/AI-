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

The worker has two independent discovery paths:

- stable official source hubs are fetched directly and content-hashed;
- a keyless current-news RSS layer searches changing events each cycle and
  retains bounded headline/source/timestamp evidence.

News/search signals are context and discovery only. They do not replace the
official-source verification gate.

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
source-domain retrieval history, up to 150 recent news signals, and up to 200
recent cycle summaries.

## Current boundary

A VERIFIED v1 cycle proves that the research/evidence collector operated and
met the evidence gate. It does **not** prove a profitable trading strategy or a
trained model. Strategy learning needs event extraction, price-reaction joins,
backtests, and paper-trade evaluation.

## Next curriculum stages

- Extract normalized facts/events from fetched documents and news signals.
- Timestamp events and classify expected supply/demand direction and horizon.
- Join those events to historical WTI/Brent price reactions.
- Backtest hypotheses and keep rejected hypotheses as failure lessons.
- Add paper-trade scenario generation with strict risk limits.
- Only after measured evidence should any separate live-execution capability be
  considered; it is intentionally absent from v1.
