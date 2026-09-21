"""Historical WTI/Brent spot-price ingestion with provenance.

The source is FRED-hosted daily spot-price data sourced from the U.S. Energy
Information Administration. This module only retrieves and normalizes price
history; it does not create trade signals.
"""
from __future__ import annotations

import csv
from io import StringIO

from .researcher import InternetResearcher


FRED_OIL_SERIES = {
    "WTI": {
        "series_id": "DCOILWTICO",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO&cosd=2010-01-01",
    },
    "BRENT": {
        "series_id": "DCOILBRENTEU",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU&cosd=2010-01-01",
    },
}


def parse_fred_price_csv(content: str, *, series_id: str) -> list[dict]:
    """Parse FRED CSV observations, dropping explicit missing-value rows."""
    reader = csv.DictReader(StringIO(content))
    if not reader.fieldnames or series_id not in reader.fieldnames:
        raise ValueError(f"FRED CSV missing expected series column: {series_id}")
    date_field = "observation_date" if "observation_date" in reader.fieldnames else reader.fieldnames[0]
    output: list[dict] = []
    for row in reader:
        date = str(row.get(date_field, "")).strip()
        raw = str(row.get(series_id, "")).strip()
        if not date or raw in {"", "."}:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        output.append({"date": date, "usd_per_barrel": value})
    if not output:
        raise ValueError(f"FRED series {series_id} contained no numeric observations")
    return output


def fetch_oil_price_history(researcher: InternetResearcher) -> dict:
    """Fetch WTI and Brent daily histories and preserve source provenance."""
    result: dict[str, dict] = {}
    for asset, spec in FRED_OIL_SERIES.items():
        document = researcher.fetch(spec["url"])
        observations = parse_fred_price_csv(document.content, series_id=spec["series_id"])
        result[asset] = {
            "series_id": spec["series_id"],
            "source_url": spec["url"],
            "source_host": "fred.stlouisfed.org",
            "retrieved_at": document.retrieved_at,
            "content_hash": document.content_hash,
            "observation_count": len(observations),
            "first_date": observations[0]["date"],
            "last_date": observations[-1]["date"],
            "observations": observations,
        }
    return result


def price_history_summary(history: dict) -> dict:
    """Return bounded verification metadata without duplicating observations."""
    assets = {}
    for asset, item in history.items():
        assets[asset] = {
            "series_id": item.get("series_id"),
            "observation_count": int(item.get("observation_count", 0)),
            "first_date": item.get("first_date"),
            "last_date": item.get("last_date"),
            "retrieved_at": item.get("retrieved_at"),
            "content_hash": item.get("content_hash"),
            "source_url": item.get("source_url"),
        }
    ready = set(assets) == {"WTI", "BRENT"} and all(
        item["observation_count"] >= 30 and item["content_hash"]
        for item in assets.values()
    )
    return {"verified": bool(ready), "assets": assets}
