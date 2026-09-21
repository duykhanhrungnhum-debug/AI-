"""Historical WTI/Brent spot-price ingestion with provenance.

The source is FRED-hosted daily spot-price data sourced from the U.S. Energy
Information Administration. This module only retrieves and normalizes price
history; it does not create trade signals.
"""
from __future__ import annotations

import csv
import re
from html.parser import HTMLParser
from io import StringIO

from .researcher import InternetResearcher


FRED_OIL_SERIES = {
    "WTI": {
        "series_id": "DCOILWTICO",
        "url": "https://fred.stlouisfed.org/data/DCOILWTICO",
    },
    "BRENT": {
        "series_id": "DCOILBRENTEU",
        "url": "https://fred.stlouisfed.org/data/DCOILBRENTEU",
    },
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_row = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in {"td", "th"}:
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_row and self.in_cell and tag in {"td", "th"}:
            self.row.append(" ".join(self.cell_parts).strip())
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.row:
                self.rows.append(self.row)
            self.in_row = False


def _normalize_rows(rows: list[tuple[str, str]]) -> list[dict]:
    output: list[dict] = []
    for date, raw in rows:
        date = date.strip()
        raw = raw.strip()
        if not _DATE_RE.match(date) or raw in {"", "."}:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        output.append({"date": date, "usd_per_barrel": value})
    if not output:
        raise ValueError("price source contained no numeric observations")
    output.sort(key=lambda item: item["date"])
    return output


def parse_fred_price_csv(content: str, *, series_id: str) -> list[dict]:
    """Parse FRED CSV observations, dropping explicit missing-value rows."""
    reader = csv.DictReader(StringIO(content))
    if not reader.fieldnames or series_id not in reader.fieldnames:
        raise ValueError(f"FRED CSV missing expected series column: {series_id}")
    date_field = "observation_date" if "observation_date" in reader.fieldnames else reader.fieldnames[0]
    return _normalize_rows([
        (str(row.get(date_field, "")), str(row.get(series_id, "")))
        for row in reader
    ])


def parse_fred_price_table(content: str) -> list[dict]:
    """Parse the official FRED Table Data HTML without external dependencies."""
    parser = _TableParser()
    parser.feed(content)
    pairs: list[tuple[str, str]] = []
    for row in parser.rows:
        if len(row) >= 2 and _DATE_RE.match(row[0].strip()):
            pairs.append((row[0], row[1]))
    return _normalize_rows(pairs)


def fetch_oil_price_history(researcher: InternetResearcher) -> dict:
    """Fetch WTI and Brent daily histories and preserve source provenance."""
    result: dict[str, dict] = {}
    for asset, spec in FRED_OIL_SERIES.items():
        document = researcher.fetch(spec["url"])
        if "html" in document.content_type.lower() or "<html" in document.content.lower():
            observations = parse_fred_price_table(document.content)
            source_format = "fred-table-html"
        else:
            observations = parse_fred_price_csv(document.content, series_id=spec["series_id"])
            source_format = "fred-csv"
        result[asset] = {
            "series_id": spec["series_id"],
            "source_url": spec["url"],
            "source_host": "fred.stlouisfed.org",
            "source_format": source_format,
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
            "source_format": item.get("source_format"),
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
