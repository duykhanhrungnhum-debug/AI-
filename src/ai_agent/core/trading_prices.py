"""Historical WTI/Brent spot-price ingestion with primary-source provenance.

Daily spot-price history is retrieved directly from U.S. Energy Information
Administration (EIA) public tables. This module only retrieves and normalizes
price history; it does not create trade signals.
"""
from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from html.parser import HTMLParser
from io import StringIO

from .researcher import InternetResearcher


EIA_OIL_SERIES = {
    "WTI": {
        "series_id": "RWTC",
        "url": "https://www.eia.gov/dnav/pet/hist/RWTCD.htm",
    },
    "BRENT": {
        "series_id": "RBRTE",
        "url": "https://www.eia.gov/dnav/pet/hist/RBRTED.htm",
    },
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EIA_WEEK_RE = re.compile(
    r"^\s*(\d{4})\s+([A-Za-z]{3})-\s*(\d{1,2})\s+to\s+[A-Za-z]{3}-\s*\d{1,2}\s*$"
)
_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


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
            text = " ".join(self.cell_parts).replace("\xa0", " ").strip()
            self.row.append(text)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.row:
                self.rows.append(self.row)
            self.in_row = False


def _numeric_value(raw: str) -> float | None:
    text = raw.strip()
    if text in {"", ".", "-", "--", "NA", "W"}:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _normalize_rows(rows: list[tuple[str, str]]) -> list[dict]:
    output: list[dict] = []
    for day, raw in rows:
        day = day.strip()
        value = _numeric_value(raw)
        if not _DATE_RE.match(day) or value is None:
            continue
        output.append({"date": day, "usd_per_barrel": value})
    if not output:
        raise ValueError("price source contained no numeric observations")
    output.sort(key=lambda item: item["date"])
    return output


def parse_eia_daily_table(content: str) -> list[dict]:
    """Parse EIA daily history rows of Week Of + Mon..Fri values."""
    parser = _TableParser()
    parser.feed(content)
    output: list[dict] = []
    seen_dates: set[str] = set()
    for row in parser.rows:
        if len(row) < 6:
            continue
        label = " ".join(row[0].split())
        match = _EIA_WEEK_RE.match(label)
        if not match:
            continue
        year_text, month_text, day_text = match.groups()
        month = _MONTHS.get(month_text.title())
        if month is None:
            continue
        try:
            week_start = date(int(year_text), month, int(day_text))
        except ValueError:
            continue
        for offset, raw in enumerate(row[1:6]):
            value = _numeric_value(raw)
            if value is None:
                continue
            day = (week_start + timedelta(days=offset)).isoformat()
            if day in seen_dates:
                continue
            seen_dates.add(day)
            output.append({"date": day, "usd_per_barrel": value})
    if not output:
        raise ValueError("EIA daily table contained no numeric observations")
    output.sort(key=lambda item: item["date"])
    return output


def parse_fred_price_csv(content: str, *, series_id: str) -> list[dict]:
    """Backward-compatible parser retained for existing fixtures/tests."""
    reader = csv.DictReader(StringIO(content))
    if not reader.fieldnames or series_id not in reader.fieldnames:
        raise ValueError(f"FRED CSV missing expected series column: {series_id}")
    date_field = "observation_date" if "observation_date" in reader.fieldnames else reader.fieldnames[0]
    return _normalize_rows([
        (str(row.get(date_field, "")), str(row.get(series_id, "")))
        for row in reader
    ])


def parse_fred_price_table(content: str) -> list[dict]:
    """Backward-compatible simple two-column table parser."""
    parser = _TableParser()
    parser.feed(content)
    pairs: list[tuple[str, str]] = []
    for row in parser.rows:
        if len(row) >= 2 and _DATE_RE.match(row[0].strip()):
            pairs.append((row[0], row[1]))
    return _normalize_rows(pairs)


def fetch_oil_price_history(researcher: InternetResearcher) -> dict:
    """Fetch WTI and Brent daily histories directly from EIA."""
    result: dict[str, dict] = {}
    for asset, spec in EIA_OIL_SERIES.items():
        document = researcher.fetch(spec["url"])
        observations = parse_eia_daily_table(document.content)
        result[asset] = {
            "series_id": spec["series_id"],
            "source_url": spec["url"],
            "source_host": "eia.gov",
            "source_format": "eia-daily-html",
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
            "source_host": item.get("source_host"),
        }
    ready = set(assets) == {"WTI", "BRENT"} and all(
        item["observation_count"] >= 30
        and item["content_hash"]
        and item["source_host"] == "eia.gov"
        for item in assets.values()
    )
    return {"verified": bool(ready), "assets": assets}
