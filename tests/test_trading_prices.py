from ai_agent.core.trading_prices import parse_fred_price_csv, parse_fred_price_table, price_history_summary


def test_parse_fred_price_csv_accepts_observation_date_and_skips_missing():
    content = "observation_date,DCOILWTICO\n2026-09-01,100.5\n2026-09-02,.\n2026-09-03,101.25\n"
    rows = parse_fred_price_csv(content, series_id="DCOILWTICO")
    assert rows == [
        {"date": "2026-09-01", "usd_per_barrel": 100.5},
        {"date": "2026-09-03", "usd_per_barrel": 101.25},
    ]


def test_parse_fred_price_csv_accepts_first_column_date_fallback():
    content = "DATE,DCOILBRENTEU\n2026-09-01,120.0\n"
    rows = parse_fred_price_csv(content, series_id="DCOILBRENTEU")
    assert rows[0]["date"] == "2026-09-01"
    assert rows[0]["usd_per_barrel"] == 120.0


def test_price_history_summary_requires_both_assets_and_provenance():
    history = {}
    for asset, series in (("WTI", "DCOILWTICO"), ("BRENT", "DCOILBRENTEU")):
        history[asset] = {
            "series_id": series,
            "observation_count": 40,
            "first_date": "2026-01-01",
            "last_date": "2026-09-01",
            "retrieved_at": "2026-09-21T00:00:00+00:00",
            "content_hash": "hash-" + asset,
            "source_url": "https://fred.stlouisfed.org/example",
        }
    summary = price_history_summary(history)
    assert summary["verified"] is True
    assert summary["assets"]["WTI"]["observation_count"] == 40


def test_price_history_summary_rejects_missing_asset():
    summary = price_history_summary({
        "WTI": {
            "series_id": "DCOILWTICO",
            "observation_count": 40,
            "content_hash": "hash",
        }
    })
    assert summary["verified"] is False


def test_parse_fred_price_table_extracts_daily_rows():
    html = """<html><table>
    <tr><th>DATE</th><th>VALUE</th></tr>
    <tr><td>2026-09-01</td><td>100.50</td></tr>
    <tr><td>2026-09-02</td><td>.</td></tr>
    <tr><td>2026-09-03</td><td>101.25</td></tr>
    </table></html>"""
    rows = parse_fred_price_table(html)
    assert rows == [
        {"date": "2026-09-01", "usd_per_barrel": 100.5},
        {"date": "2026-09-03", "usd_per_barrel": 101.25},
    ]
