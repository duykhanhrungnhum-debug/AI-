from ai_agent.core.trading_prices import parse_eia_daily_table, parse_fred_price_csv, parse_fred_price_table, price_history_summary


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
    for asset, series in (("WTI", "RWTC"), ("BRENT", "RBRTE")):
        history[asset] = {
            "series_id": series,
            "source_format": "eia-daily-html",
            "observation_count": 40,
            "first_date": "2026-01-01",
            "last_date": "2026-09-01",
            "retrieved_at": "2026-09-21T00:00:00+00:00",
            "content_hash": "hash-" + asset,
            "source_url": "https://www.eia.gov/dnav/pet/hist/example.htm",
            "source_host": "eia.gov",
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


def test_parse_eia_daily_table_reconstructs_weekday_dates():
    html = """<html><table>
    <tr><th>Week Of</th><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th></tr>
    <tr><td>1986 Jan- 6 to Jan-10</td><td>26.53</td><td>25.85</td><td>25.87</td><td>26.03</td><td>25.65</td></tr>
    <tr><td>1986 Jan-13 to Jan-17</td><td>25.08</td><td></td><td>25.18</td><td>23.98</td><td>23.63</td></tr>
    </table></html>"""
    rows = parse_eia_daily_table(html)
    assert rows[0] == {"date": "1986-01-06", "usd_per_barrel": 26.53}
    assert {"date": "1986-01-14", "usd_per_barrel": 25.18} not in rows
    assert {"date": "1986-01-15", "usd_per_barrel": 25.18} in rows
    assert len(rows) == 9


def test_parse_eia_daily_table_handles_cross_year_week():
    html = """<table><tr><td>1986 Dec-29 to Jan- 2</td><td></td><td>17.73</td><td></td><td></td><td>18.13</td></tr></table>"""
    rows = parse_eia_daily_table(html)
    assert rows == [
        {"date": "1986-12-30", "usd_per_barrel": 17.73},
        {"date": "1987-01-02", "usd_per_barrel": 18.13},
    ]
