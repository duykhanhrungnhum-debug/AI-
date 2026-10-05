"""Public delayed intraday reference adapter for research-only runtime evidence.

Uses Yahoo Finance chart data for price/volume bars only. It does not claim exchange
microstructure, bid/ask, depth, aggressor side, or open interest. Official CME pages
remain the authority for contract definitions and delayed-quote limitations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen

SYMBOLS = {"WTI": "CL=F", "BRENT": "BZ=F"}


def fetch_reference_bars(asset: str, *, interval: str = "5m", range_: str = "1d") -> dict:
    if asset not in SYMBOLS:
        raise ValueError("asset must be WTI or BRENT")
    symbol = SYMBOLS[asset]
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + quote(symbol, safe="")
        + f"?interval={interval}&range={range_}&includePrePost=true&events=div%2Csplits"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 research-runtime-evidence"})
    with urlopen(req, timeout=20) as resp:
        payload = json.load(resp)
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote0 = result["indicators"]["quote"][0]
    bars = []
    for i, epoch in enumerate(timestamps):
        close = quote0["close"][i]
        if close is None:
            continue
        bars.append({
            "timestamp": datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(),
            "close": float(close),
            "volume": float((quote0.get("volume") or [0] * len(timestamps))[i] or 0),
        })
    return {
        "asset": asset,
        "symbol": symbol,
        "source": "Yahoo Finance public delayed/reference chart endpoint",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "interval": interval,
        "range": range_,
        "bars": bars,
        "microstructure_available": False,
        "limitations": [
            "reference/delayed price-volume bars only",
            "no bid/ask or order-book depth",
            "no aggressor-side classification",
            "no exchange-grade validation",
        ],
    }
