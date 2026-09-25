"""Small auditable event-momentum backtest for WTI/Brent daily histories.

Research/paper evaluation only. No broker integration or live-order capability.
"""
from __future__ import annotations

from math import sqrt


def _metrics(trades: list[dict]) -> dict:
    returns=[float(t["net_return_pct"]) for t in trades]
    if not returns:
        return {"trade_count":0,"mean_net_return_pct":None,"win_rate":None,"cumulative_net_return_pct":None,"max_drawdown_pct":None,"sharpe_like":None}
    equity=1.0; peak=1.0; max_dd=0.0
    for r in returns:
        equity*=1.0+r/100.0; peak=max(peak,equity); max_dd=min(max_dd,(equity/peak-1.0)*100.0)
    mean=sum(returns)/len(returns)
    if len(returns)>1:
        var=sum((r-mean)**2 for r in returns)/(len(returns)-1); sd=sqrt(var)
        sharpe_like=(mean/sd)*sqrt(len(returns)) if sd else None
    else: sharpe_like=None
    return {"trade_count":len(returns),"mean_net_return_pct":round(mean,6),"win_rate":round(sum(r>0 for r in returns)/len(returns),6),"cumulative_net_return_pct":round((equity-1.0)*100.0,6),"max_drawdown_pct":round(max_dd,6),"sharpe_like":None if sharpe_like is None else round(sharpe_like,6)}


def run_small_backtest(events: list[dict], history: dict, *, hold_trading_days: int=3, in_sample_fraction: float=0.6,
                       spread_bps: float=8.0, slippage_bps: float=2.0, fee_bps: float=1.0) -> dict:
    """Use only information observable at entry; evaluate chronologically held-out trades.

    Signal: direction of the first post-event close versus the strictly pre-event
    anchor. Entry is that first post-event close, so the signal is known before
    the simulated position exists. Exit is N later observed market closes.
    """
    if hold_trading_days < 1: raise ValueError("hold_trading_days must be >= 1")
    if not 0.0 < in_sample_fraction < 1.0: raise ValueError("in_sample_fraction must be between 0 and 1")
    round_trip_cost_pct=2.0*(spread_bps/2.0+slippage_bps+fee_bps)/100.0
    trades=[]
    for event in events:
        published=str(event.get("published_at") or "")
        if not published: continue
        event_day=published[:10]
        for asset in ("WTI","BRENT"):
            obs=history.get(asset,{}).get("observations",[])
            before=[x for x in obs if x.get("date","") < event_day]
            after=[x for x in obs if x.get("date","") >= event_day]
            if not before or len(after) <= hold_trading_days: continue
            anchor=before[-1]; entry=after[0]; exit_=after[hold_trading_days]
            a=float(anchor["usd_per_barrel"]); e=float(entry["usd_per_barrel"]); x=float(exit_["usd_per_barrel"])
            signal=1 if e>a else (-1 if e<a else 0)
            if signal==0: continue
            gross=signal*(x/e-1.0)*100.0
            trades.append({"event_id":event.get("event_id"),"asset":asset,"published_at":published,
                "anchor_date":anchor["date"],"entry_date":entry["date"],"exit_date":exit_["date"],"signal":signal,
                "gross_return_pct":round(gross,6),"cost_pct":round(round_trip_cost_pct,6),
                "net_return_pct":round(gross-round_trip_cost_pct,6)})
    trades.sort(key=lambda t:(t["entry_date"],t["event_id"] or "",t["asset"]))
    split=max(1,min(len(trades)-1,int(len(trades)*in_sample_fraction))) if len(trades)>=2 else len(trades)
    ins=trades[:split]; oos=trades[split:]
    chronology_ok=not ins or not oos or max(t["entry_date"] for t in ins) <= min(t["entry_date"] for t in oos)
    no_lookahead=all(t["anchor_date"] < t["entry_date"] <= t["exit_date"] for t in trades)
    verified=len(ins)>=2 and len(oos)>=2 and chronology_ok and no_lookahead
    return {"verified":verified,"research_only":True,"strategy":"post-event first-close momentum; enter at that close; fixed holding period",
        "hold_trading_days":hold_trading_days,"split":{"method":"chronological","in_sample_fraction":in_sample_fraction,"split_index":split},
        "costs":{"spread_bps":spread_bps,"slippage_bps":slippage_bps,"fee_bps":fee_bps,"round_trip_cost_pct":round(round_trip_cost_pct,6)},
        "no_lookahead":no_lookahead,"chronology_ok":chronology_ok,"trade_count":len(trades),
        "in_sample_metrics":_metrics(ins),"out_of_sample_metrics":_metrics(oos),"trades":trades}
