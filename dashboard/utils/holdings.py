"""Current sector ETF holdings with purchase price, current price, and P&L."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def enrich_holdings_with_current(
    holdings: list[dict], current_value: float | None
) -> list[dict[str, Any]]:
    """
    Enrich pipeline holdings with current price, invested, current_value, P&L.
    holdings: [{ticker, weight, purchase_price}]
    current_value: total portfolio value (from state)
    """
    if not holdings or current_value is None or current_value <= 0:
        return []
    out = []
    for h in holdings:
        t = h.get("ticker")
        w = h.get("weight", 0) / 100
        buy = h.get("purchase_price", 0)
        if not t or w <= 0:
            continue
        cur = _get_current_price(t)
        if cur is None:
            cur = buy
        pos_value = current_value * w
        invested = pos_value * (buy / cur) if cur > 0 else pos_value
        pnl_amt = pos_value - invested
        pnl_pct = (cur / buy - 1) * 100 if buy > 0 else 0
        out.append({
            "ticker": t,
            "weight": w * 100,
            "purchase_price": round(buy, 2),
            "current_price": round(cur, 2),
            "invested": round(invested, 2),
            "current_value": round(pos_value, 2),
            "pnl_amt": round(pnl_amt, 2),
            "pnl_pct": round(pnl_pct, 2),
        })
    return out
SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU",
]
REBALANCE_B2 = 21
MOM_LOOKBACK = 252


def _get_raw_prices() -> pd.DataFrame | None:
    """Load sector ETF prices from data/ or data_refresh/."""
    for base in ["data_refresh", "data"]:
        for name in ["raw_data_extended_2005.csv", "raw_data.csv"]:
            p = ROOT / base / name
            if p.exists():
                try:
                    df = pd.read_csv(p, index_col=0, parse_dates=True)
                    df.index = pd.to_datetime(df.index)
                    cols = [c for c in SECTOR_ETFS if c in df.columns]
                    if len(cols) >= 3:
                        return df[cols].ffill().dropna(how="all")
                except Exception:
                    pass
    return None


def _get_current_price(ticker: str) -> float | None:
    """Get latest price from yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        try:
            return float(t.fast_info["last_price"])
        except (KeyError, TypeError):
            pass
        hist = t.history(period="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
        return None
    except Exception:
        return None


def _fetch_historical_prices() -> pd.DataFrame | None:
    """Fallback: fetch 1.5 years of sector ETF prices from yfinance."""
    try:
        import yfinance as yf
        end = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start = (pd.Timestamp.now() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        df = yf.download(SECTOR_ETFS, start=start, end=end, progress=False, auto_adjust=True, group_by="ticker")
        if df is None or df.empty:
            return None
        out = pd.DataFrame(index=df.index)
        for t in SECTOR_ETFS:
            if isinstance(df.columns, pd.MultiIndex):
                if (t, "Close") in df.columns:
                    out[t] = df[(t, "Close")]
                elif (t,) in df.columns:
                    c = df[t]
                    out[t] = c["Close"] if isinstance(c, pd.DataFrame) and "Close" in c.columns else c.iloc[:, 0]
            elif t in df.columns:
                out[t] = df[t]
        out = out.dropna(how="all")
        return out if len(out.columns) >= 3 else None
    except Exception:
        return None


def get_block2_holdings() -> list[dict[str, Any]] | None:
    """
    Get Block2 (70%) current holdings: top 3 sectors by 12M-1M momentum
    at last rebalance. Returns list of {ticker, weight, purchase_price, current_price, pnl_pct}.
    """
    prices = _get_raw_prices()
    if prices is None or len(prices) < MOM_LOOKBACK + 50:
        prices = _fetch_historical_prices()
    if prices is None or len(prices) < MOM_LOOKBACK + 50:
        return None
    prices = prices.sort_index()
    log_p = np.log(prices)
    ret_12m = log_p - log_p.shift(MOM_LOOKBACK)
    ret_1m = log_p - log_p.shift(REBALANCE_B2)
    mom = ret_12m - ret_1m
    dates = mom.dropna(how="all").index
    if len(dates) <= MOM_LOOKBACK:
        return None
    dates = dates[MOM_LOOKBACK:]
    last_idx = (len(dates) - 1) // REBALANCE_B2 * REBALANCE_B2
    rebal_date = dates[last_idx]
    mom_row = mom.loc[rebal_date]
    if mom_row.isna().all():
        return None
    rank = mom_row.rank(ascending=False, method="min")
    top3 = rank[rank <= 3].index.tolist()
    if len(top3) < 3:
        top3 = rank.nsmallest(3).index.tolist()
    top3 = top3[:3]
    weight = 1.0 / 3
    purchase_prices = prices.loc[rebal_date]
    if isinstance(purchase_prices, pd.Series):
        purchase_prices = purchase_prices
    else:
        purchase_prices = purchase_prices.squeeze()
    out = []
    for t in top3:
        if t not in purchase_prices.index or pd.isna(purchase_prices[t]):
            continue
        buy = float(purchase_prices[t])
        cur = _get_current_price(t)
        if cur is None:
            cur = buy
        pnl = (cur / buy - 1) * 100 if buy > 0 else 0
        out.append({
            "ticker": t,
            "weight": weight * 100,
            "purchase_price": round(buy, 2),
            "current_price": round(cur, 2),
            "pnl_pct": round(pnl, 2),
        })
    return out if out else None
