"""Intraday data for Today's P&L chart. Fetches SPY 5m bars via yfinance."""

from datetime import datetime

import pandas as pd

try:
    from zoneinfo import ZoneInfo
    US_ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    US_ET = pytz.timezone("America/New_York")

MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)


def is_market_open() -> bool:
    """True if US market is open (Mon–Fri 9:30–16:00 ET)."""
    now = datetime.now(US_ET)
    if now.weekday() >= 5:  # Sat=5, Sun=6
        return False
    h, m = now.hour, now.minute
    if h < MARKET_OPEN[0] or (h == MARKET_OPEN[0] and m < MARKET_OPEN[1]):
        return False
    if h > MARKET_CLOSE[0] or (h == MARKET_CLOSE[0] and m >= MARKET_CLOSE[1]):
        return False
    return True


def fetch_intraday_spy() -> pd.DataFrame | None:
    """
    Fetch SPY 5-minute bars for today. Returns DataFrame with DatetimeIndex and 'Close'.
    Returns None on error.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        df = yf.download("SPY", interval="5m", period="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 3]
        return close.to_frame("Close")
    except Exception:
        return None


def compute_today_cumulative_return(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Compute cumulative return from market open. Returns DataFrame with time and cum_return.
    """
    if df is None or len(df) < 2:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(US_ET)
    else:
        df.index = df.index.tz_localize(US_ET, ambiguous="infer")
    today = datetime.now(US_ET).date()
    mask = df.index.date == today
    sub = df.loc[mask].sort_index()
    if len(sub) < 2:
        return None
    close_col = "Close" if "Close" in sub.columns else sub.columns[0]
    open_price = float(sub[close_col].iloc[0])
    sub = sub.copy()
    sub["cum_return"] = (sub[close_col].astype(float) / open_price) - 1
    return sub[["cum_return"]].reset_index()

