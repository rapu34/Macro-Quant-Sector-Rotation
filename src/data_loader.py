"""
Data loader for Macro-Quant Sector Rotation.

Fetches macroeconomic indicators from FRED API and US sector ETF prices from Yahoo Finance.
Designed for pipeline use: cleaning and alignment can be extended in downstream steps.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from src.config import FRED_SERIES, SECTOR_ETFS

# Load .env from project root (parent of src/)
_load_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env_path)


def get_fred_client() -> Fred:
    """Build FRED client using API key from environment."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not set. Add it to .env or set the environment variable."
        )
    return Fred(api_key=api_key)


def load_fred_macro(
    series_map: Optional[dict[str, str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load macroeconomic series from FRED.

    Parameters
    ----------
    series_map : dict, optional
        Mapping of FRED series ID -> output column name. Defaults to FRED_SERIES.
    start, end : str, optional
        Date strings (YYYY-MM-DD). If None, full history is requested.

    Returns
    -------
    pd.DataFrame
        Index: date (DatetimeIndex). Columns: renamed series. Drops rows with all NaN.
    """
    series_map = series_map or FRED_SERIES.copy()
    client = get_fred_client()
    frames = []

    for series_id, col_name in series_map.items():
        try:
            s = client.get_series(series_id, observation_start=start, observation_end=end)
            s.name = col_name
            frames.append(s)
        except Exception as e:
            raise RuntimeError(f"FRED series '{series_id}' failed: {e}") from e

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.dropna(how="all")


def _extract_single_close(data: pd.DataFrame) -> pd.Series:
    """Extract Close series from yf download (single ticker)."""
    if isinstance(data.columns, pd.MultiIndex):
        lev1 = data.columns.get_level_values(1)
        for c in ("Close", "Adj Close"):
            if c in lev1:
                out = data.xs(c, axis=1, level=1)
                return out.iloc[:, 0] if out.ndim == 2 else out.squeeze()
    if "Close" in data.columns:
        return data["Close"].squeeze()
    return data.iloc[:, 0].squeeze()


def _get_price_column_name(data: pd.DataFrame) -> str:
    """
    Return the price column to use. Recent yfinance versions dropped 'Adj Close';
    use 'Close' when auto_adjust=True (adjusted close), else prefer 'Adj Close'.
    """
    if isinstance(data.columns, pd.MultiIndex):
        level1 = data.columns.get_level_values(1)
        for name in ("Adj Close", "Close"):
            if name in level1:
                return name
        raise ValueError(
            "No price column found. Expected 'Adj Close' or 'Close' in columns. "
            f"Got: {level1.unique().tolist()}"
        )
    for name in ("Adj Close", "Close"):
        if name in data.columns:
            return name
    raise ValueError(
        "No price column found. Expected 'Adj Close' or 'Close'. "
        f"Got: {data.columns.tolist()}"
    )


def _extract_price_columns(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract a single price column (Adj Close or Close) into a DataFrame with ticker columns."""
    price_col = _get_price_column_name(data)
    is_multi = isinstance(data.columns, pd.MultiIndex)

    if len(tickers) == 1:
        if is_multi:
            out = data.xs(price_col, axis=1, level=1).copy()
            out.columns = [tickers[0]]
        else:
            out = data[[price_col]].copy()
            out.columns = [tickers[0]]
        return out

    if is_multi:
        return data.xs(price_col, axis=1, level=1).copy()
    # Multiple tickers requested but only one returned (e.g. others failed)
    out = data[[price_col]].copy()
    out.columns = [tickers[0]]
    return out


def load_sector_etf_prices(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load adjusted close prices for US sector ETFs from Yahoo Finance.

    Uses auto_adjust=True so 'Close' is adjusted when 'Adj Close' is not provided
    (e.g. in newer yfinance versions). Falls back to 'Close' if 'Adj Close' is missing.

    Parameters
    ----------
    tickers : list of str, optional
        ETF symbols. Defaults to SECTOR_ETFS.
    start, end : str, optional
        Date strings (YYYY-MM-DD). If None, full history is requested.

    Returns
    -------
    pd.DataFrame
        Index: date (DatetimeIndex). Columns: ticker symbols (price series).

    Raises
    ------
    RuntimeError
        When no data could be downloaded (empty result or all tickers failed).
    """
    tickers = tickers or SECTOR_ETFS.copy()
    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        progress=False,
        auto_adjust=True,
        threads=True,
    )

    if data is None or data.empty:
        raise RuntimeError(
            "No sector ETF data was downloaded. Possible causes: "
            "network/connectivity issue, Yahoo Finance rate limit or temporary block, "
            "or invalid date range. Try again later or check your connection."
        )

    try:
        result = _extract_price_columns(data, tickers)
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            "Failed to read price columns from Yahoo Finance response. "
            "The API may have changed. Details: " + str(e)
        ) from e

    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result = result.dropna(how="all")

    if result.empty:
        raise RuntimeError(
            "Sector ETF download returned no valid rows. "
            "Try a different date range or check for ticker symbols (e.g. XLK, XLF)."
        )

    missing = [t for t in tickers if t not in result.columns]
    if missing:
        import warnings
        warnings.warn(
            f"Some tickers had no data and were omitted: {missing}. "
            "You may be rate-limited or the symbols may be invalid.",
            UserWarning,
            stacklevel=2,
        )

    return result


def load_cnn_fear_greed_index(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Attempt to fetch CNN Fear & Greed Index. Returns None if unavailable.
    (CNN has no official free API; often requires scraping or paid source.)
    """
    try:
        import requests
        # Alternative: fear greed api (e.g. alternative.me) - not always stable
        url = "https://api.alternative.me/fng/?limit=0"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if "data" in data:
                rows = [{"date": d["timestamp"], "fear_greed": float(d["value"])} for d in data["data"]]
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], unit="s").dt.normalize()
                df = df.set_index("date").sort_index()
                if start:
                    df = df[df.index >= pd.Timestamp(start)]
                if end:
                    df = df[df.index <= pd.Timestamp(end)]
                return df
    except Exception:
        pass
    return None


def load_market_sentiment_score(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Market Sentiment Score (0~100). CNN Fear & Greed if available;
    otherwise fallback: VIX + SPY MA deviation.
    """
    fg = load_cnn_fear_greed_index(start=start, end=end)
    if fg is not None and not fg.empty:
        fg = fg.rename(columns={"fear_greed": "market_sentiment"})
        return fg

    # Fallback: VIX + SPY MA20 deviation -> Market Sentiment Score 0~100
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if vix.empty or spy.empty:
        return pd.DataFrame(columns=["market_sentiment"])
    try:
        spy_prices = _extract_single_close(spy)
        vix_vals = _extract_single_close(vix)
    except (KeyError, ValueError):
        return pd.DataFrame(columns=["market_sentiment"])
    ma20 = spy_prices.rolling(20, min_periods=5).mean()
    spy_dev = (spy_prices / ma20 - 1)
    common = spy_prices.index.intersection(vix_vals.index)
    vix_aligned = vix_vals.reindex(common).ffill().bfill()
    spy_dev_aligned = spy_dev.reindex(common).ffill().bfill()
    vix_score = np.clip(100 - (vix_aligned - 12).astype(float) * 2, 0, 100)
    spy_score = np.clip(50 + spy_dev_aligned.astype(float) * 500, 0, 100)
    sentiment = (vix_score + spy_score) / 2
    return sentiment.to_frame("market_sentiment").dropna(how="all")


def get_sector_news(tickers: Optional[list[str]] = None, max_per_ticker: int = 5) -> dict[str, list[dict]]:
    """
    Fetch latest news headlines for sector ETFs from Yahoo Finance.
    Returns {ticker: [{title, link, publisher}, ...]}.
    """
    tickers = tickers or SECTOR_ETFS.copy()
    result = {}
    for t in tickers:
        try:
            tkr = yf.Ticker(t)
            raw = getattr(tkr, "news", [])
            if callable(raw):
                raw = raw(max_per_ticker) if raw else []
            items = (raw or [])[:max_per_ticker]
            parsed = []
            for n in items:
                if isinstance(n, dict):
                    parsed.append({
                        "title": str(n.get("title", n.get("link", "")))[:200],
                        "link": str(n.get("link", "")),
                        "publisher": str(n.get("publisher", "")),
                    })
                else:
                    parsed.append({"title": str(n)[:200], "link": "", "publisher": ""})
            result[t] = parsed
        except Exception:
            result[t] = []
    return result


def load_all(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both macro and sector ETF data for the same date range.

    Parameters
    ----------
    start, end : str, optional
        Date strings (YYYY-MM-DD).

    Returns
    -------
    macro_df : pd.DataFrame
        FRED macro indicators (date index).
    sector_df : pd.DataFrame
        Sector ETF adjusted close prices (date index).
    """
    macro_df = load_fred_macro(start=start, end=end)
    sector_df = load_sector_etf_prices(start=start, end=end)
    return macro_df, sector_df


def _save_daily_raw(merged: pd.DataFrame, daily_dir: Path) -> None:
    """
    Save each date's row to data_refresh/daily/raw_YYYY-MM-DD.csv.
    Used in refresh mode for date-based storage (audit, incremental).
    """
    daily_dir.mkdir(parents=True, exist_ok=True)
    merged.index = pd.to_datetime(merged.index)
    for dt in merged.index:
        date_str = dt.strftime("%Y-%m-%d")
        row_path = daily_dir / f"raw_{date_str}.csv"
        row_df = merged.loc[[dt]]
        row_df.to_csv(row_path)


def load_raw_from_daily(daily_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load raw_data by concatenating all raw_YYYY-MM-DD.csv in daily_dir.
    Returns None if no daily files exist.
    """
    if not daily_dir.exists():
        return None
    files = sorted(daily_dir.glob("raw_*.csv"))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, axis=0).sort_index().drop_duplicates()


if __name__ == "__main__":
    import os
    from datetime import datetime, timedelta

    root = Path(__file__).resolve().parent.parent
    _refresh = os.environ.get("PIPELINE_REFRESH_MODE") == "1"
    _data_dir = root / ("data_refresh" if _refresh else "data")
    _daily_dir = _data_dir / "daily" if _refresh else None

    end_d = datetime.now() + timedelta(days=3)  # +3 days: yfinance end exclusive + typical 1–2 day delay
    start_d = end_d - timedelta(days=365 * 21)  # ~21 years for extended-like range
    start_str = start_d.strftime("%Y-%m-%d")
    end_str = end_d.strftime("%Y-%m-%d")

    try:
        macro, sector = load_all(start=start_str, end=end_str)
        merged = pd.concat([macro, sector], axis=1, join="inner")

        out_path = _data_dir / "raw_data.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path)
        print(f"Data saved to {out_path}")

        if _refresh and _daily_dir is not None:
            _save_daily_raw(merged, _daily_dir)
            n_days = len(merged)
            print(f"Daily snapshots saved to {_daily_dir}/raw_YYYY-MM-DD.csv ({n_days} files)")
    except RuntimeError as e:
        print("Error:", e)
        raise
