"""
Phase 2: Feature engineering for Macro-Quant Sector Rotation.

Builds features from raw macro and sector price data:
- Macro indicators with 20-day lag (no look-ahead bias)
- Log returns, 20-day rolling volatility
- Moving averages (short/long), RSI per sector
- Cross-sectional relative performance
- Target: binary (1 if sector in top 3 by 20-day forward return, else 0)

Input: raw_data.csv or (macro_df, sector_df). Output: (X, y) and processed_features.csv.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    FRED_SERIES,
    MACRO_LAG_DAYS,
    MACRO_ZSCORE_WINDOW,
    MA_LONG,
    MA_SHORT,
    RSI_PERIOD,
    SECTOR_ETFS,
    SENTIMENT_SMA_WINDOW,
    TARGET_HORIZON,
    TOP_K,
    VOLATILITY_WINDOW,
)

# Macro column names (from FRED output)
MACRO_COLS = list(FRED_SERIES.values())


def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: log(p_t / p_{t-1})."""
    return np.log(prices / prices.shift(1))


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI = 100 - 100/(1 + RS), RS = avg_gain / avg_loss."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _build_sector_features(
    sector_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build sector-level features: log return, volatility, RSI, MA, relative performance.
    Returns DataFrame with MultiIndex (date, sector) and feature columns.
    """
    log_ret = _compute_log_returns(sector_df)

    # 20-day rolling volatility (std of daily log returns)
    vol_20d = log_ret.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std()

    # 20-day cumulative log return (for relative performance)
    ret_20d = log_ret.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).sum()

    # Cross-sectional: sector return vs mean (all sectors at same date)
    ret_20d_mean = ret_20d.mean(axis=1)
    relative_ret_20d = ret_20d.sub(ret_20d_mean, axis=0)

    # RSI per sector
    rsi_dict = {ticker: _compute_rsi(sector_df[ticker], RSI_PERIOD) for ticker in sector_df.columns}
    rsi_df = pd.DataFrame(rsi_dict)

    # Sentiment Dispersion: cross-sectional std. Shrink low-news (high noise) sectors toward global mean.
    sector_sentiment_proxy = rsi_df / 100.0
    global_mean = sector_sentiment_proxy.mean(axis=1)
    roll_std = sector_sentiment_proxy.rolling(5, min_periods=1).std().fillna(0.5)
    alpha = 1.0 / (1.0 + roll_std)
    gm = np.asarray(global_mean).reshape(-1, 1)
    shrunk = alpha * sector_sentiment_proxy.values + (1 - alpha.values) * gm
    sentiment_dispersion = pd.Series(np.nanstd(shrunk, axis=1), index=rsi_df.index)

    # Moving averages
    ma_short = sector_df.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    ma_long = sector_df.rolling(MA_LONG, min_periods=MA_LONG).mean()
    ma_ratio = ma_short / ma_long

    # Target: 20-day forward log return per sector: log(p_{t+20} / p_t)
    log_prices = np.log(sector_df)
    fwd_ret_20d = log_prices.shift(-TARGET_HORIZON) - log_prices

    # Rank sectors at each date by forward return; top K = 1, else 0
    rank_desc = fwd_ret_20d.rank(axis=1, ascending=False, method="min")
    target = (rank_desc <= TOP_K).astype(int)

    # Stack to long format: (date, sector) -> one row per (date, sector)
    rows = []
    for ticker in sector_df.columns:
        row_df = pd.DataFrame(
            {
                "volatility_20d": vol_20d[ticker],
                "relative_ret_20d": relative_ret_20d[ticker],
                "rsi": rsi_df[ticker],
                "ma_ratio": ma_ratio[ticker],
                "sentiment_dispersion": sentiment_dispersion,
                "target": target[ticker],
            },
            index=log_ret.index,
        )
        row_df = row_df.reset_index()
        row_df = row_df.rename(columns={row_df.columns[0]: "date"})
        row_df["sector"] = ticker
        rows.append(row_df)

    sector_long = pd.concat(rows, axis=0, ignore_index=True)
    sector_long = sector_long.set_index(["date", "sector"])
    return sector_long


def _apply_macro_lag(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Shift macro indicators by MACRO_LAG_DAYS to prevent look-ahead bias."""
    lagged = macro_df.shift(MACRO_LAG_DAYS)
    return lagged.add_suffix("_lag20")


def _build_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 5-4: Macro feature engineering with Z-Score, MoM/YoY, Yield Curve.
    - Z-Score: (x - rolling_mean_252) / rolling_std_252
    - MoM: 21-day (1 month) pct change
    - YoY: 252-day (1 year) pct change
    - Yield Curve: treasury_10y - treasury_2y
    """
    out = {}
    w = MACRO_ZSCORE_WINDOW
    mom_days = 21
    yoy_days = 252

    for col in macro_df.columns:
        s = macro_df[col].ffill()
        # Z-Score
        roll_mean = s.rolling(w, min_periods=w // 2).mean()
        roll_std = s.rolling(w, min_periods=w // 2).std().replace(0, np.nan)
        z = (s - roll_mean) / roll_std
        out[f"{col}_zscore"] = z

        # MoM (1-month change rate)
        mom = s.pct_change(mom_days)
        out[f"{col}_mom"] = mom

        # YoY (1-year change rate)
        yoy = s.pct_change(yoy_days)
        out[f"{col}_yoy"] = yoy

    # Yield Curve: 10y - 2y (level, then Z-score)
    if "treasury_10y" in macro_df.columns and "treasury_2y" in macro_df.columns:
        tc10 = macro_df["treasury_10y"].ffill()
        tc2 = macro_df["treasury_2y"].ffill()
        yc = tc10 - tc2
        out["yield_curve_10y2y"] = yc
        roll_mean = yc.rolling(w, min_periods=w // 2).mean()
        roll_std = yc.rolling(w, min_periods=w // 2).std().replace(0, np.nan)
        out["yield_curve_10y2y_zscore"] = (yc - roll_mean) / roll_std

    # Rate shock: 20-day change (expected vs actual) for sector beta correction
    for col in ["treasury_10y", "treasury_2y", "fed_funds_rate"]:
        if col in macro_df.columns:
            s = macro_df[col].ffill()
            out[f"{col}_shock_20d"] = s - s.shift(20)

    return pd.DataFrame(out, index=macro_df.index)


def build_features(
    raw_path: Optional[str | Path] = None,
    macro_df: Optional[pd.DataFrame] = None,
    sector_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y from raw data.

    Parameters
    ----------
    raw_path : str or Path, optional
        Path to raw_data.csv (merged macro + sector from Phase 1).
    macro_df, sector_df : pd.DataFrame, optional
        Alternative: pass DataFrames directly. Ignored if raw_path is provided.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix. Index: (date, sector). Dropped rows with NaN after ffill.
    y : pd.Series
        Target (0 or 1). Same index as X.
    """
    if raw_path is not None:
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        macro_cols = [c for c in raw.columns if c in MACRO_COLS]
        sector_cols = [c for c in raw.columns if c in SECTOR_ETFS]
        macro_df = raw[macro_cols].copy()
        sector_df = raw[sector_cols].copy()
    elif macro_df is not None and sector_df is not None:
        macro_df = macro_df.copy()
        sector_df = sector_df.copy()
    else:
        raise ValueError("Provide either raw_path or both macro_df and sector_df.")

    macro_df.index = pd.to_datetime(macro_df.index)
    sector_df.index = pd.to_datetime(sector_df.index)

    # 1. Macro: Z-Score, MoM, YoY, Yield Curve, then 20-day lag (no look-ahead bias)
    macro_transformed = _build_macro_features(macro_df)
    macro_lagged = _apply_macro_lag(macro_transformed)

    # 1b. Sentiment Acceleration: load market sentiment, Sentiment_Acc = Score_t - SMA(Score, 5)
    sentiment_acc = None
    if raw_path is not None:
        start = raw.index.min().strftime("%Y-%m-%d")
        end = raw.index.max().strftime("%Y-%m-%d")
        try:
            from src.data_loader import load_market_sentiment_score
            sent_df = load_market_sentiment_score(start=start, end=end)
            if not sent_df.empty:
                score = sent_df["market_sentiment"].squeeze()
                sma5 = score.rolling(SENTIMENT_SMA_WINDOW, min_periods=1).mean()
                sentiment_acc = (score - sma5).reindex(macro_lagged.index).ffill().bfill()
        except Exception:
            pass
    elif macro_df is not None and "market_sentiment" in macro_df.columns:
        score = macro_df["market_sentiment"]
        sma5 = score.rolling(SENTIMENT_SMA_WINDOW, min_periods=1).mean()
        sentiment_acc = (score - sma5).reindex(macro_lagged.index).ffill().bfill()

    # 2. Sector features (long format)
    sector_long = _build_sector_features(sector_df)

    # 3. Merge: macro (broadcast to each sector per date) + sector features + sentiment_acc
    sector_long = sector_long.reset_index()
    macro_reset = macro_lagged.reset_index()
    macro_reset = macro_reset.rename(columns={macro_reset.columns[0]: "date"})
    merged = sector_long.merge(macro_reset, on="date", how="inner")
    if sentiment_acc is not None:
        sent_acc_df = pd.DataFrame({"date": sentiment_acc.index, "sentiment_acc": sentiment_acc.values})
        merged = merged.merge(sent_acc_df, on="date", how="left")
    merged = merged.set_index(["date", "sector"])

    # 4. Forward fill, then drop remaining NaNs
    merged = merged.ffill()
    merged = merged.dropna()

    X = merged.drop(columns=["target"])
    y = merged["target"].astype(int)

    return X, y


def run_and_save(
    raw_path: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw data, build features, save to CSV, and return (X, y).

    Parameters
    ----------
    raw_path : str or Path, optional
        Default: project_root/data/raw_data.csv
    out_path : str or Path, optional
        Default: project_root/data/processed_features.csv

    Returns
    -------
    X, y : feature matrix and target
    """
    root = Path(__file__).resolve().parent.parent
    raw_path = raw_path or root / "data" / "raw_data.csv"
    out_path = out_path or root / "data" / "processed_features.csv"

    X, y = build_features(raw_path=raw_path)

    out_df = X.copy()
    out_df["target"] = y
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path)
    print("Data saved to data/processed_features.csv")

    return X, y


if __name__ == "__main__":
    run_and_save()
