#!/usr/bin/env python3
"""
True Daily Portfolio Returns — ETF price-based, no period smoothing.

Replaces period-return-spread with: r_p,t = Σ(w_i,t-1 × r_i,t)
where r_i,t = (P_t / P_{t-1}) - 1 (simple return).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
_suffix = "_refresh" if os.environ.get("PIPELINE_REFRESH_MODE") == "1" else ""
DATA_DIR = ROOT / f"data{_suffix}"
EXP_DATA = Path(__file__).resolve().parent.parent / f"data{_suffix}"
EXP_OUT = Path(__file__).resolve().parent.parent / f"outputs{_suffix}"
sys.path.insert(0, str(ROOT))

from src.config import SECTOR_ETFS

COST_RATE = 0.001  # 0.1% per side
REBALANCE_B1 = 20
REBALANCE_B2 = 21
PPY = 252
ROLLING_36M_DAYS = 36 * 21


def _load_prices(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)
    cols = [c for c in SECTOR_ETFS if c in raw.columns]
    return raw[cols].ffill().dropna(how="all")


def _etf_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """STEP 1: r_i,t = (P_t / P_{t-1}) - 1 (simple return)."""
    return prices.pct_change()


def _get_block1_weight_timeline(raw_path: Path, proc_path: Path) -> list:
    """STEP 2: Block 1 rebalance dates and weights via model_trainer backtest."""
    from src.model_trainer import (
        _load_data,
        _get_feature_cols,
        _load_sentiment,
        _walk_forward_backtest,
    )
    from sklearn.preprocessing import StandardScaler

    selected_path = ROOT / f"outputs{_suffix}" / "selected_features.json"
    df = _load_data(proc_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols) < 3:
        feature_cols = [c for c in df.columns if c not in {"date", "sector", "target", "fwd_ret_20d"}][:6]

    scaler = StandardScaler()
    sentiment = _load_sentiment(raw_path)
    regime_df = pd.DataFrame()
    hmm_X, hmm_dates = np.array([]), pd.DatetimeIndex([])
    try:
        from src.strategy_analyzer import get_hmm_regime_model, get_hmm_input_data
        start = df["date"].min().strftime("%Y-%m-%d")
        end = df["date"].max().strftime("%Y-%m-%d")
        _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
        hmm_X, hmm_dates = get_hmm_input_data(start=start, end=end)
    except Exception:
        pass

    weights_log = []
    _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=True,
        show_progress=False,
        weights_log=weights_log,
    )
    return weights_log


def _get_block2_weight_timeline(raw_path: Path) -> list:
    """STEP 2: Block 2 rebalance dates and weights (12M-1M momentum)."""
    prices = _load_prices(raw_path)
    log_p = np.log(prices)
    ret_12m = log_p - log_p.shift(252)
    ret_1m = log_p - log_p.shift(21)
    mom = ret_12m - ret_1m
    dates = mom.dropna(how="all").index
    first_idx = 252
    if len(dates) <= first_idx:
        return []
    dates = dates[first_idx:]
    weight_timeline = []
    prev_weights = None

    for i in range(0, len(dates), REBALANCE_B2):
        rebal_date = dates[i]
        mom_row = mom.loc[rebal_date]
        if mom_row.isna().all():
            continue
        rank = mom_row.rank(ascending=False, method="min")
        top3 = rank[rank <= 3].index.tolist()
        if len(top3) < 3:
            top3 = rank.nsmallest(3).index.tolist()
        top3 = top3[:3]
        weights = {s: (1.0 / 3 if s in top3 else 0.0) for s in SECTOR_ETFS}

        turnover = 1.0
        if prev_weights is not None:
            turnover = sum(abs(weights.get(s, 0) - prev_weights.get(s, 0)) for s in SECTOR_ETFS)
        cost = turnover * COST_RATE * 2  # round-trip
        if prev_weights is None:
            cost = 3 * COST_RATE * (1.0 / 3) * 2  # round-trip for initial entry

        weight_timeline.append((rebal_date, weights, cost))
        prev_weights = weights

    return weight_timeline


def _build_true_daily_returns(
    weight_timeline: list,
    daily_ret: pd.DataFrame,
    is_block1: bool,
) -> pd.Series:
    """
    STEP 3: r_p,t = Σ(w_i,t-1 × r_i,t).
    Use previous day's weights. Cost on rebalance day.
    Block 1: (date, weights, cost, in_cash)
    Block 2: (date, weights, cost)
    """
    all_dates = daily_ret.index.sort_values()
    out = pd.Series(0.0, index=all_dates)
    cost_map = {pd.Timestamp(item[0]): item[2] for item in weight_timeline}
    wt_sorted = sorted(weight_timeline, key=lambda x: x[0])

    for i in range(1, len(all_dates)):
        d = all_dates[i]
        d_prev = all_dates[i - 1]

        # Weights in effect at end of d_prev = last rebalance on or before d_prev
        w_hold = {s: 0.0 for s in SECTOR_ETFS}
        for rdate, w, *rest in wt_sorted:
            if pd.Timestamp(rdate) <= d_prev:
                if is_block1 and len(rest) > 1 and rest[1]:  # in_cash
                    w_hold = {s: 0.0 for s in SECTOR_ETFS}
                else:
                    w_hold = w

        r_p = 0.0
        for s in SECTOR_ETFS:
            if s in daily_ret.columns:
                r_val = daily_ret.loc[d, s]
                if pd.notna(r_val):
                    r_p += w_hold.get(s, 0) * r_val
        out.loc[d] = r_p

        if pd.Timestamp(d) in cost_map:
            out.loc[d] -= cost_map[pd.Timestamp(d)]

    return out


def _metrics(rets: pd.Series) -> dict:
    """STEP 4: Risk metrics on daily returns."""
    arr = rets.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(arr) < 2:
        return {"ann_ret": 0, "ann_vol": 0, "sharpe": 0, "mdd": 0, "cvar": 0, "skew": 0, "kurt": 0}
    ann_ret = float(np.prod(1 + arr) ** (PPY / len(arr)) - 1) * 100
    ann_vol = float(np.std(arr) * np.sqrt(PPY)) * 100
    sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(PPY)) if np.std(arr) > 1e-12 else 0
    wealth = np.ones(len(arr) + 1)
    for j, r in enumerate(arr):
        wealth[j + 1] = max(0.0, wealth[j] * (1 + r))
    peak = np.maximum.accumulate(wealth[1:])
    dd = np.where(peak > 1e-12, (wealth[1:] - peak) / peak, 0)
    mdd = float(np.min(dd)) * 100
    n_tail = max(1, int(len(arr) * 0.05))
    cvar = float(np.mean(np.partition(arr, n_tail - 1)[:n_tail])) * 100
    skew = float(pd.Series(arr).skew()) if len(arr) > 2 else 0
    kurt = float(pd.Series(arr).kurtosis()) if len(arr) > 3 else 0
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "mdd": mdd, "cvar": cvar, "skew": skew, "kurt": kurt}


def main():
    EXP_OUT.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "raw_data_extended_2005.csv"
    proc_path = DATA_DIR / "processed_features_extended_2005.csv"
    if not raw_path.exists() and not _suffix and (EXP_DATA / "raw_data_extended_2005.csv").exists():
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(EXP_DATA / "raw_data_extended_2005.csv", raw_path)
        shutil.copy(EXP_DATA / "processed_features_extended_2005.csv", proc_path)
    if not raw_path.exists():
        raw_path = DATA_DIR / "raw_data.csv"
        proc_path = DATA_DIR / "processed_features.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Data not found: {raw_path}")

    print("\n=== True Daily Returns ===\n")

    # STEP 1
    print("STEP 1: ETF Daily Returns")
    prices = _load_prices(raw_path)
    daily_ret = _etf_daily_returns(prices)
    daily_ret = daily_ret.dropna(how="all")
    print(f"  Created daily returns: {daily_ret.shape}")

    # STEP 2
    print("\nSTEP 2: Weight Timeline")
    print("  Block 1...")
    wt_b1 = _get_block1_weight_timeline(raw_path, proc_path)
    print(f"    {len(wt_b1)} rebalances")
    print("  Block 2...")
    wt_b2 = _get_block2_weight_timeline(raw_path)
    print(f"    {len(wt_b2)} rebalances")

    # STEP 3
    print("\nSTEP 3: True Daily Portfolio Return")
    true_b1 = _build_true_daily_returns(wt_b1, daily_ret, is_block1=True)
    true_b2 = _build_true_daily_returns(wt_b2, daily_ret, is_block1=False)

    all_dates = daily_ret.index.sort_values()
    ew = 0.5 * true_b1.reindex(all_dates).ffill().bfill().fillna(0) + 0.5 * true_b2.reindex(all_dates).ffill().bfill().fillna(0)

    # Inverse Vol
    roll_vol_b1 = true_b1.rolling(ROLLING_36M_DAYS, min_periods=ROLLING_36M_DAYS).std() * np.sqrt(PPY)
    roll_vol_b2 = true_b2.rolling(ROLLING_36M_DAYS, min_periods=ROLLING_36M_DAYS).std() * np.sqrt(PPY)
    inv_vol_b1 = (1.0 / roll_vol_b1).fillna(0.5)
    inv_vol_b2 = (1.0 / roll_vol_b2).fillna(0.5)
    total_iv = inv_vol_b1 + inv_vol_b2
    w1_iv = (inv_vol_b1 / total_iv).fillna(0.5)
    w2_iv = (inv_vol_b2 / total_iv).fillna(0.5)
    w1_iv = w1_iv.reindex(all_dates).ffill().bfill().fillna(0.5)
    w2_iv = w2_iv.reindex(all_dates).ffill().bfill().fillna(0.5)
    inv_vol = w1_iv * true_b1.reindex(all_dates).ffill().bfill().fillna(0) + w2_iv * true_b2.reindex(all_dates).ffill().bfill().fillna(0)

    # STEP 4
    m_b1 = _metrics(true_b1)
    m_b2 = _metrics(true_b2)
    m_ew = _metrics(ew)
    m_iv = _metrics(inv_vol)

    # STEP 5 — Old vs True
    old_path = EXP_OUT / "portfolio_combined_returns.csv"
    comp_rows = []
    if old_path.exists():
        old = pd.read_csv(old_path, parse_dates=["date"]).set_index("date")
        for name, true_s, old_col in [
            ("Block 1", true_b1, "block1"),
            ("Block 2", true_b2, "block2"),
            ("Equal Weight", ew, "equal_weight"),
        ]:
            if old_col not in old.columns:
                continue
            old_s = old[old_col]
            common = true_s.index.intersection(old_s.index)
            if len(common) < 10:
                continue
            true_aligned = true_s.reindex(common).ffill().bfill().fillna(0)
            old_aligned = old_s.reindex(common).ffill().bfill().fillna(0)
            mo = _metrics(old_aligned)
            mt = _metrics(true_aligned)
            comp_rows.append({
                "strategy": name,
                "old_sharpe": mo["sharpe"], "true_sharpe": mt["sharpe"],
                "old_vol": mo["ann_vol"], "true_vol": mt["ann_vol"],
                "old_mdd": mo["mdd"], "true_mdd": mt["mdd"],
                "old_cvar": mo["cvar"], "true_cvar": mt["cvar"],
            })
    comp_df = pd.DataFrame(comp_rows)

    # STEP 6 — Save
    pd.DataFrame({"date": true_b1.index, "block1": true_b1.values}).to_csv(EXP_OUT / "true_daily_block1.csv", index=False)
    pd.DataFrame({"date": true_b2.index, "block2": true_b2.values}).to_csv(EXP_OUT / "true_daily_block2.csv", index=False)
    pd.DataFrame({
        "date": all_dates,
        "block1": true_b1.reindex(all_dates).ffill().bfill().fillna(0).values,
        "block2": true_b2.reindex(all_dates).ffill().bfill().fillna(0).values,
        "equal_weight": ew.reindex(all_dates).ffill().bfill().fillna(0).values,
        "inverse_vol": inv_vol.reindex(all_dates).ffill().bfill().fillna(0).values,
    }).to_csv(EXP_OUT / "true_daily_portfolio.csv", index=False)

    report = f"""# True Daily Metrics Report

> ETF price-based daily returns. No period smoothing.

## STEP 4 — Risk Metrics (True Daily)

| Strategy | Ann Ret | Ann Vol | Sharpe | MDD | CVaR (95%) | Skew | Kurtosis |
|----------|---------|---------|--------|-----|------------|------|----------|
| Block 1 | {m_b1['ann_ret']:.2f}% | {m_b1['ann_vol']:.2f}% | {m_b1['sharpe']:.4f} | {m_b1['mdd']:.2f}% | {m_b1['cvar']:.2f}% | {m_b1['skew']:.4f} | {m_b1['kurt']:.4f} |
| Block 2 | {m_b2['ann_ret']:.2f}% | {m_b2['ann_vol']:.2f}% | {m_b2['sharpe']:.4f} | {m_b2['mdd']:.2f}% | {m_b2['cvar']:.2f}% | {m_b2['skew']:.4f} | {m_b2['kurt']:.4f} |
| Equal Weight | {m_ew['ann_ret']:.2f}% | {m_ew['ann_vol']:.2f}% | {m_ew['sharpe']:.4f} | {m_ew['mdd']:.2f}% | {m_ew['cvar']:.2f}% | {m_ew['skew']:.4f} | {m_ew['kurt']:.4f} |
| Inverse Vol | {m_iv['ann_ret']:.2f}% | {m_iv['ann_vol']:.2f}% | {m_iv['sharpe']:.4f} | {m_iv['mdd']:.2f}% | {m_iv['cvar']:.2f}% | {m_iv['skew']:.4f} | {m_iv['kurt']:.4f} |

## STEP 5 — Old (Smoothed) vs True Daily

| Strategy | Metric | Old Smoothed | True Daily | Δ |
|----------|--------|--------------|------------|---|
"""
    for _, row in comp_df.iterrows():
        report += f"| {row['strategy']} | Sharpe | {row['old_sharpe']:.4f} | {row['true_sharpe']:.4f} | {row['true_sharpe'] - row['old_sharpe']:+.4f} |\n"
        report += f"| {row['strategy']} | Ann Vol | {row['old_vol']:.2f}% | {row['true_vol']:.2f}% | {row['true_vol'] - row['old_vol']:+.2f}% |\n"
        report += f"| {row['strategy']} | MDD | {row['old_mdd']:.2f}% | {row['true_mdd']:.2f}% | {row['true_mdd'] - row['old_mdd']:+.2f}% |\n"
        report += f"| {row['strategy']} | CVaR | {row['old_cvar']:.2f}% | {row['true_cvar']:.2f}% | {row['true_cvar'] - row['old_cvar']:+.2f}% |\n"

    report += """
---
*Generated by experiments/scripts/true_daily_returns.py*
*True daily: r_p,t = Σ(w_i,t-1 × r_i,t), r_i,t = (P_t/P_{t-1})-1*
"""

    with open(EXP_OUT / "true_daily_metrics_report.md", "w") as f:
        f.write(report)

    print("\nSTEP 4: Metrics")
    print(f"  Block 1: Sharpe={m_b1['sharpe']:.4f}, Vol={m_b1['ann_vol']:.2f}%")
    print(f"  Block 2: Sharpe={m_b2['sharpe']:.4f}, Vol={m_b2['ann_vol']:.2f}%")
    print(f"  EW: Sharpe={m_ew['sharpe']:.4f}, Vol={m_ew['ann_vol']:.2f}%")
    print(f"  InvVol: Sharpe={m_iv['sharpe']:.4f}, Vol={m_iv['ann_vol']:.2f}%")
    print(f"\nSaved: {EXP_OUT / 'true_daily_block1.csv'}")
    print(f"Saved: {EXP_OUT / 'true_daily_block2.csv'}")
    print(f"Saved: {EXP_OUT / 'true_daily_portfolio.csv'}")
    print(f"Saved: {EXP_OUT / 'true_daily_metrics_report.md'}")


if __name__ == "__main__":
    main()
