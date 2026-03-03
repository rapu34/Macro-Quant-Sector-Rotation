#!/usr/bin/env python3
"""
Block 1 Extended Validation — Rebuild Block 1 using 2005–2026 dataset.

STEP 1: Generate extended raw + processed datasets (2005–latest)
STEP 2: Run Block 1 training with extended dataset (expanding walk-forward)
STEP 3: Report metrics (Sharpe, MDD, Ann Ret, Rolling 12M Sharpe, 2008/2020/2022)
STEP 4: Regime segmentation (rising/falling/flat real rate)

Deliver: experiments/outputs/block1_extended_validation.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

PPY = 252 / 20  # periods per year (20-day rebalance)
REAL_RATE_CHANGE_THRESHOLD = 0.5  # bps for rising/falling classification
ROLLING_12M_PERIODS = 12  # ~12 rebalances ≈ 12 months


EXP_DATA = Path(__file__).resolve().parent.parent / "data"


def step1_generate_extended_data() -> tuple[Path, Path]:
    """Generate raw_data_extended_2005.csv and processed_features_extended_2005.csv."""
    print("\n=== STEP 1: Generate Extended Data (2005–latest) ===")
    raw_path = DATA_DIR / "raw_data_extended_2005.csv"
    proc_path = DATA_DIR / "processed_features_extended_2005.csv"

    # Use existing experiments/data if available (avoids FRED/yfinance fetch + potential segfault)
    exp_raw = EXP_DATA / "raw_data_extended_2005.csv"
    exp_proc = EXP_DATA / "processed_features_extended_2005.csv"
    if exp_raw.exists() and exp_proc.exists():
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(exp_raw, raw_path)
        shutil.copy(exp_proc, proc_path)
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True, nrows=1)
        proc = pd.read_csv(proc_path, nrows=1)
        print(f"  Using existing extended data (copied from experiments/data/)")
        print(f"  Raw: {raw_path} ({raw.index[0]} ...)")
        print(f"  Processed: {proc_path}")
        return raw_path, proc_path

    from src.data_loader import load_all
    from src.feature_engineer import build_features
    from src.config import SECTOR_ETFS

    start_str = "2005-01-01"
    end_str = None  # latest

    print(f"  Loading macro + sector from {start_str} to latest...")
    macro_df, sector_df = load_all(start=start_str, end=end_str)
    merged = pd.concat([macro_df, sector_df], axis=1, join="inner")
    merged = merged.dropna(how="all", subset=list(macro_df.columns) + SECTOR_ETFS)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(raw_path)
    print(f"  Saved: {raw_path} ({len(merged)} rows, {merged.index.min()} to {merged.index.max()})")

    print("  Building features (20-day macro lag, no look-ahead)...")
    X, y = build_features(raw_path=raw_path)
    out_df = X.copy()
    out_df["target"] = y
    out_df = out_df.reset_index()
    out_df.to_csv(proc_path, index=False)
    print(f"  Saved: {proc_path} ({len(out_df)} rows)")

    return raw_path, proc_path


def step2_run_backtest(raw_path: Path, proc_path: Path) -> tuple:
    """Run Block 1 walk-forward backtest with extended data."""
    print("\n=== STEP 2: Block 1 Backtest (Extended Dataset) ===")

    from src.model_trainer import (
        _load_data,
        _get_feature_cols,
        _load_sentiment,
        _walk_forward_backtest,
    )
    from sklearn.preprocessing import StandardScaler

    selected_path = ROOT / "outputs" / "selected_features.json"
    df = _load_data(proc_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    # Filter to available columns only
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
    except Exception as e:
        print(f"  [HMM] {e}")

    print(f"  Features: {feature_cols}")
    print(f"  Data: {df['date'].min()} to {df['date'].max()}")

    gross, net, rebal_dates, turnover_list = _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=True,
        show_progress=True,
    )
    return net, rebal_dates, turnover_list, feature_cols


def _sharpe(rets):
    arr = np.array(rets)
    if len(arr) < 2 or arr.std() < 1e-10:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(PPY))


def _mdd(rets):
    arr = np.array(rets)
    wealth = np.ones(len(arr) + 1)
    for i, r in enumerate(arr):
        wealth[i + 1] = max(0.0, wealth[i] * (1.0 + r))
    peak = np.maximum.accumulate(wealth[1:])
    dd = np.where(peak > 1e-12, (wealth[1:] - peak) / peak, 0)
    return float(np.min(dd)) * 100


def _ann_ret(rets):
    arr = np.array(rets)
    return float(np.prod(1 + arr) ** (PPY / len(arr)) - 1) * 100 if len(arr) > 0 else 0


def _rolling_12m_sharpe(net_rets, rebal_dates):
    """Rolling 12-month (≈12 period) Sharpe."""
    arr = np.array(net_rets)
    dates = pd.DatetimeIndex(rebal_dates)
    roll = pd.Series(arr, index=dates).rolling(ROLLING_12M_PERIODS, min_periods=ROLLING_12M_PERIODS)
    roll_sh = roll.apply(lambda x: (x.mean() / x.std()) * np.sqrt(PPY) if x.std() > 1e-10 else np.nan)
    return roll_sh


def _get_subperiod_rets(net_rets, rebal_dates, start_str, end_str):
    """Extract returns for subperiod [start_str, end_str]."""
    mask = [
        pd.Timestamp(start_str) <= pd.Timestamp(d) <= pd.Timestamp(end_str)
        for d in rebal_dates
    ]
    return [net_rets[i] for i in range(len(net_rets)) if i < len(mask) and mask[i]]


def _get_real_rate_regime(raw_path: Path, rebal_dates):
    """Classify each rebalance date: rising, falling, or flat real rate."""
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if "treasury_10y" not in raw.columns or "cpi_all_urban" not in raw.columns:
        return {}
    tc10 = raw["treasury_10y"].ffill()
    cpi = raw["cpi_all_urban"].ffill()
    cpi_yoy = cpi.pct_change(252) * 100
    real_rate = tc10 - cpi_yoy
    rr_change_6m = real_rate - real_rate.shift(126)
    regimes = {}
    for d in rebal_dates:
        dts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
        try:
            idx = raw.index.get_indexer([dts], method="ffill")[0]
        except Exception:
            regimes[d] = "unknown"
            continue
        if idx < 126:
            regimes[d] = "unknown"
            continue
        chg = rr_change_6m.iloc[idx]
        if pd.isna(chg):
            regimes[d] = "unknown"
        elif chg > REAL_RATE_CHANGE_THRESHOLD:
            regimes[d] = "rising"
        elif chg < -REAL_RATE_CHANGE_THRESHOLD:
            regimes[d] = "falling"
        else:
            regimes[d] = "flat"
    return regimes


def main():
    EXP_OUT.mkdir(parents=True, exist_ok=True)

    # STEP 1
    raw_path, proc_path = step1_generate_extended_data()

    # STEP 2
    net_rets, rebal_dates, turnover_list, feature_cols = step2_run_backtest(raw_path, proc_path)

    # STEP 3 — Metrics
    print("\n=== STEP 3: Metrics ===")
    full_sharpe = _sharpe(net_rets)
    full_mdd = _mdd(net_rets)
    full_ann = _ann_ret(net_rets)
    roll_sh = _rolling_12m_sharpe(net_rets, rebal_dates)
    roll_sh_valid = roll_sh.dropna()
    roll_sh_mean = float(roll_sh_valid.mean()) if len(roll_sh_valid) > 0 else np.nan
    roll_sh_std = float(roll_sh_valid.std()) if len(roll_sh_valid) > 1 else np.nan

    subperiods = [
        ("2008", "2008-01-01", "2008-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
    ]
    subperiod_results = []
    for name, start, end in subperiods:
        sub_rets = _get_subperiod_rets(net_rets, rebal_dates, start, end)
        if len(sub_rets) >= 2:
            subperiod_results.append({
                "period": name,
                "sharpe": _sharpe(sub_rets),
                "mdd": _mdd(sub_rets),
                "ann_ret": _ann_ret(sub_rets),
                "n": len(sub_rets),
            })
        else:
            subperiod_results.append({"period": name, "sharpe": np.nan, "mdd": np.nan, "ann_ret": np.nan, "n": len(sub_rets)})

    print(f"  Net Sharpe (full): {full_sharpe:.4f}")
    print(f"  Net MDD: {full_mdd:.2f}%")
    print(f"  Annual Return: {full_ann:.2f}%")
    print(f"  Rolling 12M Sharpe: mean={roll_sh_mean:.4f}, std={roll_sh_std:.4f}")
    for r in subperiod_results:
        print(f"  {r['period']}: Sharpe={r['sharpe']:.4f}, MDD={r['mdd']:.2f}%, n={r['n']}")

    # STEP 4 — Regime segmentation
    print("\n=== STEP 4: Regime Segmentation ===")
    rr_regimes = _get_real_rate_regime(raw_path, rebal_dates)
    rate_results = {}
    for regime in ["rising", "falling", "flat"]:
        idxs = [i for i, d in enumerate(rebal_dates) if rr_regimes.get(d) == regime]
        sub_rets = [net_rets[i] for i in idxs if i < len(net_rets)]
        if len(sub_rets) >= 2:
            rate_results[regime] = {
                "sharpe": _sharpe(sub_rets),
                "mdd": _mdd(sub_rets),
                "ann_ret": _ann_ret(sub_rets),
                "n": len(sub_rets),
            }
        else:
            rate_results[regime] = {"sharpe": np.nan, "mdd": np.nan, "ann_ret": np.nan, "n": len(sub_rets)}
        print(f"  {regime}: Sharpe={rate_results[regime]['sharpe']:.4f}, n={rate_results[regime]['n']}")

    # Save rolling Sharpe for reference
    roll_df = pd.DataFrame({"date": roll_sh.index, "rolling_12m_sharpe": roll_sh.values})
    roll_df.to_csv(EXP_OUT / "block1_extended_rolling_sharpe.csv", index=False)

    # Write report
    report_path = EXP_OUT / "block1_extended_validation.md"
    report = f"""# Block 1 Extended Validation Report

> Rebuild Block 1 using extended dataset (2005–2026). Structural validation, no hyperparameter changes.

## Setup

- **Period**: 2005-01-01 to latest (data-dependent)
- **Data**: `data/raw_data_extended_2005.csv`, `data/processed_features_extended_2005.csv`
- **Walk-forward**: Expanding window
- **Rebalance**: 20 days
- **Cost**: 0.1% per side (round-trip 0.2%)
- **Features**: {', '.join(feature_cols)}
- **Risk**: HMM 2-state, institutional framework (Target Vol 15%, Kelly Cap 0.25)

## STEP 3 — Full Sample Metrics

| Metric | Value |
|--------|-------|
| **Net Sharpe** | {full_sharpe:.4f} |
| **Net MDD** | {full_mdd:.2f}% |
| **Annual Return** | {full_ann:.2f}% |
| **Rolling 12M Sharpe (mean)** | {roll_sh_mean:.4f} |
| **Rolling 12M Sharpe (std)** | {roll_sh_std:.4f} |
| **Number of rebalances** | {len(net_rets)} |

## Subperiod Performance (2008, 2020, 2022)

| Period | Net Sharpe | Net MDD | Ann Return | N |
|--------|------------|---------|------------|---|
"""
    for r in subperiod_results:
        sh_s = f"{r['sharpe']:.4f}" if not np.isnan(r['sharpe']) else "—"
        mdd_s = f"{r['mdd']:.2f}%" if not np.isnan(r['mdd']) else "—"
        ann_s = f"{r['ann_ret']:.2f}%" if not np.isnan(r['ann_ret']) else "—"
        report += f"| {r['period']} | {sh_s} | {mdd_s} | {ann_s} | {r['n']} |\n"

    if any(r["period"] == "2008" and r["n"] == 0 for r in subperiod_results):
        report += "\n*2008: No rebalances — walk-forward starts after 40% of data (first rebalance ~2013).*\n"

    report += """
## STEP 4 — Regime Segmentation (Real Rate)

| Regime | Net Sharpe | Net MDD | Ann Return | N |
|--------|------------|---------|------------|---|
"""
    for regime in ["rising", "falling", "flat"]:
        r = rate_results[regime]
        sh_s = f"{r['sharpe']:.4f}" if not np.isnan(r['sharpe']) else "—"
        mdd_s = f"{r['mdd']:.2f}%" if not np.isnan(r['mdd']) else "—"
        ann_s = f"{r['ann_ret']:.2f}%" if not np.isnan(r['ann_ret']) else "—"
        report += f"| {regime} | {sh_s} | {mdd_s} | {ann_s} | {r['n']} |\n"

    report += f"""
## Data Integrity

- **As-of-date alignment**: Macro 20-day lag applied; no look-ahead bias
- **Target**: 20-day forward return, top 3 = 1
- **Rolling Sharpe**: 12 rebalance periods (~12 months)

---
*Generated by experiments/scripts/block1_extended_validation.py*
"""

    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
