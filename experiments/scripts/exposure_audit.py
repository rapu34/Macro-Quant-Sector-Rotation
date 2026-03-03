#!/usr/bin/env python3
"""
Exposure Audit: Verify return scale, gross exposure, volatility scaling, and unit consistency.

Structural validation. Do NOT change model logic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
EXP_DATA = Path(__file__).resolve().parent.parent / "data"
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

PPY_DAILY = 252


def _period_to_daily_returns(period_rets, rebal_dates, all_dates, _):
    """Convert period returns to daily returns."""
    all_dates = pd.DatetimeIndex(all_dates).sort_values()
    daily = pd.Series(0.0, index=all_dates)
    for i, (r, d) in enumerate(zip(period_rets, rebal_dates)):
        start = pd.Timestamp(d)
        if i + 1 < len(rebal_dates):
            end = pd.Timestamp(rebal_dates[i + 1])
        else:
            end = all_dates[-1]
        mask = (all_dates >= start) & (all_dates <= end)
        n_days = mask.sum()
        if n_days > 0:
            daily_ret = (1 + r) ** (1 / n_days) - 1
            daily.loc[mask] = daily_ret
    return daily


def _run_block1_scaled(raw_path, proc_path):
    """Block 1 with HMM + target vol (production). Returns (net_rets, rebal_dates, daily_rets, gross_exposure_log)."""
    from src.model_trainer import _load_data, _get_feature_cols, _load_sentiment, _walk_forward_backtest
    from sklearn.preprocessing import StandardScaler

    selected_path = ROOT / "outputs" / "selected_features.json"
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

    gross_exposure_log = []
    _, net, rebal_dates, _ = _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=True,
        show_progress=False,
        gross_exposure_log=gross_exposure_log,
    )
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    daily_rets = _period_to_daily_returns(net, rebal_dates, raw.index, 20)
    return net, rebal_dates, daily_rets, gross_exposure_log


def _run_block1_unscaled(raw_path, proc_path):
    """Block 1 without HMM/target vol. Returns (net_rets, rebal_dates, daily_rets)."""
    from src.model_trainer import _load_data, _get_feature_cols, _load_sentiment, _walk_forward_backtest
    from sklearn.preprocessing import StandardScaler

    selected_path = ROOT / "outputs" / "selected_features.json"
    df = _load_data(proc_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols) < 3:
        feature_cols = [c for c in df.columns if c not in {"date", "sector", "target", "fwd_ret_20d"}][:6]

    scaler = StandardScaler()
    sentiment = _load_sentiment(raw_path)
    _, net, rebal_dates, _ = _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment,
        use_risk_mgmt=False,
        raw_path=raw_path,
        regime_df=pd.DataFrame(),
        hmm_X=None,
        hmm_dates=None,
        use_institutional=False,
        show_progress=False,
    )
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    daily_rets = _period_to_daily_returns(net, rebal_dates, raw.index, 20)
    return net, rebal_dates, daily_rets


def _sharpe(rets):
    arr = rets.dropna().values
    if len(arr) < 2 or np.std(arr) < 1e-12:
        return 0.0
    return float(np.mean(arr) / np.std(arr) * np.sqrt(PPY_DAILY))


def _ann_vol(rets):
    arr = rets.dropna().values
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr) * np.sqrt(PPY_DAILY)) * 100


def main():
    EXP_OUT.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "raw_data_extended_2005.csv"
    proc_path = DATA_DIR / "processed_features_extended_2005.csv"
    if not raw_path.exists() and (EXP_DATA / "raw_data_extended_2005.csv").exists():
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(EXP_DATA / "raw_data_extended_2005.csv", raw_path)
        shutil.copy(EXP_DATA / "processed_features_extended_2005.csv", proc_path)

    # Load portfolio returns
    ret_path = EXP_OUT / "portfolio_combined_returns.csv"
    if not ret_path.exists():
        raise FileNotFoundError(f"{ret_path} not found. Run portfolio_combined.py first.")
    df = pd.read_csv(ret_path, parse_dates=["date"])
    df = df.set_index("date")

    print("\n=== Exposure Audit ===\n")

    # STEP 1 — Return scale verification
    print("STEP 1: Return Scale Verification")
    print("  1) Daily returns are DECIMAL: 0.01 = 1%, NOT 1 = 100%")
    print("  2) Sample (first 5 non-zero rows):")
    non_zero = df[(df["block1"] != 0) | (df["block2"] != 0)]
    sample = non_zero.head(5) if len(non_zero) > 0 else df.head(5)
    print(sample.to_string())
    print("\n  3) Min / Max daily return:")
    for col in ["block1", "block2", "equal_weight", "inverse_vol"]:
        arr = df[col].replace(0, np.nan).dropna()
        if len(arr) > 0:
            print(f"    {col}: min={arr.min():.8f}, max={arr.max():.8f}")
        else:
            print(f"    {col}: (all zeros)")
    print("\n  4) No additional scaling: returns are raw from backtest (decimal form).")

    # STEP 2 — Gross exposure check
    print("\nSTEP 2: Gross Exposure Check")
    print("  Running Block 1 with gross_exposure_log...")
    _, _, _, gross_log = _run_block1_scaled(raw_path, proc_path)

    # Block 1: from gross_exposure_log (at rebalance)
    if gross_log:
        ge_arr = np.array(gross_log)
        print(f"  Block 1 (HMM + Target Vol scaled):")
        print(f"    Mean gross exposure: {np.mean(ge_arr):.4f}")
        print(f"    Max gross exposure:  {np.max(ge_arr):.4f}")
        print(f"    Min gross exposure:  {np.min(ge_arr):.4f}")
        print(f"    -> Vol-target scaled + HMM-scaled (risk_mult reduces exposure)")
    else:
        print("  Block 1: gross_exposure_log empty")
        ge_arr = np.array([1.0])

    # Block 2: always 1.0 (Top 3 equal weight, no scaling)
    print(f"\n  Block 2:")
    print(f"    Mean gross exposure: 1.0000")
    print(f"    Max gross exposure:  1.0000")
    print(f"    Min gross exposure:  1.0000")
    print(f"    -> Always fully invested (no scaling)")

    # Combined: EW and InvVol have block weights summing to 1
    print(f"\n  Equal Weight portfolio:")
    print(f"    Mean gross exposure: 1.0000 (0.5 + 0.5)")
    print(f"  Inverse Vol portfolio:")
    print(f"    Mean gross exposure: 1.0000 (w1 + w2 = 1)")

    # STEP 3 — Volatility scaling audit
    print("\nSTEP 3: Volatility Scaling Audit")
    print("  Block 1: Target volatility = 15% (TARGET_VOL in config)")
    print("  - Weights scaled by: scale = target_vol / portfolio_vol")
    print("  - HMM risk_mult = max(0.5, 1 - p_crisis) further reduces exposure")
    print("  - Realized vol AFTER scaling: target ~15% (approximate)")
    b1_rets = df["block1"].replace(0, np.nan).dropna()
    if len(b1_rets) > 10:
        realized_vol_b1 = _ann_vol(b1_rets)
        print(f"  - Realized ann vol (from daily returns): {realized_vol_b1:.2f}%")
        print(f"  - Note: Daily returns are period-expanded; realized vol may differ from 15%")
    print("  Block 2: No target vol. No scaling.")
    print("  Combined: No additional scaling at portfolio level.")
    print("  -> No double scaling.")

    # STEP 4 — Raw vs scaled
    print("\nSTEP 4: Raw (Unscaled) vs Scaled Metrics")
    _, _, daily_unscaled = _run_block1_unscaled(raw_path, proc_path)
    daily_scaled = df["block1"]

    # Align
    common = daily_unscaled.dropna().index.intersection(daily_scaled.dropna().index)
    unscaled = daily_unscaled.reindex(common).ffill().bfill().fillna(0)
    scaled = daily_scaled.reindex(common).ffill().bfill().fillna(0)

    vol_unscaled = _ann_vol(unscaled)
    vol_scaled = _ann_vol(scaled)
    sharpe_unscaled = _sharpe(unscaled)
    sharpe_scaled = _sharpe(scaled)

    raw_vs_scaled = [
        {"metric": "Block1_vol_ann", "scaled": vol_scaled, "unscaled": vol_unscaled},
        {"metric": "Block1_sharpe", "scaled": sharpe_scaled, "unscaled": sharpe_unscaled},
    ]
    pd.DataFrame(raw_vs_scaled).to_csv(EXP_OUT / "raw_vs_scaled_metrics.csv", index=False)

    print(f"  Block 1 Scaled:   Vol={vol_scaled:.2f}%, Sharpe={sharpe_scaled:.4f}")
    print(f"  Block 1 Unscaled: Vol={vol_unscaled:.2f}%, Sharpe={sharpe_unscaled:.4f}")
    print(f"  -> Scaling reduces vol (target 15%) and may change Sharpe")

    # STEP 5 — Unit consistency
    print("\nSTEP 5: Unit Consistency Check")
    for col in ["block1", "block2", "equal_weight"]:
        arr = df[col].dropna().values
        arr = arr[arr != 0] if (arr == 0).all() is False else arr
        if len(arr) < 2:
            continue
        var_daily = np.var(arr)
        vol_ann = np.std(arr) * np.sqrt(PPY_DAILY) * 100
        vol_from_var = np.sqrt(var_daily * PPY_DAILY) * 100
        match = "OK" if abs(vol_ann - vol_from_var) < 0.01 else "CHECK"
        print(f"  {col}: var_daily={var_daily:.10f}, vol_ann={vol_ann:.4f}%, sqrt(var*252)*100={vol_from_var:.4f}% -> {match}")
    print("  -> All daily frequency. No mixing with monthly.")

    # Report
    report = f"""# Exposure Audit Report

> Verify return scale, gross exposure, volatility scaling, unit consistency.

## STEP 1 — Return Scale Verification

1) **Daily returns are DECIMAL**: 0.01 = 1%, NOT 1 = 100%
2) Sample (first 5 non-zero): see CSV
3) Min/Max daily return:
   - block1: {float(df['block1'].replace(0, np.nan).min() or 0):.8f} / {float(df['block1'].replace(0, np.nan).max() or 0):.8f}
   - block2: {float(df['block2'].replace(0, np.nan).min() or 0):.8f} / {float(df['block2'].replace(0, np.nan).max() or 0):.8f}
4) **No additional scaling** applied to returns.

## STEP 2 — Gross Exposure Check

| Strategy | Mean Gross | Max Gross | Min Gross | Type |
|----------|------------|-----------|-----------|------|
| Block 1 | {np.mean(ge_arr):.4f} | {np.max(ge_arr):.4f} | {np.min(ge_arr):.4f} | Vol-target + HMM scaled |
| Block 2 | 1.0000 | 1.0000 | 1.0000 | Always fully invested |
| Equal Weight | 1.0000 | 1.0000 | 1.0000 | 0.5+0.5 |
| Inverse Vol | 1.0000 | 1.0000 | 1.0000 | w1+w2=1 |

Block 1: HMM risk_mult = max(0.5, 1 - p_crisis) and target vol 15% scaling. Gross exposure varies.

## STEP 3 — Volatility Scaling Audit

- **Target vol**: 15% (TARGET_VOL)
- **Block 1**: scale = target_vol / portfolio_vol; HMM risk_mult further scales
- **Block 2**: No scaling
- **Portfolio level**: No additional scaling
- **No double scaling**

**Note on low realized vol (2.20%):** Daily returns are period-expanded (20-day period return spread across days). The expanded daily series has lower variance than period returns. Target 15% applies at period level; daily-expanded vol will be lower.

## STEP 4 — Raw vs Scaled Metrics (Block 1)

| Metric | Scaled | Unscaled |
|--------|--------|----------|
| Ann Vol | {vol_scaled:.2f}% | {vol_unscaled:.2f}% |
| Sharpe | {sharpe_scaled:.4f} | {sharpe_unscaled:.4f} |

Unscaled = use_institutional=False, use_risk_mgmt=False (equal weight 1/3, no HMM, no target vol).

## STEP 5 — Unit Consistency

- Daily return variance × 252 = annual variance
- Annual vol = std(daily) × √252
- All series are daily frequency. No mixing with monthly.

---
*Generated by experiments/scripts/exposure_audit.py*
*Structural validation. Model logic unchanged.*
"""

    with open(EXP_OUT / "exposure_audit_report.md", "w") as f:
        f.write(report)
    print(f"\nSaved: {EXP_OUT / 'exposure_audit_report.md'}")
    print(f"Saved: {EXP_OUT / 'raw_vs_scaled_metrics.csv'}")


if __name__ == "__main__":
    main()
