#!/usr/bin/env python3
"""
Block2 HMM Expanding-Window Variants — apples-to-apples with Block1.

Run: python experiments/scripts/block2_hmm_expanding_variants.py

Uses get_p_crisis_expanding (same as Block1) — NO full-sample hmm_regime.csv.
Two variants: HMM_DAILY (risk_mult daily) vs HMM_REBAL_ONLY (risk_mult at rebalance, held).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
REBALANCE_B2 = 21
MOMENTUM_LOOKBACK = 252


def _get_block2_rebal_dates(block2_raw: pd.Series) -> list:
    """Block2 rebalance dates (every 21 trading days from first valid)."""
    dates = block2_raw.index.sort_values()
    if len(dates) < REBALANCE_B2:
        return []
    rebal_dates = []
    for i in range(0, len(dates), REBALANCE_B2):
        rebal_dates.append(dates[i])
    return rebal_dates


def _compute_expanding_p_crisis(hmm_X: np.ndarray, hmm_dates: pd.DatetimeIndex, target_dates: list) -> pd.Series:
    """Compute p_crisis for each target date via get_p_crisis_expanding. No bfill."""
    from src.strategy_analyzer import get_p_crisis_expanding

    try:
        from tqdm import tqdm
        iterator = tqdm(target_dates, desc="P_Crisis expanding")
    except ImportError:
        iterator = target_dates

    out = {}
    for i, d in enumerate(iterator):
        if (i + 1) % 500 == 0:
            print(f"  P_Crisis: {i+1}/{len(target_dates)}")
        out[pd.Timestamp(d)] = get_p_crisis_expanding(hmm_X, hmm_dates, pd.Timestamp(d))
    return pd.Series(out)


def _build_risk_mult_daily(p_crisis: pd.Series, block2_index: pd.DatetimeIndex) -> pd.Series:
    """risk_mult = clip(1 - p_crisis, 0.5, 1.0). Lag-1. NO bfill."""
    p_aligned = p_crisis.reindex(block2_index).ffill().fillna(0.0)
    risk_mult = (1.0 - p_aligned).clip(0.5, 1.0)
    risk_mult_lag1 = risk_mult.shift(1).fillna(1.0)
    return risk_mult_lag1


def _build_risk_mult_rebal_only(
    p_crisis_at_rebal: pd.Series,
    rebal_dates: list,
    block2_index: pd.DatetimeIndex,
) -> pd.Series:
    """risk_mult at rebalance, held until next. Lag-1 for return[t] uses risk_mult[t-1]. NO bfill."""
    risk_at_rebal = (1.0 - p_crisis_at_rebal).clip(0.5, 1.0)
    risk_series = risk_at_rebal.reindex(block2_index).ffill().fillna(1.0)
    risk_lag1 = risk_series.shift(1).fillna(1.0)
    return risk_lag1


def compute_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ["AnnReturn", "AnnVol", "Sharpe", "MDD", "CVaR", "Worst1pct", "Worst5pct", "Best5pct"]}
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax()).min() - 1
    q05 = r.quantile(0.05)
    cvar = r[r <= q05].mean()
    worst1 = r[r <= r.quantile(0.01)].mean() if len(r) >= 100 else np.nan
    worst5 = r[r <= r.quantile(0.05)].mean() if len(r) >= 20 else np.nan
    best5 = r[r >= r.quantile(0.95)].mean() if len(r) >= 20 else np.nan
    return {
        "AnnReturn": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe,
        "MDD": mdd, "CVaR": cvar,
        "Worst1pct": worst1, "Worst5pct": worst5, "Best5pct": best5,
    }


def stress_corr_on_dates(x: pd.Series, y: pd.Series, cond_dates: pd.DatetimeIndex) -> float:
    common = cond_dates.intersection(x.index).intersection(y.index)
    if len(common) < 2:
        return np.nan
    return x.loc[common].corr(y.loc[common])


def main():
    print("\n=== Block2 HMM Expanding Variants ===\n")

    block1 = pd.read_csv(EXP_OUT / "true_daily_block1.csv", index_col=0, parse_dates=True).squeeze()
    block2_raw = pd.read_csv(EXP_OUT / "true_daily_block2.csv", index_col=0, parse_dates=True).squeeze()

    start = block2_raw.index.min().strftime("%Y-%m-%d")
    end = block2_raw.index.max().strftime("%Y-%m-%d")
    print(f"Date range: {start} ~ {end}")

    print("\nLoading HMM input (same as Block1)...")
    from src.strategy_analyzer import get_hmm_input_data
    hmm_X, hmm_dates = get_hmm_input_data(start=start, end=end)
    if len(hmm_X) == 0 or len(hmm_dates) == 0:
        raise RuntimeError("get_hmm_input_data returned empty. Check data/network.")

    print(f"HMM dates: {len(hmm_dates)}")

    # Compute p_crisis for each hmm_date (efficient: one per unique date)
    print("\nComputing expanding-window P_Crisis...")
    p_crisis_series = _compute_expanding_p_crisis(hmm_X, hmm_dates, hmm_dates.tolist())

    # Align to block2: ffill only, fillna(0.0). NO bfill.
    p_crisis_block2 = p_crisis_series.reindex(block2_raw.index).ffill().fillna(0.0)

    # HMM_DAILY: risk_mult updated daily
    risk_daily = _build_risk_mult_daily(p_crisis_block2, block2_raw.index)
    block2_hmm_daily = block2_raw * risk_daily

    # HMM_REBAL_ONLY: risk_mult only at rebalance
    rebal_dates = _get_block2_rebal_dates(block2_raw)
    print(f"Block2 rebalance dates: {len(rebal_dates)}")
    p_crisis_rebal = _compute_expanding_p_crisis(hmm_X, hmm_dates, [pd.Timestamp(r) for r in rebal_dates])
    risk_rebal = _build_risk_mult_rebal_only(p_crisis_rebal, rebal_dates, block2_raw.index)
    block2_hmm_rebal = block2_raw * risk_rebal

    # Combined portfolios
    equal_raw = 0.5 * block1 + 0.5 * block2_raw
    equal_daily = 0.5 * block1 + 0.5 * block2_hmm_daily
    equal_rebal = 0.5 * block1 + 0.5 * block2_hmm_rebal

    # Metrics
    variants = {
        "Block2_Raw": block2_raw,
        "HMM_DAILY": block2_hmm_daily,
        "HMM_REBAL_ONLY": block2_hmm_rebal,
    }
    single_metrics = {k: compute_metrics(v) for k, v in variants.items()}
    single_df = pd.DataFrame(single_metrics).T

    portfolio_variants = {
        "Equal_Raw": equal_raw,
        "Equal_HMM_DAILY": equal_daily,
        "Equal_HMM_REBAL_ONLY": equal_rebal,
    }
    portfolio_metrics = {k: compute_metrics(v) for k, v in portfolio_variants.items()}
    portfolio_df = pd.DataFrame(portfolio_metrics).T

    # Tail
    tail_rows = []
    for name in ["Block2_Raw", "HMM_DAILY", "HMM_REBAL_ONLY"]:
        m = single_metrics[name]
        tail_rows.append({k: m[k] for k in ["Worst1pct", "Worst5pct", "Best5pct", "CVaR"]})
    for name in ["Equal_Raw", "Equal_HMM_DAILY", "Equal_HMM_REBAL_ONLY"]:
        m = portfolio_metrics[name]
        tail_rows.append({k: m[k] for k in ["Worst1pct", "Worst5pct", "Best5pct", "CVaR"]})
    tail_df = pd.DataFrame(tail_rows, index=list(variants.keys()) + list(portfolio_variants.keys()))

    # Correlations
    n_worst = max(1, int(len(equal_raw) * 0.1))
    worst_eq_raw = equal_raw.nsmallest(n_worst).index
    worst_eq_daily = equal_daily.nsmallest(n_worst).index
    worst_eq_rebal = equal_rebal.nsmallest(n_worst).index
    worst_b2_raw = block2_raw.nsmallest(n_worst).index
    worst_b2_daily = block2_hmm_daily.nsmallest(n_worst).index
    worst_b2_rebal = block2_hmm_rebal.nsmallest(n_worst).index

    corr_data = [
        ("Full_corr_raw", block1.corr(block2_raw)),
        ("Full_corr_HMM_DAILY", block1.corr(block2_hmm_daily)),
        ("Full_corr_HMM_REBAL_ONLY", block1.corr(block2_hmm_rebal)),
        ("Stress_portfolio_worst10_raw", stress_corr_on_dates(block1, block2_raw, worst_eq_raw)),
        ("Stress_portfolio_worst10_HMM_DAILY", stress_corr_on_dates(block1, block2_hmm_daily, worst_eq_daily)),
        ("Stress_portfolio_worst10_HMM_REBAL_ONLY", stress_corr_on_dates(block1, block2_hmm_rebal, worst_eq_rebal)),
        ("Stress_block2_worst10_raw", stress_corr_on_dates(block1, block2_raw, worst_b2_raw)),
        ("Stress_block2_worst10_HMM_DAILY", stress_corr_on_dates(block1, block2_hmm_daily, worst_b2_daily)),
        ("Stress_block2_worst10_HMM_REBAL_ONLY", stress_corr_on_dates(block1, block2_hmm_rebal, worst_b2_rebal)),
    ]
    corr_df = pd.Series({k: v for k, v in corr_data})

    # Recommendation
    raw_cvar = single_metrics["Block2_Raw"]["CVaR"]
    daily_cvar = single_metrics["HMM_DAILY"]["CVaR"]
    rebal_cvar = single_metrics["HMM_REBAL_ONLY"]["CVaR"]
    raw_mdd = single_metrics["Block2_Raw"]["MDD"]
    daily_mdd = single_metrics["HMM_DAILY"]["MDD"]
    rebal_mdd = single_metrics["HMM_REBAL_ONLY"]["MDD"]
    full_raw = corr_data[0][1]
    full_daily = corr_data[1][1]
    full_rebal = corr_data[2][1]

    tail_improves_daily = (daily_cvar > raw_cvar if not np.isnan(daily_cvar) else False) or (daily_mdd > raw_mdd if not np.isnan(daily_mdd) else False)
    tail_improves_rebal = (rebal_cvar > raw_cvar if not np.isnan(rebal_cvar) else False) or (rebal_mdd > raw_mdd if not np.isnan(rebal_mdd) else False)
    stress_not_worse_daily = abs(full_daily) <= abs(full_raw) + 0.1 if not np.isnan(full_raw) else True
    stress_not_worse_rebal = abs(full_rebal) <= abs(full_raw) + 0.1 if not np.isnan(full_raw) else True

    if tail_improves_daily and stress_not_worse_daily:
        rec = "Use HMM_DAILY"
    elif tail_improves_rebal and stress_not_worse_rebal:
        rec = "Use HMM_REBAL_ONLY"
    elif tail_improves_daily:
        rec = "Use HMM_DAILY (tail improved; monitor stress correlation)"
    elif tail_improves_rebal:
        rec = "Use HMM_REBAL_ONLY (tail improved; monitor stress correlation)"
    else:
        rec = "Keep Block2 Raw (HMM did not improve tail without increasing stress correlation)"

    # Save
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": block2_hmm_daily.index, "HMM_DAILY": block2_hmm_daily.values}).to_csv(EXP_OUT / "block2_hmm_expanding_daily.csv", index=False)
    pd.DataFrame({"date": block2_hmm_rebal.index, "HMM_REBAL_ONLY": block2_hmm_rebal.values}).to_csv(EXP_OUT / "block2_hmm_expanding_rebalonly.csv", index=False)
    single_df.to_csv(EXP_OUT / "block2_hmm_expanding_metrics.csv")
    tail_df.to_csv(EXP_OUT / "block2_hmm_expanding_tail.csv")
    corr_df.to_csv(EXP_OUT / "block2_hmm_expanding_corr.csv")

    report = f"""# Block2 HMM Expanding-Window Report

Run: `python experiments/scripts/block2_hmm_expanding_variants.py`

## Summary

- **P_Crisis source**: get_p_crisis_expanding (same as Block1) — NO full-sample regime.
- **HMM_DAILY**: risk_mult updated daily. Lag-1 applied.
- **HMM_REBAL_ONLY**: risk_mult at Block2 rebalance (21d), held until next. Matches Block1 cadence.
- **No bfill**: ffill + fillna(0.0) only.

## Recommendation

**{rec}**

- Tail improves (HMM_DAILY): {tail_improves_daily}; stress preserved: {stress_not_worse_daily}
- Tail improves (HMM_REBAL_ONLY): {tail_improves_rebal}; stress preserved: {stress_not_worse_rebal}

## Single-Block Metrics

| Strategy | AnnReturn | AnnVol | Sharpe | MDD | CVaR | Worst1% | Worst5% | Best5% |
|----------|-----------|--------|--------|-----|------|--------|--------|--------|
| Block2_Raw | {single_metrics['Block2_Raw']['AnnReturn']:.4f} | {single_metrics['Block2_Raw']['AnnVol']:.4f} | {single_metrics['Block2_Raw']['Sharpe']:.4f} | {single_metrics['Block2_Raw']['MDD']:.4f} | {single_metrics['Block2_Raw']['CVaR']:.4f} | {single_metrics['Block2_Raw']['Worst1pct']:.4f} | {single_metrics['Block2_Raw']['Worst5pct']:.4f} | {single_metrics['Block2_Raw']['Best5pct']:.4f} |
| HMM_DAILY | {single_metrics['HMM_DAILY']['AnnReturn']:.4f} | {single_metrics['HMM_DAILY']['AnnVol']:.4f} | {single_metrics['HMM_DAILY']['Sharpe']:.4f} | {single_metrics['HMM_DAILY']['MDD']:.4f} | {single_metrics['HMM_DAILY']['CVaR']:.4f} | {single_metrics['HMM_DAILY']['Worst1pct']:.4f} | {single_metrics['HMM_DAILY']['Worst5pct']:.4f} | {single_metrics['HMM_DAILY']['Best5pct']:.4f} |
| HMM_REBAL_ONLY | {single_metrics['HMM_REBAL_ONLY']['AnnReturn']:.4f} | {single_metrics['HMM_REBAL_ONLY']['AnnVol']:.4f} | {single_metrics['HMM_REBAL_ONLY']['Sharpe']:.4f} | {single_metrics['HMM_REBAL_ONLY']['MDD']:.4f} | {single_metrics['HMM_REBAL_ONLY']['CVaR']:.4f} | {single_metrics['HMM_REBAL_ONLY']['Worst1pct']:.4f} | {single_metrics['HMM_REBAL_ONLY']['Worst5pct']:.4f} | {single_metrics['HMM_REBAL_ONLY']['Best5pct']:.4f} |

## Portfolio Metrics (0.5*Block1 + 0.5*Block2)

| Strategy | AnnReturn | AnnVol | Sharpe | MDD | CVaR |
|----------|-----------|--------|--------|-----|------|
| Equal_Raw | {portfolio_metrics['Equal_Raw']['AnnReturn']:.4f} | {portfolio_metrics['Equal_Raw']['AnnVol']:.4f} | {portfolio_metrics['Equal_Raw']['Sharpe']:.4f} | {portfolio_metrics['Equal_Raw']['MDD']:.4f} | {portfolio_metrics['Equal_Raw']['CVaR']:.4f} |
| Equal_HMM_DAILY | {portfolio_metrics['Equal_HMM_DAILY']['AnnReturn']:.4f} | {portfolio_metrics['Equal_HMM_DAILY']['AnnVol']:.4f} | {portfolio_metrics['Equal_HMM_DAILY']['Sharpe']:.4f} | {portfolio_metrics['Equal_HMM_DAILY']['MDD']:.4f} | {portfolio_metrics['Equal_HMM_DAILY']['CVaR']:.4f} |
| Equal_HMM_REBAL_ONLY | {portfolio_metrics['Equal_HMM_REBAL_ONLY']['AnnReturn']:.4f} | {portfolio_metrics['Equal_HMM_REBAL_ONLY']['AnnVol']:.4f} | {portfolio_metrics['Equal_HMM_REBAL_ONLY']['Sharpe']:.4f} | {portfolio_metrics['Equal_HMM_REBAL_ONLY']['MDD']:.4f} | {portfolio_metrics['Equal_HMM_REBAL_ONLY']['CVaR']:.4f} |

## Correlations (Block1 vs Block2)

| Metric | Value |
|--------|-------|
"""
    for k, v in corr_data:
        report += f"| {k} | {v:.4f} |\n"

    report += """
---
*Expanding-window P_Crisis. No look-ahead. No bfill.*
"""

    with open(EXP_OUT / "block2_hmm_expanding_report.md", "w") as f:
        f.write(report)

    print("\nSingle-block:")
    print(single_df)
    print("\nPortfolio:")
    print(portfolio_df)
    print("\nCorrelations:")
    print(corr_df)
    print(f"\nRecommendation: {rec}")
    print(f"\nSaved: {EXP_OUT / 'block2_hmm_expanding_daily.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_expanding_rebalonly.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_expanding_metrics.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_expanding_tail.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_expanding_corr.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_expanding_report.md'}")


if __name__ == "__main__":
    main()
