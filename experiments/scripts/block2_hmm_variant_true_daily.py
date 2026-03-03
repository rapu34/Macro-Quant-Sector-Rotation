#!/usr/bin/env python3
"""
DEPRECATED: Uses full-sample hmm_regime.csv (look-ahead). Use block2_hmm_expanding_variants.py instead.

Block2 Raw vs Block2 + HMM — apples-to-apples with Block1.

Run: python experiments/scripts/block2_hmm_expanding_variants.py  # preferred

1) Block2 Raw vs Block2 + HMM (single-block)
2) (Block1 + Block2 Raw) vs (Block1 + Block2 HMM) (combined portfolio)

All metrics on TRUE DAILY PnL. Extended period 2005-01-03 ~ 2026-01-28.
Uses EXACT SAME HMM regime and risk_mult as Block1: clip(1 - P_Crisis, 0.5, 1.0).
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS = ROOT / "outputs"

HMM_PATHS = [
    EXP_OUT / "hmm_regime.csv",
    OUTPUTS / "hmm_regime.csv",
]
P_CRISIS_COLS = ["P_Crisis", "p_crisis", "prob_crisis", "P_crisis"]


def _load_regime() -> pd.DataFrame:
    """Load regime with P_Crisis-like column."""
    for p in HMM_PATHS:
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["date"])
        if "date" not in df.columns:
            continue
        df = df.set_index("date")
        for col in P_CRISIS_COLS:
            if col in df.columns:
                return df[[col]].rename(columns={col: "P_Crisis"})
    raise FileNotFoundError("hmm_regime.csv with P_Crisis not found")


def _build_risk_mult(block2_index: pd.DatetimeIndex, regime_df: pd.DataFrame) -> pd.Series:
    """risk_mult = clip(1 - P_Crisis, 0.5, 1.0). Lag-1 for no look-ahead."""
    p_crisis = regime_df["P_Crisis"].reindex(block2_index).ffill().fillna(0.0)  # no bfill
    risk_mult = (1.0 - p_crisis).clip(0.5, 1.0)
    risk_mult_lag1 = risk_mult.shift(1).fillna(1.0)
    return risk_mult_lag1


def compute_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ["AnnReturn", "AnnVol", "Sharpe", "MDD", "CVaR", "Worst1pct", "Worst5pct", "Best5pct", "Skew", "Kurtosis"]}
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
    skew = r.skew() if len(r) > 2 else np.nan
    kurt = r.kurtosis() if len(r) > 3 else np.nan
    return {
        "AnnReturn": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MDD": mdd,
        "CVaR": cvar,
        "Worst1pct": worst1,
        "Worst5pct": worst5,
        "Best5pct": best5,
        "Skew": skew,
        "Kurtosis": kurt,
    }


def stress_corr_on_dates(x: pd.Series, y: pd.Series, cond_dates: pd.DatetimeIndex) -> float:
    common = cond_dates.intersection(x.index).intersection(y.index)
    if len(common) < 2:
        return np.nan
    return x.loc[common].corr(y.loc[common])


def main():
    # A) Load daily returns
    block1 = pd.read_csv(EXP_OUT / "true_daily_block1.csv", index_col=0, parse_dates=True).squeeze()
    block2_raw = pd.read_csv(EXP_OUT / "true_daily_block2.csv", index_col=0, parse_dates=True).squeeze()

    # B) Build risk_mult (same as Block1)
    regime_df = _load_regime()
    risk_mult_lag1 = _build_risk_mult(block2_raw.index, regime_df)

    # C) Block2 HMM
    block2_hmm = block2_raw * risk_mult_lag1

    # D) Combined portfolios
    equal_raw = 0.5 * block1 + 0.5 * block2_raw
    equal_hmm = 0.5 * block1 + 0.5 * block2_hmm

    # --- Single-block metrics ---
    single_metrics = {
        "Block2_Raw": compute_metrics(block2_raw),
        "Block2_HMM": compute_metrics(block2_hmm),
    }
    single_df = pd.DataFrame(single_metrics).T

    # --- Portfolio metrics ---
    portfolio_metrics = {
        "Equal_Raw": compute_metrics(equal_raw),
        "Equal_Block2_HMM": compute_metrics(equal_hmm),
    }
    portfolio_df = pd.DataFrame(portfolio_metrics).T

    # --- Tail comparison ---
    tail_rows = []
    for name, m in [("Block2_Raw", single_metrics["Block2_Raw"]), ("Block2_HMM", single_metrics["Block2_HMM"]),
                   ("Equal_Raw", portfolio_metrics["Equal_Raw"]), ("Equal_Block2_HMM", portfolio_metrics["Equal_Block2_HMM"])]:
        tail_rows.append({k: m[k] for k in ["Worst1pct", "Worst5pct", "Best5pct", "CVaR"]})
    tail_df = pd.DataFrame(tail_rows, index=["Block2_Raw", "Block2_HMM", "Equal_Raw", "Equal_Block2_HMM"])

    # --- Stress correlations ---
    n_worst = max(1, int(len(equal_raw) * 0.1))
    worst_eq_raw = equal_raw.nsmallest(n_worst).index
    worst_eq_hmm = equal_hmm.nsmallest(n_worst).index
    worst_b2_raw = block2_raw.nsmallest(n_worst).index
    worst_b2_hmm = block2_hmm.nsmallest(n_worst).index

    corr_rows = [
        ("Full_corr_raw", block1.corr(block2_raw)),
        ("Full_corr_hmm", block1.corr(block2_hmm)),
        ("Stress_portfolio_worst10_raw", stress_corr_on_dates(block1, block2_raw, worst_eq_raw)),
        ("Stress_portfolio_worst10_hmm", stress_corr_on_dates(block1, block2_hmm, worst_eq_hmm)),
        ("Stress_block2_worst10_raw", stress_corr_on_dates(block1, block2_raw, worst_b2_raw)),
        ("Stress_block2_worst10_hmm", stress_corr_on_dates(block1, block2_hmm, worst_b2_hmm)),
    ]
    corr_df = pd.Series({k: v for k, v in corr_rows})

    # --- Recommendation ---
    raw_cvar = single_metrics["Block2_Raw"]["CVaR"]
    hmm_cvar = single_metrics["Block2_HMM"]["CVaR"]
    raw_mdd = single_metrics["Block2_Raw"]["MDD"]
    hmm_mdd = single_metrics["Block2_HMM"]["MDD"]
    full_corr_raw = corr_rows[0][1]
    full_corr_hmm = corr_rows[1][1]
    tail_improves = (hmm_cvar > raw_cvar if not np.isnan(hmm_cvar) and not np.isnan(raw_cvar) else False) or (
        hmm_mdd > raw_mdd if not np.isnan(hmm_mdd) and not np.isnan(raw_mdd) else False
    )
    div_preserved = abs(full_corr_hmm) <= abs(full_corr_raw) + 0.1 if not np.isnan(full_corr_raw) else True
    recommendation = (
        "Use Block2-HMM" if tail_improves and div_preserved
        else "Use Block2-HMM (tail improved, monitor diversification)" if tail_improves
        else "Keep Block2 Raw (HMM did not improve tail)" if not tail_improves
        else "Keep Block2 Raw"
    )

    # --- Save ---
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    single_df.to_csv(EXP_OUT / "block2_hmm_single_metrics.csv")
    portfolio_df.to_csv(EXP_OUT / "block2_hmm_portfolio_metrics.csv")
    tail_df.to_csv(EXP_OUT / "block2_hmm_tail_comparison.csv")
    corr_df.to_csv(EXP_OUT / "block2_hmm_corr_comparison.csv")

    report = f"""# Block2-HMM Variant Report (True Daily PnL)

Run: `python experiments/scripts/block2_hmm_variant_true_daily.py`

## Summary

- **Block2 Raw vs Block2 HMM**: HMM applies `risk_mult = clip(1 - P_Crisis, 0.5, 1.0)` (lag-1), identical to Block1.
- **Tail improvement**: CVaR Raw={raw_cvar:.4f} vs HMM={hmm_cvar:.4f}; MDD Raw={raw_mdd:.2%} vs HMM={hmm_mdd:.2%}.
- **Stress correlation**: Full corr(Block1, Block2) Raw={full_corr_raw:.4f}, HMM={full_corr_hmm:.4f}.

## Recommendation

**{recommendation}**

- Tail improves: {tail_improves}
- Diversification preserved: {div_preserved}

## Single-Block Metrics

| Strategy | AnnReturn | AnnVol | Sharpe | MDD | CVaR | Worst1% | Worst5% | Best5% | Skew | Kurtosis |
|----------|----------|--------|--------|-----|------|---------|---------|--------|------|----------|
| Block2_Raw | {single_metrics['Block2_Raw']['AnnReturn']:.4f} | {single_metrics['Block2_Raw']['AnnVol']:.4f} | {single_metrics['Block2_Raw']['Sharpe']:.4f} | {single_metrics['Block2_Raw']['MDD']:.4f} | {single_metrics['Block2_Raw']['CVaR']:.4f} | {single_metrics['Block2_Raw']['Worst1pct']:.4f} | {single_metrics['Block2_Raw']['Worst5pct']:.4f} | {single_metrics['Block2_Raw']['Best5pct']:.4f} | {single_metrics['Block2_Raw']['Skew']:.4f} | {single_metrics['Block2_Raw']['Kurtosis']:.4f} |
| Block2_HMM | {single_metrics['Block2_HMM']['AnnReturn']:.4f} | {single_metrics['Block2_HMM']['AnnVol']:.4f} | {single_metrics['Block2_HMM']['Sharpe']:.4f} | {single_metrics['Block2_HMM']['MDD']:.4f} | {single_metrics['Block2_HMM']['CVaR']:.4f} | {single_metrics['Block2_HMM']['Worst1pct']:.4f} | {single_metrics['Block2_HMM']['Worst5pct']:.4f} | {single_metrics['Block2_HMM']['Best5pct']:.4f} | {single_metrics['Block2_HMM']['Skew']:.4f} | {single_metrics['Block2_HMM']['Kurtosis']:.4f} |

## Portfolio Metrics

| Strategy | AnnReturn | AnnVol | Sharpe | MDD | CVaR |
|----------|----------|--------|--------|-----|------|
| Equal_Raw | {portfolio_metrics['Equal_Raw']['AnnReturn']:.4f} | {portfolio_metrics['Equal_Raw']['AnnVol']:.4f} | {portfolio_metrics['Equal_Raw']['Sharpe']:.4f} | {portfolio_metrics['Equal_Raw']['MDD']:.4f} | {portfolio_metrics['Equal_Raw']['CVaR']:.4f} |
| Equal_Block2_HMM | {portfolio_metrics['Equal_Block2_HMM']['AnnReturn']:.4f} | {portfolio_metrics['Equal_Block2_HMM']['AnnVol']:.4f} | {portfolio_metrics['Equal_Block2_HMM']['Sharpe']:.4f} | {portfolio_metrics['Equal_Block2_HMM']['MDD']:.4f} | {portfolio_metrics['Equal_Block2_HMM']['CVaR']:.4f} |

## Stress Correlations (Block1 vs Block2)

| Metric | Value |
|--------|-------|
| Full_corr_raw | {corr_rows[0][1]:.4f} |
| Full_corr_hmm | {corr_rows[1][1]:.4f} |
| Stress_portfolio_worst10_raw | {corr_rows[2][1]:.4f} |
| Stress_portfolio_worst10_hmm | {corr_rows[3][1]:.4f} |
| Stress_block2_worst10_raw | {corr_rows[4][1]:.4f} |
| Stress_block2_worst10_hmm | {corr_rows[5][1]:.4f} |

---
*Generated by block2_hmm_variant_true_daily.py. True daily PnL, no smoothing.*
"""

    with open(EXP_OUT / "block2_hmm_variant_report.md", "w") as f:
        f.write(report)

    print("Block2 Raw vs Block2 HMM (single):")
    print(single_df)
    print("\nPortfolio (Equal Raw vs Equal HMM):")
    print(portfolio_df)
    print("\nStress correlations:")
    print(corr_df)
    print(f"\nRecommendation: {recommendation}")
    print(f"\nSaved: {EXP_OUT / 'block2_hmm_single_metrics.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_portfolio_metrics.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_tail_comparison.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_corr_comparison.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_hmm_variant_report.md'}")


if __name__ == "__main__":
    main()
