#!/usr/bin/env python3
"""
Portfolio Audit: Verify Sharpe scaling, CVaR contribution, drawdown attribution, risk budget sensitivity.

Structural validation exercise. No parameter tuning.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

PPY_DAILY = 252
CVAR_ALPHA = 0.95


def _load_returns() -> pd.DataFrame:
    """Load portfolio combined returns from CSV."""
    path = EXP_OUT / "portfolio_combined_returns.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run portfolio_combined.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def _sharpe_daily(rets: pd.Series) -> float:
    """Daily Sharpe: mean/std * sqrt(252). Single annualization."""
    arr = rets.dropna()
    if len(arr) < 2 or arr.std() < 1e-12:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(PPY_DAILY))


def _mdd(rets: pd.Series) -> float:
    arr = rets.dropna().values
    wealth = np.ones(len(arr) + 1)
    for i, r in enumerate(arr):
        wealth[i + 1] = max(0.0, wealth[i] * (1.0 + r))
    peak = np.maximum.accumulate(wealth[1:])
    dd = np.where(peak > 1e-12, (wealth[1:] - peak) / peak, 0)
    return float(np.min(dd)) * 100


def _ann_ret(rets: pd.Series) -> float:
    arr = rets.dropna().values
    n = len(arr)
    if n == 0:
        return 0.0
    return float(np.prod(1 + arr) ** (PPY_DAILY / n) - 1) * 100


def _ann_vol(rets: pd.Series) -> float:
    arr = rets.dropna().values
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr) * np.sqrt(PPY_DAILY)) * 100


def _cvar95(rets: pd.Series) -> float:
    arr = rets.dropna().values
    n_tail = max(1, int(len(arr) * (1 - CVAR_ALPHA)))
    worst = np.partition(arr, n_tail - 1)[:n_tail]
    return float(np.mean(worst)) * 100


def _find_drawdown_periods(rets: pd.Series, n: int = 3) -> list:
    """Identify top n drawdown periods (peak to trough)."""
    arr = rets.dropna().values
    dates = rets.dropna().index.tolist()
    wealth = np.ones(len(arr) + 1)
    for i, r in enumerate(arr):
        wealth[i + 1] = max(0.0, wealth[i] * (1.0 + r))
    w = wealth[1:]  # wealth at end of each day

    # Find peaks: indices where wealth reaches a new high
    peak_run = np.maximum.accumulate(w)
    is_new_peak = np.concatenate([[True], peak_run[1:] > peak_run[:-1]])
    peak_indices = np.where(is_new_peak)[0].tolist()

    # For each peak, find trough (min wealth before next peak)
    periods = []
    for k in range(len(peak_indices) - 1):
        peak_idx = peak_indices[k]
        next_peak_idx = peak_indices[k + 1]
        segment = w[peak_idx : next_peak_idx + 1]
        if len(segment) < 2:
            continue
        trough_rel = np.argmin(segment)
        trough_idx = peak_idx + trough_rel
        depth = (segment[trough_rel] - segment[0]) / segment[0] if segment[0] > 1e-12 else 0
        if depth < -0.001:  # meaningful drawdown
            periods.append({
                "start_idx": peak_idx,
                "trough_idx": trough_idx,
                "depth": depth,
                "start_date": dates[peak_idx] if peak_idx < len(dates) else dates[0],
                "trough_date": dates[trough_idx] if trough_idx < len(dates) else dates[0],
            })
    # Also check last peak to end
    if len(peak_indices) >= 1:
        peak_idx = peak_indices[-1]
        segment = w[peak_idx:]
        if len(segment) > 1:
            trough_rel = np.argmin(segment)
            trough_idx = peak_idx + trough_rel
            depth = (segment[trough_rel] - segment[0]) / segment[0] if segment[0] > 1e-12 else 0
            if depth < -0.001:
                periods.append({
                    "start_idx": peak_idx,
                    "trough_idx": trough_idx,
                    "depth": depth,
                    "start_date": dates[peak_idx] if peak_idx < len(dates) else dates[0],
                    "trough_date": dates[trough_idx] if trough_idx < len(dates) else dates[-1],
                })
    periods.sort(key=lambda x: x["depth"])
    return periods[:n]


def _mcr(w: float, rets_i: pd.Series, rets_p: pd.Series) -> float:
    """Marginal contribution to risk: w * Cov(R_i, R_p) / Var(R_p)."""
    common = rets_i.dropna().index.intersection(rets_p.dropna().index)
    if len(common) < 10:
        return 0.5
    ri = rets_i.reindex(common).ffill().bfill().fillna(0).values
    rp = rets_p.reindex(common).ffill().bfill().fillna(0).values
    cov = np.cov(ri, rp)
    if cov.size < 4:
        return 0.5
    cov_ip = cov[0, 1] * PPY_DAILY
    var_p = np.var(rp) * PPY_DAILY
    return w * cov_ip / var_p if var_p > 1e-12 else 0.5


def main():
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    df = _load_returns()

    b1 = df["block1"]
    b2 = df["block2"]
    ew = df["equal_weight"]
    inv = df["inverse_vol"]

    # Use full dataset
    b1 = df["block1"]
    b2 = df["block2"]
    ew = df["equal_weight"]
    inv = df["inverse_vol"]

    print("\n=== Portfolio Audit ===\n")

    # STEP 1 — Sharpe verification
    print("STEP 1: Sharpe Calculation Verification")
    print("  Return frequency: DAILY")
    print("  Annualization: Sharpe = mean(daily) / std(daily) * sqrt(252)")
    print("  No double annualization.\n")

    step1_rows = []
    for name, rets in [("Block 1", b1), ("Block 2", b2), ("Equal Weight", ew), ("Inverse Vol", inv)]:
        arr = rets.dropna()
        arr = arr[arr != 0] if (arr == 0).all() is False else arr  # Avoid div by zero
        if len(arr) < 2:
            step1_rows.append({"strategy": name, "mean_daily": np.nan, "std_daily": np.nan, "ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan})
            continue
        mean_d = float(arr.mean())
        std_d = float(arr.std())
        ann_ret = _ann_ret(rets)
        ann_vol = _ann_vol(rets)
        sharpe = _sharpe_daily(rets)
        step1_rows.append({
            "strategy": name,
            "mean_daily": mean_d,
            "std_daily": std_d,
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
        })
        print(f"  {name}:")
        print(f"    Mean (daily): {mean_d:.8f}")
        print(f"    Std (daily):  {std_d:.8f}")
        print(f"    Ann Return:   {ann_ret:.4f}%")
        print(f"    Ann Vol:      {ann_vol:.4f}%")
        print(f"    Verified Sharpe: {sharpe:.4f}\n")

    # STEP 2 — CVaR contribution
    print("STEP 2: CVaR Contribution (Tail Risk Decomposition)")
    n_tail = max(1, int(len(ew.dropna()) * 0.05))
    tail_idx = ew.dropna().nsmallest(n_tail).index
    ew_tail = ew.loc[tail_idx]
    b1_tail = b1.loc[tail_idx]
    b2_tail = b2.loc[tail_idx]
    cvar_ew = _cvar95(ew)
    avg_b1_tail = float(b1_tail.mean())
    avg_b2_tail = float(b2_tail.mean())
    avg_port_tail = float(ew_tail.mean())
    # Contribution: 0.5 * avg_b1_tail and 0.5 * avg_b2_tail. Total = avg_port_tail = 0.5*avg_b1 + 0.5*avg_b2
    cvar_contrib_b1 = 0.5 * avg_b1_tail
    cvar_contrib_b2 = 0.5 * avg_b2_tail
    total = cvar_contrib_b1 + cvar_contrib_b2
    if abs(total) > 1e-12:
        pct_b1 = 100 * cvar_contrib_b1 / total
        pct_b2 = 100 * cvar_contrib_b2 / total
    else:
        pct_b1 = 50.0
        pct_b2 = 50.0
    cvar_contrib = [
        {"block": "Block 1", "avg_return_tail": avg_b1_tail, "contribution": cvar_contrib_b1, "pct_contribution": pct_b1},
        {"block": "Block 2", "avg_return_tail": avg_b2_tail, "contribution": cvar_contrib_b2, "pct_contribution": pct_b2},
    ]
    print(f"  Worst 5% days: {n_tail}")
    print(f"  Portfolio CVaR (95%): {cvar_ew:.4f}%")
    print(f"  Block 1 avg tail return: {avg_b1_tail:.6f}")
    print(f"  Block 2 avg tail return: {avg_b2_tail:.6f}")
    print(f"  CVaR contribution %: Block 1 = {pct_b1:.1f}%, Block 2 = {pct_b2:.1f}%")

    pd.DataFrame(cvar_contrib).to_csv(EXP_OUT / "cvar_contribution.csv", index=False)

    # STEP 3 — Drawdown attribution
    print("\nSTEP 3: Drawdown Attribution")
    ew_clean = ew.dropna()
    dd_periods = _find_drawdown_periods(ew_clean, n=3)
    dd_attr_rows = []
    for i, p in enumerate(dd_periods):
        start_idx = p["start_idx"]
        trough_idx = p["trough_idx"]
        start_date = p["start_date"]
        trough_date = p["trough_date"]
        depth = p["depth"] * 100

        ew_dd = ew_clean.iloc[start_idx : trough_idx + 1]
        # Align b1, b2 to ew_clean by date
        dd_dates = ew_dd.index
        b1_dd = b1.reindex(dd_dates).ffill().bfill().fillna(0)
        b2_dd = b2.reindex(dd_dates).ffill().bfill().fillna(0)

        cum_ret_ew = float(np.prod(1 + ew_dd.values) - 1) * 100
        cum_ret_b1 = float(np.prod(1 + b1_dd.values) - 1) * 100
        cum_ret_b2 = float(np.prod(1 + b2_dd.values) - 1) * 100
        contrib_b1 = 0.5 * cum_ret_b1
        contrib_b2 = 0.5 * cum_ret_b2

        dd_attr_rows.append({
            "rank": i + 1,
            "start_date": str(start_date)[:10],
            "trough_date": str(trough_date)[:10],
            "total_drawdown_pct": depth,
            "block1_cumulative_contrib": contrib_b1,
            "block2_cumulative_contrib": contrib_b2,
        })
        print(f"  DD #{i+1}: {start_date} → {trough_date}, Total: {depth:.2f}%")
        print(f"    Block 1 contrib: {contrib_b1:.2f}%, Block 2 contrib: {contrib_b2:.2f}%")

    pd.DataFrame(dd_attr_rows).to_csv(EXP_OUT / "drawdown_attribution.csv", index=False)

    # STEP 4 — Risk budget sensitivity
    print("\nSTEP 4: Risk Budget Sensitivity")
    weights = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
    risk_budget_rows = []
    for w1, w2 in weights:
        ret_p = w1 * b1 + w2 * b2
        sharpe = _sharpe_daily(ret_p)
        mdd = _mdd(ret_p)
        cvar = _cvar95(ret_p)
        vol = _ann_vol(ret_p)
        mcr1 = 100 * _mcr(w1, b1, ret_p)
        mcr2 = 100 * _mcr(w2, b2, ret_p)
        risk_budget_rows.append({
            "w_block1": w1,
            "w_block2": w2,
            "sharpe": sharpe,
            "mdd": mdd,
            "cvar95": cvar,
            "volatility": vol,
            "mcr_block1_pct": mcr1,
            "mcr_block2_pct": mcr2,
        })
        print(f"  {int(w1*100)}/{int(w2*100)}: Sharpe={sharpe:.4f}, MDD={mdd:.2f}%, CVaR={cvar:.2f}%, Vol={vol:.2f}%, MCR B1={mcr1:.1f}% B2={mcr2:.1f}%")

    pd.DataFrame(risk_budget_rows).to_csv(EXP_OUT / "risk_budget_sensitivity.csv", index=False)

    # STEP 5 — Report
    report = f"""# Portfolio Audit Report

> Structural validation: Sharpe scaling, CVaR contribution, drawdown attribution, risk budget sensitivity.

## STEP 1 — Sharpe Calculation Verification

**Return frequency:** DAILY

**Annualization formula:**
- Daily Sharpe = mean(daily_return) / std(daily_return) × √252
- Annualized return = (∏(1+r))^(252/n) - 1
- Annualized volatility = std(daily) × √252

**No double annualization.**

| Strategy | Mean (daily) | Std (daily) | Ann Return | Ann Vol | Verified Sharpe |
|----------|--------------|--------------|------------|---------|-----------------|
"""
    for r in step1_rows:
        m = f"{r['mean_daily']:.8f}" if not np.isnan(r.get('mean_daily', np.nan)) else "—"
        s = f"{r['std_daily']:.8f}" if not np.isnan(r.get('std_daily', np.nan)) else "—"
        ar = f"{r['ann_ret']:.4f}%" if not np.isnan(r.get('ann_ret', np.nan)) else "—"
        av = f"{r['ann_vol']:.4f}%" if not np.isnan(r.get('ann_vol', np.nan)) else "—"
        sh = f"{r['sharpe']:.4f}" if not np.isnan(r.get('sharpe', np.nan)) else "—"
        report += f"| {r['strategy']} | {m} | {s} | {ar} | {av} | {sh} |\n"

    report += f"""
## STEP 2 — CVaR Contribution (Tail Risk Decomposition)

Equal Weight portfolio. Worst 5% daily returns.

| Block | Avg Return (tail days) | Contribution to CVaR | % Contribution |
|-------|------------------------|----------------------|----------------|
| Block 1 | {avg_b1_tail:.6f} | {cvar_contrib_b1:.6f} | {pct_b1:.1f}% |
| Block 2 | {avg_b2_tail:.6f} | {cvar_contrib_b2:.6f} | {pct_b2:.1f}% |

Portfolio CVaR (95%): {cvar_ew:.4f}%

## STEP 3 — Drawdown Attribution

Top 3 drawdown periods in Equal Weight portfolio.
Contributions = 0.5 × (cumulative return of each block over the period). May not sum to total due to geometric compounding.

| Rank | Start | Trough | Total DD % | Block 1 Contrib | Block 2 Contrib |
|------|-------|--------|-------------|-----------------|-----------------|
"""
    for r in dd_attr_rows:
        report += f"| {r['rank']} | {r['start_date']} | {r['trough_date']} | {r['total_drawdown_pct']:.2f}% | {r['block1_cumulative_contrib']:.2f}% | {r['block2_cumulative_contrib']:.2f}% |\n"

    report += """
## STEP 4 — Risk Budget Sensitivity

Static weight allocations. No optimization.

| Weights (B1/B2) | Sharpe | MDD | CVaR (95%) | Volatility | MCR B1 | MCR B2 |
|-----------------|--------|-----|------------|-------------|--------|--------|
"""
    for r in risk_budget_rows:
        report += f"| {int(r['w_block1']*100)}/{int(r['w_block2']*100)} | {r['sharpe']:.4f} | {r['mdd']:.2f}% | {r['cvar95']:.2f}% | {r['volatility']:.2f}% | {r['mcr_block1_pct']:.1f}% | {r['mcr_block2_pct']:.1f}% |\n"

    report += """
---
*Generated by experiments/scripts/portfolio_audit.py*
*Structural validation. No parameter tuning.*
"""

    with open(EXP_OUT / "portfolio_audit_report.md", "w") as f:
        f.write(report)
    print(f"\nSaved: {EXP_OUT / 'portfolio_audit_report.md'}")
    print(f"Saved: {EXP_OUT / 'cvar_contribution.csv'}")
    print(f"Saved: {EXP_OUT / 'drawdown_attribution.csv'}")
    print(f"Saved: {EXP_OUT / 'risk_budget_sensitivity.csv'}")


if __name__ == "__main__":
    main()
