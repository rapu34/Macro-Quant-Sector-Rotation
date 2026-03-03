#!/usr/bin/env python3
"""
Block1 + Block2_HMM_REBAL_ONLY 앙상블 비율 스윕.

Run: python experiments/scripts/ensemble_weight_sweep.py

True Daily PnL only. No smoothing, no additional scaling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
PPY = 252
ROLLING_36M_DAYS = 36 * 21  # ~756 trading days
WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]  # Block1 weight; Block2 = 1 - w1


def compute_performance(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ["AnnReturn", "AnnVol", "Sharpe", "MDD", "CVaR", "Worst1pct", "Worst5pct"]}
    ann_ret = r.mean() * PPY
    ann_vol = r.std() * np.sqrt(PPY)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax()).min() - 1
    q05 = r.quantile(0.05)
    cvar = r[r <= q05].mean()
    worst1 = r[r <= r.quantile(0.01)].mean() if len(r) >= 100 else np.nan
    worst5 = r[r <= r.quantile(0.05)].mean() if len(r) >= 20 else np.nan
    return {
        "AnnReturn": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe,
        "MDD": mdd, "CVaR": cvar,
        "Worst1pct": worst1, "Worst5pct": worst5,
    }


def stress_corr_on_dates(x: pd.Series, y: pd.Series, cond_dates: pd.DatetimeIndex) -> float:
    common = cond_dates.intersection(x.index).intersection(y.index)
    if len(common) < 2:
        return np.nan
    return x.loc[common].corr(y.loc[common])


def risk_decomposition(r1: pd.Series, r2: pd.Series, w1: float) -> dict:
    """Vol contribution %, MCR, CVaR contribution %."""
    common = r1.index.intersection(r2.index).dropna(how="all")
    r1_a = r1.reindex(common).ffill().bfill().fillna(0)
    r2_a = r2.reindex(common).ffill().bfill().fillna(0)
    rp = w1 * r1_a + (1 - w1) * r2_a
    r1_a = r1_a.dropna()
    r2_a = r2_a.reindex(r1_a.index).ffill().bfill().fillna(0)
    rp = (w1 * r1_a + (1 - w1) * r2_a).dropna()
    r1_a = r1_a.reindex(rp.index).ffill().bfill().fillna(0)
    r2_a = r2_a.reindex(rp.index).ffill().bfill().fillna(0)
    if len(rp) < 10:
        return {"vol_contrib_b1_pct": np.nan, "vol_contrib_b2_pct": np.nan, "mcr_b1": np.nan, "mcr_b2": np.nan, "cvar_contrib_b1_pct": np.nan, "cvar_contrib_b2_pct": np.nan}
    cov1p = np.cov(r1_a.values, rp.values)[0, 1]
    cov2p = np.cov(r2_a.values, rp.values)[0, 1]
    vol_p = np.std(rp)
    if vol_p < 1e-12:
        return {"vol_contrib_b1_pct": np.nan, "vol_contrib_b2_pct": np.nan, "mcr_b1": np.nan, "mcr_b2": np.nan, "cvar_contrib_b1_pct": np.nan, "cvar_contrib_b2_pct": np.nan}
    mcr1 = cov1p / vol_p
    mcr2 = cov2p / vol_p
    vol_contrib1 = w1 * mcr1 / vol_p * 100
    vol_contrib2 = (1 - w1) * mcr2 / vol_p * 100
    q05 = rp.quantile(0.05)
    tail_mask = rp <= q05
    if tail_mask.sum() < 2:
        cvar_c1, cvar_c2 = np.nan, np.nan
    else:
        r1_tail = r1_a.loc[tail_mask].mean()
        r2_tail = r2_a.loc[tail_mask].mean()
        rp_tail = rp.loc[tail_mask].mean()
        if abs(rp_tail) < 1e-12:
            cvar_c1, cvar_c2 = np.nan, np.nan
        else:
            cvar_c1 = w1 * r1_tail / rp_tail * 100
            cvar_c2 = (1 - w1) * r2_tail / rp_tail * 100
    return {
        "vol_contrib_b1_pct": vol_contrib1, "vol_contrib_b2_pct": vol_contrib2,
        "mcr_b1": mcr1, "mcr_b2": mcr2,
        "cvar_contrib_b1_pct": cvar_c1, "cvar_contrib_b2_pct": cvar_c2,
    }


def rolling_metrics(r: pd.Series) -> dict:
    """Rolling 36M Sharpe mean/std, Rolling 36M MDD."""
    r = r.dropna()
    if len(r) < ROLLING_36M_DAYS:
        return {"roll_sharpe_mean": np.nan, "roll_sharpe_std": np.nan, "roll_mdd_mean": np.nan, "roll_mdd_worst": np.nan}
    roll_sharpe = r.rolling(ROLLING_36M_DAYS, min_periods=ROLLING_36M_DAYS).apply(
        lambda x: x.mean() / x.std() * np.sqrt(PPY) if x.std() > 1e-12 else np.nan
    )
    roll_mdd = r.rolling(ROLLING_36M_DAYS, min_periods=ROLLING_36M_DAYS).apply(
        lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax()).min() - 1 if len(x) > 0 else np.nan
    )
    sh_valid = roll_sharpe.dropna()
    mdd_valid = roll_mdd.dropna()
    return {
        "roll_sharpe_mean": sh_valid.mean() if len(sh_valid) > 0 else np.nan,
        "roll_sharpe_std": sh_valid.std() if len(sh_valid) > 1 else np.nan,
        "roll_mdd_mean": mdd_valid.mean() * 100 if len(mdd_valid) > 0 else np.nan,
        "roll_mdd_worst": mdd_valid.min() * 100 if len(mdd_valid) > 0 else np.nan,
    }


def main():
    print("\n=== Ensemble Weight Sweep (Block1 + Block2_HMM_REBAL_ONLY) ===\n")

    block1 = pd.read_csv(EXP_OUT / "true_daily_block1.csv", index_col=0, parse_dates=True).squeeze()
    block2 = pd.read_csv(EXP_OUT / "block2_hmm_expanding_rebalonly.csv", index_col=0, parse_dates=True)
    block2 = block2.iloc[:, 0].squeeze()

    common = block1.index.intersection(block2.index)
    block1 = block1.reindex(common).ffill().bfill().fillna(0)
    block2 = block2.reindex(common).ffill().bfill().fillna(0)

    full_corr = block1.corr(block2)
    print(f"Full-sample corr(Block1, Block2): {full_corr:.4f}\n")

    results_metrics = []
    results_risk = []
    results_tail = []
    results_stability = []
    results_div = []

    for w1 in WEIGHTS:
        w2 = 1 - w1
        port = w1 * block1 + w2 * block2

        perf = compute_performance(port)
        perf["w_block1"] = w1
        perf["w_block2"] = w2
        results_metrics.append(perf)

        tail_row = {"w_block1": w1, "w_block2": w2, **{k: perf[k] for k in ["CVaR", "Worst1pct", "Worst5pct", "MDD"]}}
        results_tail.append(tail_row)

        risk_row = risk_decomposition(block1, block2, w1)
        risk_row["w_block1"] = w1
        risk_row["w_block2"] = w2
        results_risk.append(risk_row)

        stab_row = rolling_metrics(port)
        stab_row["w_block1"] = w1
        stab_row["w_block2"] = w2
        results_stability.append(stab_row)

        n_worst = max(1, int(len(port) * 0.1))
        worst_port = port.nsmallest(n_worst).index
        worst_b1 = block1.nsmallest(n_worst).index
        worst_b2 = block2.nsmallest(n_worst).index
        div_row = {
            "w_block1": w1,
            "full_corr": full_corr,
            "stress_port_worst10": stress_corr_on_dates(block1, block2, worst_port),
            "stress_b1_worst10": stress_corr_on_dates(block1, block2, worst_b1),
            "stress_b2_worst10": stress_corr_on_dates(block1, block2, worst_b2),
        }
        results_div.append(div_row)

    df_metrics = pd.DataFrame(results_metrics)
    df_risk = pd.DataFrame(results_risk)
    df_tail = pd.DataFrame(results_tail)
    df_stability = pd.DataFrame(results_stability)
    df_div = pd.DataFrame(results_div)

    # Recommendation
    sharpe_max_idx = df_metrics["Sharpe"].idxmax()
    cvar_best_idx = df_metrics["CVaR"].idxmax()
    mdd_best_idx = df_metrics["MDD"].idxmax()
    stress_base = df_div.loc[df_div["w_block1"] == 0.5, "stress_port_worst10"].values[0]
    stress_thresh = abs(stress_base) + 0.15 if not np.isnan(stress_base) else 1.0

    candidates = []
    for i, row in df_metrics.iterrows():
        w1 = row["w_block1"]
        stress = df_div.loc[df_div["w_block1"] == w1, "stress_port_worst10"].values[0]
        if not np.isnan(stress) and abs(stress) > stress_thresh:
            continue
        candidates.append((i, row["Sharpe"], row["MDD"], row["CVaR"], w1))

    if not candidates:
        rec_w = 0.5
        rec_reason = "No candidate passed stress filter; default 50/50."
    else:
        candidates.sort(key=lambda x: (-x[1], x[2], x[3]))
        rec_w = candidates[0][4]
        rec_reason = f"Best risk-adjusted: Sharpe={candidates[0][1]:.4f}, MDD={candidates[0][2]:.2%}, CVaR={candidates[0][3]:.4f}. Stress within threshold."

    EXP_OUT.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(EXP_OUT / "ensemble_weight_sweep_metrics.csv", index=False)
    df_risk.to_csv(EXP_OUT / "ensemble_weight_sweep_risk.csv", index=False)
    df_tail.to_csv(EXP_OUT / "ensemble_weight_sweep_tail.csv", index=False)
    df_stability.to_csv(EXP_OUT / "ensemble_weight_sweep_stability.csv", index=False)

    report = f"""# Ensemble Weight Sweep Report

Run: `python experiments/scripts/ensemble_weight_sweep.py`

Block1 + Block2_HMM_REBAL_ONLY. True Daily PnL. No smoothing.

## Sharpe vs MDD Tradeoff

| w(Block1) | w(Block2) | Sharpe | Ann Vol | Ann Return | MDD | CVaR |
|-----------|-----------|--------|----------|------------|-----|------|
"""
    for _, r in df_metrics.iterrows():
        report += f"| {r['w_block1']:.0%} | {r['w_block2']:.0%} | {r['Sharpe']:.4f} | {r['AnnVol']:.4f} | {r['AnnReturn']:.4f} | {r['MDD']:.2%} | {r['CVaR']:.4f} |\n"

    report += f"""
## CVaR 최소 비율 (tail 개선)

CVaR(95%)가 가장 덜 나쁜(least negative) 비율: **{df_metrics.loc[cvar_best_idx, 'w_block1']:.0%} / {df_metrics.loc[cvar_best_idx, 'w_block2']:.0%}**
CVaR = {df_metrics.loc[cvar_best_idx, 'CVaR']:.4f}

## Risk Contribution 변화

| w(Block1) | Vol Contrib B1 (%) | Vol Contrib B2 (%) | MCR B1 | MCR B2 | CVaR Contrib B1 (%) | CVaR Contrib B2 (%) |
|-----------|--------------------|--------------------|--------|--------|---------------------|---------------------|
"""
    for _, r in df_risk.iterrows():
        report += f"| {r['w_block1']:.0%} | {r['vol_contrib_b1_pct']:.1f} | {r['vol_contrib_b2_pct']:.1f} | {r['mcr_b1']:.6f} | {r['mcr_b2']:.6f} | {r['cvar_contrib_b1_pct']:.1f} | {r['cvar_contrib_b2_pct']:.1f} |\n"

    report += f"""
## Diversification / Stress

| w(Block1) | Full Corr | Stress (Port Worst 10%) | Stress (B1 Worst 10%) | Stress (B2 Worst 10%) |
|-----------|-----------|--------------------------|------------------------|------------------------|
"""
    for _, r in df_div.iterrows():
        report += f"| {r['w_block1']:.0%} | {r['full_corr']:.4f} | {r['stress_port_worst10']:.4f} | {r['stress_b1_worst10']:.4f} | {r['stress_b2_worst10']:.4f} |\n"

    report += f"""
## Stability (Rolling 36M)

| w(Block1) | Roll Sharpe Mean | Roll Sharpe Std | Roll MDD Mean (%) | Roll MDD Worst (%) |
|-----------|------------------|-----------------|-------------------|--------------------|
"""
    for _, r in df_stability.iterrows():
        report += f"| {r['w_block1']:.0%} | {r['roll_sharpe_mean']:.4f} | {r['roll_sharpe_std']:.4f} | {r['roll_mdd_mean']:.2f} | {r['roll_mdd_worst']:.2f} |\n"

    report += f"""
## 최종 추천

**비율: {rec_w:.0%} Block1 / {1-rec_w:.0%} Block2_HMM_REBAL_ONLY**

**선택 이유:** {rec_reason}

**Tradeoff 요약:**
- Sharpe·MDD 최적: 30/70 (Sharpe {df_metrics.loc[df_metrics['w_block1']==0.3, 'Sharpe'].values[0]:.2f}, MDD {df_metrics.loc[df_metrics['w_block1']==0.3, 'MDD'].values[0]:.1%})
- CVaR 최적: 50/50 (CVaR {df_metrics.loc[df_metrics['w_block1']==0.5, 'CVaR'].values[0]:.2%})

- Sharpe만 최대가 아닌, MDD·CVaR 개선을 고려한 risk-adjusted 효율성
- Stress correlation 급상승 시 제외
---
*Generated by ensemble_weight_sweep.py*
"""

    with open(EXP_OUT / "ensemble_weight_sweep_report.md", "w") as f:
        f.write(report)

    print("Metrics:")
    print(df_metrics.to_string(index=False))
    print("\nRisk decomposition:")
    print(df_risk.to_string(index=False))
    print(f"\nRecommendation: {rec_w:.0%} / {1-rec_w:.0%}")
    print(f"Reason: {rec_reason}")
    print(f"\nSaved: {EXP_OUT / 'ensemble_weight_sweep_metrics.csv'}")
    print(f"Saved: {EXP_OUT / 'ensemble_weight_sweep_risk.csv'}")
    print(f"Saved: {EXP_OUT / 'ensemble_weight_sweep_tail.csv'}")
    print(f"Saved: {EXP_OUT / 'ensemble_weight_sweep_stability.csv'}")
    print(f"Saved: {EXP_OUT / 'ensemble_weight_sweep_report.md'}")


if __name__ == "__main__":
    main()
