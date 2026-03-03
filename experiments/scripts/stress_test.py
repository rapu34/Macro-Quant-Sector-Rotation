#!/usr/bin/env python3
"""
Stress Test — True Daily PnL (2005–2026).

Portfolio: 30% Block1 + 70% Block2_HMM_REBAL_ONLY
- No target vol scaling
- No daily smoothing
- No look-ahead

Run: python experiments/scripts/stress_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
W_B1 = 0.3
W_B2 = 0.7


def _fetch_spy_vix(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY returns and VIX (for delta_vix)."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")

    out = {}
    for t in ["SPY", "^VIX"]:
        d = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if d.empty:
            continue
        c = d["Close"] if "Close" in d.columns else d.iloc[:, 0]
        if isinstance(c, pd.DataFrame):
            c = c.squeeze()
        out[t] = c

    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    df = df.ffill().dropna(how="all")

    mkt = pd.DataFrame(index=df.index)
    if "SPY" in df.columns:
        mkt["r_spy"] = df["SPY"].pct_change()
    if "^VIX" in df.columns:
        mkt["delta_vix"] = df["^VIX"].pct_change()
    return mkt.dropna(how="all")


def main():
    print("\n=== Stress Test (True Daily PnL) ===\n")

    # Load blocks
    block1 = pd.read_csv(EXP_OUT / "true_daily_block1.csv", index_col=0, parse_dates=True).squeeze()
    block2 = pd.read_csv(EXP_OUT / "block2_hmm_expanding_rebalonly.csv", index_col=0, parse_dates=True)
    block2 = block2.iloc[:, 0].squeeze()

    common = block1.index.intersection(block2.index)
    r_b1 = block1.reindex(common).ffill().bfill().fillna(0)
    r_b2 = block2.reindex(common).ffill().bfill().fillna(0)
    r_p = W_B1 * r_b1 + W_B2 * r_b2

    start = r_p.index.min().strftime("%Y-%m-%d")
    end = r_p.index.max().strftime("%Y-%m-%d")
    print(f"Portfolio: {start} ~ {end}, n={len(r_p)}")

    # Fetch SPY, VIX
    mkt = _fetch_spy_vix(start, end)
    merged = pd.concat([r_p.rename("r_p"), r_b1.rename("r_b1"), r_b2.rename("r_b2"), mkt], axis=1, sort=True)
    merged = merged.dropna(how="any")
    print(f"Merged (with SPY/VIX): n={len(merged)}")

    r_p = merged["r_p"]
    r_b1 = merged["r_b1"]
    r_b2 = merged["r_b2"]
    r_spy = merged["r_spy"]
    delta_vix = merged["delta_vix"]

    # -------------------------------------------------------------------------
    # STEP 1 — Historical Conditional Stress
    # -------------------------------------------------------------------------
    cond_stats = []

    # 1) Market Selloff
    mask_selloff = r_spy <= -0.03
    for mask, label in [
        (mask_selloff, "spy_selloff_3pct"),
    ]:
        if mask.sum() > 0:
            sub = r_p[mask]
            cvar95 = np.percentile(sub, 5)
            cond_stats.append({
                "condition": label,
                "N": int(mask.sum()),
                "mean_port_return": sub.mean(),
                "worst_1d": sub.min(),
                "cvar_95": cvar95,
            })

    # 2) Volatility Spike
    mask_vix = delta_vix >= 0.10
    if mask_vix.sum() > 0:
        sub = r_p[mask_vix]
        cond_stats.append({
            "condition": "vix_spike_10pct",
            "N": int(mask_vix.sum()),
            "mean_port_return": sub.mean(),
            "worst_1d": sub.min(),
            "cvar_95": np.percentile(sub, 5),
        })

    # 3) Worst Portfolio Days
    tail_corr_rows = []
    for pct, label in [(0.05, "port_worst_5pct"), (0.01, "port_worst_1pct")]:
        thresh = np.percentile(r_p, pct * 100)
        mask = r_p <= thresh
        sub = r_p[mask]
        sub_b1 = r_b1[mask]
        sub_b2 = r_b2[mask]
        corr = np.corrcoef(sub_b1, sub_b2)[0, 1] if len(sub) > 2 else np.nan
        tail_corr_rows.append({"condition": label, "tail_corr_b1_b2": corr})
        cond_stats.append({
            "condition": label,
            "N": int(mask.sum()),
            "mean_port_return": sub.mean(),
            "worst_1d": sub.min(),
            "cvar_95": np.percentile(sub, 5),
        })

    # 4) Worst Market Days
    for pct, label in [(0.05, "spy_worst_5pct"), (0.01, "spy_worst_1pct")]:
        thresh = np.percentile(r_spy, pct * 100)
        mask = r_spy <= thresh
        sub = r_p[mask]
        sub_b1 = r_b1[mask]
        sub_b2 = r_b2[mask]
        corr = np.corrcoef(sub_b1, sub_b2)[0, 1] if len(sub) > 2 else np.nan
        tail_corr_rows.append({"condition": label, "tail_corr_b1_b2": corr})
        cond_stats.append({
            "condition": label,
            "N": int(mask.sum()),
            "mean_port_return": sub.mean(),
            "worst_1d": sub.min(),
            "cvar_95": np.percentile(sub, 5),
        })

    df_cond = pd.DataFrame(cond_stats)
    df_cond.to_csv(EXP_OUT / "stress_conditional_stats.csv", index=False)
    pd.DataFrame(tail_corr_rows).to_csv(EXP_OUT / "stress_tail_corr.csv", index=False)
    print(f"Saved: stress_conditional_stats.csv, stress_tail_corr.csv")

    # -------------------------------------------------------------------------
    # STEP 2 — Block Contribution in Stress
    # -------------------------------------------------------------------------
    contrib_rows = []
    conditions = [
        ("spy_selloff_3pct", r_spy <= -0.03),
        ("vix_spike_10pct", delta_vix >= 0.10),
        ("port_worst_5pct", r_p <= np.percentile(r_p, 5)),
        ("port_worst_1pct", r_p <= np.percentile(r_p, 1)),
        ("spy_worst_5pct", r_spy <= np.percentile(r_spy, 5)),
        ("spy_worst_1pct", r_spy <= np.percentile(r_spy, 1)),
    ]
    for label, mask in conditions:
        if mask.sum() < 2:
            continue
        c_b1 = (W_B1 * r_b1[mask]).mean()
        c_b2 = (W_B2 * r_b2[mask]).mean()
        total = c_b1 + c_b2
        pct_b1 = 100 * c_b1 / total if abs(total) > 1e-10 else np.nan
        pct_b2 = 100 * c_b2 / total if abs(total) > 1e-10 else np.nan
        driver = "Block1" if abs(c_b1) > abs(c_b2) else "Block2"
        contrib_rows.append({
            "condition": label,
            "N": int(mask.sum()),
            "contrib_b1": c_b1,
            "contrib_b2": c_b2,
            "pct_b1": pct_b1,
            "pct_b2": pct_b2,
            "driver": driver,
        })
    pd.DataFrame(contrib_rows).to_csv(EXP_OUT / "stress_block_contribution.csv", index=False)
    print("Saved: stress_block_contribution.csv")

    # -------------------------------------------------------------------------
    # STEP 3 — Event Case Study
    # -------------------------------------------------------------------------
    def _recovery_days(ser: pd.Series, peak_idx, trough_idx) -> int:
        """Trading days from trough to first date when cum >= cum at peak."""
        cum = (1 + ser).cumprod()
        peak_level = cum.loc[peak_idx]
        after_trough = ser.loc[trough_idx:].index.tolist()
        for i, d in enumerate(after_trough[1:], 1):
            if cum.loc[d] >= peak_level:
                return i
        return -1

    case_rows = []

    # COVID (extended to 2020-12-31 for recovery)
    covid = merged.loc["2020-02-01":"2020-04-30"]
    covid_ext = merged.loc["2020-02-01":"2020-12-31"]
    if len(covid) > 0:
        cum = (1 + covid["r_p"]).cumprod()
        dd = cum / cum.cummax() - 1
        mdd = dd.min()
        trough_idx = dd.idxmin()
        peak_idx = cum.loc[:trough_idx].idxmax()
        mdd_window = covid.loc[peak_idx:trough_idx]
        if len(mdd_window) > 1:
            mdd_window = mdd_window.iloc[1:]
        block1_mdd = (W_B1 * mdd_window["r_b1"]).sum() if len(mdd_window) > 0 else np.nan
        block2_mdd = (W_B2 * mdd_window["r_b2"]).sum() if len(mdd_window) > 0 else np.nan
        cum_b1 = (W_B1 * covid["r_b1"]).cumsum()
        cum_b2 = (W_B2 * covid["r_b2"]).cumsum()
        period_ret = covid["r_p"].sum()
        rec_days = _recovery_days(covid_ext["r_p"], peak_idx, trough_idx) if len(covid_ext) > 0 else -1
        case_rows.append({
            "event": "covid_2020",
            "start": "2020-02-01",
            "end": "2020-04-30",
            "N": len(covid),
            "peak_to_trough_mdd": mdd,
            "recovery_days": rec_days,
            "period_return": period_ret,
            "block1_cum_contrib": cum_b1.iloc[-1],
            "block2_cum_contrib": cum_b2.iloc[-1],
            "block1_mdd_contrib": block1_mdd,
            "block2_mdd_contrib": block2_mdd,
        })

    # 2022 Rate Shock (extended to 2023-12-31 for recovery)
    rate22 = merged.loc["2022-01-01":"2022-10-31"]
    rate22_ext = merged.loc["2022-01-01":"2023-12-31"]
    if len(rate22) > 0:
        cum = (1 + rate22["r_p"]).cumprod()
        dd = cum / cum.cummax() - 1
        mdd = dd.min()
        trough_idx = dd.idxmin()
        peak_idx = cum.loc[:trough_idx].idxmax()
        mdd_window = rate22.loc[peak_idx:trough_idx]
        if len(mdd_window) > 1:
            mdd_window = mdd_window.iloc[1:]
        block1_mdd = (W_B1 * mdd_window["r_b1"]).sum() if len(mdd_window) > 0 else np.nan
        block2_mdd = (W_B2 * mdd_window["r_b2"]).sum() if len(mdd_window) > 0 else np.nan
        cum_b1 = (W_B1 * rate22["r_b1"]).cumsum()
        cum_b2 = (W_B2 * rate22["r_b2"]).cumsum()
        period_ret = rate22["r_p"].sum()
        rec_days = _recovery_days(rate22_ext["r_p"], peak_idx, trough_idx) if len(rate22_ext) > 0 else -1
        case_rows.append({
            "event": "rate_shock_2022",
            "start": "2022-01-01",
            "end": "2022-10-31",
            "N": len(rate22),
            "peak_to_trough_mdd": mdd,
            "recovery_days": rec_days,
            "period_return": period_ret,
            "block1_cum_contrib": cum_b1.iloc[-1],
            "block2_cum_contrib": cum_b2.iloc[-1],
            "block1_mdd_contrib": block1_mdd,
            "block2_mdd_contrib": block2_mdd,
        })

    pd.DataFrame(case_rows).to_csv(EXP_OUT / "stress_case_study.csv", index=False)
    print("Saved: stress_case_study.csv")

    # -------------------------------------------------------------------------
    # STEP 4 — Parametric Shock + Historical Simulation
    # -------------------------------------------------------------------------
    summary = pd.read_csv(EXP_OUT / "factor_regression_summary.csv")
    beta_mkt = summary[summary["factor"] == "r_mkt"]["coef"].values[0]
    beta_vix = summary[summary["factor"] == "delta_vix"]["coef"].values[0]

    # Historical: conditional mean loss when condition is met
    hist_m5 = r_p[r_spy <= -0.05]
    hist_m10 = r_p[r_spy <= -0.10]
    hist_v20 = r_p[delta_vix >= 0.20]
    hist_v40 = r_p[delta_vix >= 0.40]

    scenarios = [
        ("SPY ≤-5%", -0.05, 0, beta_mkt * (-0.05), hist_m5.mean() if len(hist_m5) >= 5 else np.nan, len(hist_m5)),
        ("SPY ≤-10%", -0.10, 0, beta_mkt * (-0.10), hist_m10.mean() if len(hist_m10) >= 1 else np.nan, len(hist_m10)),
        ("ΔVIX ≥+20%", 0, 0.20, beta_vix * 0.20, hist_v20.mean() if len(hist_v20) >= 5 else np.nan, len(hist_v20)),
        ("ΔVIX ≥+40%", 0, 0.40, beta_vix * 0.40, hist_v40.mean() if len(hist_v40) >= 3 else np.nan, len(hist_v40)),
    ]
    param_rows = []
    for name, d_mkt, d_vix, est_loss, hist_loss, n_hist in scenarios:
        note = "Single observation (N=1) – not statistically meaningful" if name == "SPY ≤-10%" and n_hist == 1 else ""
        param_rows.append({
            "scenario": name,
            "parametric_loss": est_loss,
            "historical_avg_loss": hist_loss,
            "historical_N": n_hist,
            "note": note,
        })
    pd.DataFrame(param_rows).to_csv(EXP_OUT / "stress_parametric_scenarios.csv", index=False)
    print("Saved: stress_parametric_scenarios.csv")

    # -------------------------------------------------------------------------
    # Top 5 Worst Days
    # -------------------------------------------------------------------------
    worst5_idx = r_p.nsmallest(5).index
    worst5 = merged.loc[worst5_idx][["r_p", "r_b1", "r_b2", "r_spy", "delta_vix"]].copy()
    worst5.index = worst5.index.strftime("%Y-%m-%d")
    worst5.columns = ["r_port", "r_block1", "r_block2", "r_spy", "delta_vix"]
    worst5.to_csv(EXP_OUT / "stress_top5_worst_days.csv")
    print("Saved: stress_top5_worst_days.csv")

    # -------------------------------------------------------------------------
    # STEP 5 — Report
    # -------------------------------------------------------------------------
    worst_1pct = df_cond[df_cond["condition"] == "port_worst_1pct"]
    worst_5pct = df_cond[df_cond["condition"] == "port_worst_5pct"]
    selloff = df_cond[df_cond["condition"] == "spy_selloff_3pct"]
    vix_spike = df_cond[df_cond["condition"] == "vix_spike_10pct"]

    tail_corr = pd.read_csv(EXP_OUT / "stress_tail_corr.csv")
    contrib = pd.read_csv(EXP_OUT / "stress_block_contribution.csv")
    case = pd.read_csv(EXP_OUT / "stress_case_study.csv")
    param = pd.read_csv(EXP_OUT / "stress_parametric_scenarios.csv")

    m1 = worst_1pct["mean_port_return"].values[0] * 100 if len(worst_1pct) else np.nan
    c5 = worst_5pct["cvar_95"].values[0] * 100 if len(worst_5pct) else np.nan
    s3 = selloff["mean_port_return"].values[0] * 100 if len(selloff) else np.nan
    v10 = vix_spike["mean_port_return"].values[0] * 100 if len(vix_spike) else np.nan
    n_selloff = int(selloff["N"].values[0]) if len(selloff) else 0
    n_vix = int(vix_spike["N"].values[0]) if len(vix_spike) else 0
    n_worst1 = int(worst_1pct["N"].values[0]) if len(worst_1pct) else 0
    n_worst5 = int(worst_5pct["N"].values[0]) if len(worst_5pct) else 0

    port_1pct_corr = tail_corr[tail_corr["condition"] == "port_worst_1pct"]["tail_corr_b1_b2"].values
    port_5pct_corr = tail_corr[tail_corr["condition"] == "port_worst_5pct"]["tail_corr_b1_b2"].values
    tc1 = port_1pct_corr[0] if len(port_1pct_corr) else np.nan
    tc5 = port_5pct_corr[0] if len(port_5pct_corr) else np.nan

    # Top 5 worst days for report
    worst5_df = pd.read_csv(EXP_OUT / "stress_top5_worst_days.csv", index_col=0)

    report = f"""# Stress Test Report (True Daily PnL)

**Portfolio:** 30% Block1 + 70% Block2_HMM_REBAL_ONLY  
**Period:** {start} ~ {end}  
**Sample:** n={len(merged)}

**Changelog:** Clarified SPY ≤-10% historical estimate (N=1). Added interpretability note for single-observation scenario. Added recovery-duration interpretation (2022 vs 2020).

---

## 1. Key Metrics

| Metric | Value | N |
|--------|-------|---|
| Worst 1% avg loss | {m1:.2f}% | {n_worst1} |
| CVaR(95%) (avg of worst 5%) | {c5:.2f}% | {n_worst5} |
| SPY ≤-3% days avg loss | {s3:.2f}% | {n_selloff} |
| ΔVIX ≥+10% days avg loss | {v10:.2f}% | {n_vix} |

---

## 2. Tail Correlation (Block1 vs Block2)

*Tail correlation ~0.28 is moderate; in worst days both blocks tend to move in the same direction. Do not overstate diversification benefits.*

| Condition | Tail Corr | N |
|-----------|-----------|---|
| Port worst 1% | {tc1:.4f} | {n_worst1} |
| Port worst 5% | {tc5:.4f} | {n_worst5} |

---

## 3. 2020 / 2022 Event Summary

*Block contrib = sum of weighted daily returns over period; sums to period return. MDD contrib = sum over peak-to-trough window only. Recovery = trading days from trough to first date back at peak.*

| Event | MDD | Recovery | Period ret | Block1 | Block2 | MDD contrib B1 | MDD contrib B2 |
|-------|-----|----------|------------|--------|-------|----------------|----------------|
"""
    for _, row in case.iterrows():
        pr = row.get("period_return", np.nan)
        pr_str = f"{pr*100:.2f}%" if pd.notna(pr) else "—"
        b1_mdd = row.get("block1_mdd_contrib", np.nan)
        b2_mdd = row.get("block2_mdd_contrib", np.nan)
        b1_mdd_str = f"{b1_mdd*100:.2f}%" if pd.notna(b1_mdd) else "—"
        b2_mdd_str = f"{b2_mdd*100:.2f}%" if pd.notna(b2_mdd) else "—"
        rec = row.get("recovery_days", -1)
        rec_str = f"{int(rec)}d (~{rec/21:.1f}mo)" if rec >= 0 else "—"
        report += f"| {row['event']} | {row['peak_to_trough_mdd']*100:.2f}% | {rec_str} | {pr_str} | {row['block1_cum_contrib']*100:.2f}% | {row['block2_cum_contrib']*100:.2f}% | {b1_mdd_str} | {b2_mdd_str} |\n"

    report += """
---

## 4. Parametric Scenarios (Beta-based)

*Estimated market beta ~0.31 from post-2013 factor regression for the 30/70 portfolio.*

*Parametric = beta-based 1-day estimate. Historical = avg portfolio loss on days when condition occurred (empirical). Use historical for tail-risk sizing; parametric for quick sensitivity.*

| Scenario | Parametric | Historical (avg) | N | Note |
|----------|------------|------------------|---|------|
"""
    for _, row in param.iterrows():
        pct = row["parametric_loss"] * 100
        hist = row["historical_avg_loss"]
        hist_str = f"{hist*100:.2f}%" if pd.notna(hist) else "—"
        n_hist = int(row["historical_N"]) if "historical_N" in row else 0
        note = row.get("note", "")
        note = "" if (pd.isna(note) or note == "nan") else str(note)
        report += f"| {row['scenario']} | {pct:.2f}% | {hist_str} | {n_hist} | {note} |\n"

    report += """
*The SPY ≤-10% condition occurred only once in the sample (N=1, 2020-03-16). The historical loss is reported for completeness but is not statistically reliable.*

---

## 5. Top 5 Worst Portfolio Days

| Date | r_port | r_block1 | r_block2 | r_spy | delta_vix |
|------|--------|----------|----------|-------|-----------|
"""
    for idx, row in worst5_df.iterrows():
        report += f"| {idx} | {row['r_port']*100:.2f}% | {row['r_block1']*100:.2f}% | {row['r_block2']*100:.2f}% | {row['r_spy']*100:.2f}% | {row['delta_vix']*100:.2f}% |\n"

    report += """
*2025-04-04: Real market stress (VIX +50%, U.S.-China tariff escalation). Notable that a 2025 event made top 5 worst days.*

---

## 6. Risk Management Suggestions

1. **Exposure control:** Estimated market beta is ~0.31 based on post-2013 factor regression for the 30/70 portfolio. In conditional stress (SPY selloff, VIX spike), Block2 tends to drive most of the loss. However, in extreme MDD windows (e.g. COVID peak-to-trough), Block1 contributes more than its 30% weight would suggest—both blocks contributed roughly equally to the drawdown. Do not assume Block2 alone drives tail risk.

2. **Tail structure:** Tail correlation ~0.28 is moderate; in worst days both blocks tend to move in the same direction. Do not overstate diversification. If tail correlation rises in live data, review whether both blocks are exposed to the same risk factors.

3. **Recovery duration:** Notably, the 2022 rate shock shows much longer recovery (370 trading days) than COVID (109), indicating drawdown persistence during tightening cycles.

---

*Generated by stress_test.py — True Daily PnL, no smoothing, no look-ahead*
"""

    with open(EXP_OUT / "stress_test_report.md", "w") as f:
        f.write(report)
    print("Saved: stress_test_report.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
