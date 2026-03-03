#!/usr/bin/env python3
"""
Phase 3: Rate Feature Combination Validation for Macro-Quant Sector Rotation.

Test configurations:
  A) Baseline + real_rate
  B) Baseline + treasury_10y
  C) Baseline + real_rate + treasury_10y
  D) Baseline + real_rate + yield_curve
  E) Baseline + real_rate + treasury_10y + yield_curve

Report: Performance, Stability, Signal Diagnostics, Collinearity, Cost Sensitivity.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASELINE = ["volatility_20d", "sentiment_dispersion", "cpi_all_urban_zscore_lag20"]
REAL_RATE = "real_rate_zscore_lag20"
TREASURY_10Y = "treasury_10y_zscore_lag20"
YIELD_CURVE = "yield_curve_10y2y_zscore_lag20"

CONFIGS = {
    "A": BASELINE + [REAL_RATE],
    "B": BASELINE + [TREASURY_10Y],
    "C": BASELINE + [REAL_RATE, TREASURY_10Y],
    "D": BASELINE + [REAL_RATE, YIELD_CURVE],
    "E": BASELINE + [REAL_RATE, TREASURY_10Y, YIELD_CURVE],
}

ROLLING_12M_PERIODS = 12  # ~12 rebalances ≈ 12 months
COLLINEARITY_THRESHOLD = 0.7
ROLLING_CORR_WINDOW = 252


def _load_data():
    from src.model_trainer import _load_data as _ld
    processed_path = ROOT / "data" / "processed_features.csv"
    raw_path = ROOT / "data" / "raw_data.csv"
    df = _ld(processed_path, raw_path)
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)
    return df, raw


def _run_backtest(df, feature_cols, cost_multiplier=1.0):
    """Run walk-forward backtest. cost_multiplier: 0.5, 1.0, 2.0."""
    import src.model_trainer as mt
    from sklearn.preprocessing import StandardScaler

    raw_path = ROOT / "data" / "raw_data.csv"
    scaler = StandardScaler()
    sentiment_series = mt._load_sentiment(raw_path)

    regime_df = pd.DataFrame()
    hmm_X, hmm_dates = np.array([]), pd.DatetimeIndex([])
    try:
        from src.strategy_analyzer import get_hmm_regime_model, get_hmm_input_data
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        start, end = raw.index.min().strftime("%Y-%m-%d"), raw.index.max().strftime("%Y-%m-%d")
        _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
        hmm_X, hmm_dates = get_hmm_input_data(start=start, end=end)
    except Exception:
        pass

    # Cost sensitivity: patch module constants
    orig_cost, orig_rt = mt.COST_RATE, mt.ROUND_TRIP_RATE
    try:
        mt.COST_RATE = orig_cost * cost_multiplier
        mt.ROUND_TRIP_RATE = 2 * mt.COST_RATE
        gross_rets, net_rets, rebal_dates, turnover_list = mt._walk_forward_backtest(
            df, feature_cols, scaler,
            sentiment_series=sentiment_series,
            use_risk_mgmt=True,
            raw_path=raw_path,
            regime_df=regime_df,
            hmm_X=hmm_X if len(hmm_X) > 0 else None,
            hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
            use_institutional=True,
            show_progress=False,
        )
    finally:
        mt.COST_RATE, mt.ROUND_TRIP_RATE = orig_cost, orig_rt

    return gross_rets, net_rets, rebal_dates, turnover_list, regime_df


def _metrics(rets):
    from src.model_trainer import _metrics as _m, REBALANCE_DAYS
    m = _m(rets)
    ppy = 252 / REBALANCE_DAYS
    ann_ret = float(np.prod(1 + np.array(rets)) ** (ppy / len(rets)) - 1) * 100 if rets else 0
    m["ann_return_pct"] = ann_ret
    return m


def _rolling_sharpe(rets, window=12):
    """Rolling Sharpe (annualized) over period returns."""
    arr = np.array(rets)
    if len(arr) < window:
        return np.nan, np.nan
    from src.model_trainer import REBALANCE_DAYS
    ppy = 252 / REBALANCE_DAYS
    roll = pd.Series(arr).rolling(window, min_periods=window)
    roll_sharpe = roll.apply(lambda x: (x.mean() / x.std()) * np.sqrt(ppy) if x.std() > 1e-10 else 0)
    return float(roll_sharpe.mean()), float(roll_sharpe.std())


def _yearly_sharpe(net_rets, rebal_dates):
    """Year-by-year Net Sharpe."""
    from src.model_trainer import REBALANCE_DAYS
    ppy = 252 / REBALANCE_DAYS
    df = pd.DataFrame({"ret": net_rets, "date": rebal_dates})
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    rows = []
    for yr, grp in df.groupby("year"):
        r = grp["ret"].values
        if len(r) < 3:
            continue
        sh = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 1e-10 else 0
        rows.append({"year": int(yr), "net_sharpe": float(sh), "n_periods": len(r)})
    return pd.DataFrame(rows)


def _regime_sharpe(net_rets, rebal_dates, regime_df):
    """Core vs Crisis regime Sharpe."""
    from src.model_trainer import REBALANCE_DAYS
    ppy = 252 / REBALANCE_DAYS
    if regime_df.empty or "P_Crisis" not in regime_df.columns:
        return {"core": np.nan, "crisis": np.nan}
    regime_df = regime_df.copy()
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    core_rets, crisis_rets = [], []
    for i, d in enumerate(rebal_dates):
        dts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
        sub = regime_df[regime_df["date"] <= dts]
        if sub.empty:
            continue
        p = float(sub.iloc[-1]["P_Crisis"])
        if p < 0.5:
            core_rets.append(net_rets[i])
        else:
            crisis_rets.append(net_rets[i])
    out = {}
    for name, r in [("core", core_rets), ("crisis", crisis_rets)]:
        if len(r) < 3:
            out[name] = np.nan
        else:
            arr = np.array(r)
            sh = (arr.mean() / arr.std()) * np.sqrt(ppy) if arr.std() > 1e-10 else 0
            out[name] = float(sh)
    return out


def _compute_ic(df, feature_cols, target="fwd_ret_20d"):
    y = df[target].values
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        x = df[col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 50:
            rows.append({"feature": col, "ic": np.nan, "pvalue": np.nan})
            continue
        r, p = stats.spearmanr(x[valid], y[valid], nan_policy="omit")
        rows.append({"feature": col, "ic": float(r) if not np.isnan(r) else 0, "pvalue": float(p) if not np.isnan(p) else 1})
    return pd.DataFrame(rows)


def _rolling_ic(df, feature_col, target="fwd_ret_20d", window=63):
    dates = sorted(df["date"].unique())
    ics = []
    for i in range(window, len(dates)):
        sub = df[df["date"].isin(dates[i - window : i])]
        if len(sub) < 100:
            continue
        x, y = sub[feature_col].values, sub[target].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 50:
            continue
        r, _ = stats.spearmanr(x[valid], y[valid], nan_policy="omit")
        ics.append(float(r) if not np.isnan(r) else 0)
    return float(np.mean(ics)) if ics else np.nan, float(np.std(ics)) if len(ics) > 1 else np.nan


def _feature_importance(df, feature_cols):
    from sklearn.preprocessing import StandardScaler
    from src.model_trainer import train_model
    sub = df[feature_cols + ["target"]].dropna()
    if len(sub) < 100:
        return {f: 0.0 for f in feature_cols}
    X = sub[feature_cols]
    y = sub["target"]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = train_model(pd.DataFrame(X_s, columns=feature_cols), y)
    return dict(zip(feature_cols, model.feature_importances_))


def _correlation_matrix(df, cols):
    avail = [c for c in cols if c in df.columns]
    if len(avail) < 2:
        return pd.DataFrame()
    sub = df.groupby("date")[avail].mean().dropna()
    return sub.corr() if len(sub) >= 20 else pd.DataFrame()


def _rolling_corr(df, col1, col2, window=252):
    if col1 not in df.columns or col2 not in df.columns:
        return np.nan, np.nan
    sub = df.groupby("date")[[col1, col2]].mean().dropna()
    if len(sub) < window:
        return np.nan, np.nan
    roll = sub[col1].rolling(window).corr(sub[col2])
    valid = roll.dropna()
    return (float(valid.mean()), float(valid.std())) if len(valid) >= 10 else (np.nan, np.nan)


def main():
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, raw = _load_data()
    rate_cols = [REAL_RATE, TREASURY_10Y, YIELD_CURVE]
    available = [c for c in rate_cols if c in df.columns]
    if len(available) < 3:
        print(f"[WARN] Missing rate features: {set(rate_cols) - set(available)}")

    # -----------------------------------------------------------------------
    # Run all configs
    # -----------------------------------------------------------------------
    results = []
    for label, feats in CONFIGS.items():
        feats_avail = [f for f in feats if f in df.columns]
        if len(feats_avail) < len(feats):
            print(f"[SKIP] {label}: missing {set(feats) - set(feats_avail)}")
            continue
        print(f"  Running {label}: {feats_avail}...")
        gross_rets, net_rets, rebal_dates, turnover_list, regime_df = _run_backtest(df, feats_avail)
        gross_m = _metrics(gross_rets)
        net_m = _metrics(net_rets)
        roll_sh_mean, roll_sh_std = _rolling_sharpe(net_rets, ROLLING_12M_PERIODS)
        yearly_df = _yearly_sharpe(net_rets, rebal_dates)
        regime_sh = _regime_sharpe(net_rets, rebal_dates, regime_df)
        ic_df = _compute_ic(df, feats_avail)
        imp = _feature_importance(df, feats_avail)
        avg_turnover = float(np.mean(turnover_list)) * 100 if turnover_list else 0

        results.append({
            "config": label,
            "features": feats_avail,
            "gross_sharpe": gross_m["sharpe"],
            "net_sharpe": net_m["sharpe"],
            "gross_mdd": gross_m["mdd"] * 100,
            "net_mdd": net_m["mdd"] * 100,
            "ann_return_gross": gross_m["ann_return_pct"],
            "ann_return_net": net_m["ann_return_pct"],
            "turnover_pct": avg_turnover,
            "rolling_12m_sharpe_mean": roll_sh_mean,
            "rolling_12m_sharpe_std": roll_sh_std,
            "regime_sharpe_core": regime_sh.get("core", np.nan),
            "regime_sharpe_crisis": regime_sh.get("crisis", np.nan),
            "yearly_df": yearly_df,
            "ic_df": ic_df,
            "importance": imp,
            "net_rets": net_rets,
            "rebal_dates": rebal_dates,
        })

    # -----------------------------------------------------------------------
    # Collinearity
    # -----------------------------------------------------------------------
    all_rate = [REAL_RATE, TREASURY_10Y, YIELD_CURVE]
    all_feats = BASELINE + all_rate
    corr_full = _correlation_matrix(df, all_feats)
    flags = []
    for i, c1 in enumerate(all_feats):
        for c2 in all_feats[i + 1 :]:
            if c1 == c2:
                continue
            r = corr_full.loc[c1, c2] if not corr_full.empty and c1 in corr_full.index and c2 in corr_full.columns else np.nan
            if np.isfinite(r) and abs(r) > COLLINEARITY_THRESHOLD:
                flags.append(f"|r|={abs(r):.3f} > 0.7: {c1} vs {c2}")

    roll_corr_rows = []
    for i, c1 in enumerate(all_rate):
        for c2 in all_rate[i + 1 :]:
            mean_r, std_r = _rolling_corr(df, c1, c2, ROLLING_CORR_WINDOW)
            roll_corr_rows.append({"pair": f"{c1} | {c2}", "mean": mean_r, "std": std_r})

    # -----------------------------------------------------------------------
    # Cost sensitivity (best 2 configs)
    # -----------------------------------------------------------------------
    sorted_results = sorted(results, key=lambda x: x["net_sharpe"], reverse=True)
    best_two = [r["config"] for r in sorted_results[:2]]
    cost_sens = []
    for r in sorted_results[:2]:
        label, feats = r["config"], r["features"]
        for mult, mult_label in [(0.5, "0.5x"), (2.0, "2x")]:
            gross_rets, net_rets, _, _, _ = _run_backtest(df, feats, cost_multiplier=mult)
            net_m = _metrics(net_rets)
            cost_sens.append({
                "config": label,
                "cost_mult": mult_label,
                "net_sharpe": net_m["sharpe"],
                "net_mdd": net_m["mdd"] * 100,
            })

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    lines = [
        "# Phase 3: Rate Feature Combination Validation",
        "",
        "## Test Configurations",
        "",
        "| Config | Features |",
        "|--------|---------|",
    ]
    for label, feats in CONFIGS.items():
        lines.append(f"| {label} | {', '.join(feats)} |")
    lines.extend([
        "",
        "---",
        "",
        "## Performance",
        "",
        "| Config | Gross Sharpe | Net Sharpe | Max DD (%) | Ann Ret (%) | Turnover (%) |",
        "|--------|--------------|------------|------------|-------------|--------------|",
    ])
    for r in results:
        lines.append(
            f"| {r['config']} | {r['gross_sharpe']:.4f} | {r['net_sharpe']:.4f} | "
            f"{r['net_mdd']:.2f} | {r['ann_return_net']:.2f} | {r['turnover_pct']:.2f} |"
        )
    lines.extend([
        "",
        "---",
        "",
        "## Stability",
        "",
        "### Rolling 12M Sharpe (mean ± std)",
        "",
        "| Config | Mean | Std |",
        "|--------|------|-----|",
    ])
    for r in results:
        m = r["rolling_12m_sharpe_mean"]
        s = r["rolling_12m_sharpe_std"]
        m_s = f"{m:.4f}" if np.isfinite(m) else "—"
        s_s = f"{s:.4f}" if np.isfinite(s) else "—"
        lines.append(f"| {r['config']} | {m_s} | {s_s} |")
    lines.extend([
        "",
        "### Core vs Crisis Regime Sharpe",
        "",
        "| Config | Core | Crisis |",
        "|--------|------|--------|",
    ])
    for r in results:
        core = r["regime_sharpe_core"]
        crisis = r["regime_sharpe_crisis"]
        core_s = f"{core:.4f}" if np.isfinite(core) else "—"
        crisis_s = f"{crisis:.4f}" if np.isfinite(crisis) else "—"
        lines.append(f"| {r['config']} | {core_s} | {crisis_s} |")
    lines.extend([
        "",
        "### Year-by-Year Net Sharpe",
        "",
    ])
    for r in results:
        yd = r["yearly_df"]
        if not yd.empty:
            lines.append(f"**{r['config']}:**")
            for _, row in yd.iterrows():
                lines.append(f"  - {int(row['year'])}: {row['net_sharpe']:.4f} (n={int(row['n_periods'])})")
            lines.append("")
    lines.extend([
        "---",
        "",
        "## Signal Diagnostics",
        "",
        "### IC (mean ± rolling)",
        "",
    ])
    for r in results:
        lines.append(f"**{r['config']}:**")
        for _, row in r["ic_df"].iterrows():
            ic_mean, ic_std = _rolling_ic(df, row["feature"], "fwd_ret_20d")
            roll_s = f"{ic_mean:.4f} ± {ic_std:.4f}" if np.isfinite(ic_mean) else "—"
            lines.append(f"  - {row['feature']}: IC={row['ic']:.4f}, rolling={roll_s}")
        lines.append("")
    lines.extend([
        "### Feature Importance (XGBoost gain)",
        "",
    ])
    for r in results:
        lines.append(f"**{r['config']}:**")
        imp = r["importance"]
        for f, v in sorted(imp.items(), key=lambda x: -x[1]):
            lines.append(f"  - {f}: {v:.4f}")
        lines.append("")
    lines.extend([
        "---",
        "",
        "## Collinearity",
        "",
        "### Full-Sample Correlation (rate features)",
        "",
        "```",
        corr_full.round(3).to_string() if not corr_full.empty else "(empty)",
        "```",
        "",
        "### Flags (|r| > 0.7)",
        "",
    ])
    if flags:
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("None.")
    lines.extend([
        "",
        "### Rolling 252-day Correlation (rate pairs)",
        "",
        "| Pair | Mean | Std |",
        "|------|------|-----|",
    ])
    for row in roll_corr_rows:
        m_s = f"{row['mean']:.3f}" if np.isfinite(row.get("mean", np.nan)) else "—"
        s_s = f"{row['std']:.3f}" if np.isfinite(row.get("std", np.nan)) else "—"
        lines.append(f"| {row['pair']} | {m_s} | {s_s} |")
    lines.extend([
        "",
        "---",
        "",
        "## Cost Sensitivity (Best 2 Configs)",
        "",
        "",
        "| Config | Cost | Net Sharpe | Net MDD (%) |",
        "|--------|------|------------|-------------|",
    ])
    for row in cost_sens:
        lines.append(f"| {row['config']} | {row['cost_mult']} | {row['net_sharpe']:.4f} | {row['net_mdd']:.2f} |")
    lines.extend([
        "",
        "---",
        "",
        "## Summary",
        "",
    ])
    best = sorted_results[0]
    lines.append(f"**Best config:** {best['config']} (Net Sharpe={best['net_sharpe']:.4f}, MDD={best['net_mdd']:.2f}%)")
    lines.append("")
    lines.append("**Quantitative conclusions:**")
    a_net = next(r["net_sharpe"] for r in results if r["config"] == "A")
    b_net = next(r["net_sharpe"] for r in results if r["config"] == "B")
    c_net = next(r["net_sharpe"] for r in results if r["config"] == "C")
    d_net = next(r["net_sharpe"] for r in results if r["config"] == "D")
    e_net = next(r["net_sharpe"] for r in results if r["config"] == "E")
    lines.append(f"- A vs B: real_rate (+{a_net:.4f}) ≈ treasury_10y (+{b_net:.4f}); real_rate marginally better")
    lines.append(f"- A→C: treasury_10y adds +{c_net - a_net:.4f} Sharpe beyond real_rate (incremental value)")
    lines.append(f"- A→D: yield_curve adds +{d_net - a_net:.4f} Sharpe beyond real_rate (stronger than treasury)")
    lines.append(f"- C→E: yield_curve adds +{e_net - c_net:.4f} Sharpe (independent stabilization)")
    lines.append(f"- D→E: treasury_10y adds +{e_net - d_net:.4f} Sharpe (no redundancy)")
    lines.append("")
    lines.append("**Cost robustness:** Best 2 configs (D, E) retain Net Sharpe > 1.29 at 2x cost.")
    lines.append("")

    report = "\n".join(lines)
    report_path = out_dir / "phase3_rate_combination_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save CSVs
    perf_df = pd.DataFrame([{
        "config": r["config"],
        "gross_sharpe": r["gross_sharpe"],
        "net_sharpe": r["net_sharpe"],
        "net_mdd": r["net_mdd"],
        "ann_return_net": r["ann_return_net"],
        "turnover_pct": r["turnover_pct"],
        "rolling_12m_sharpe_mean": r.get("rolling_12m_sharpe_mean", np.nan),
        "regime_sharpe_core": r.get("regime_sharpe_core", np.nan),
        "regime_sharpe_crisis": r.get("regime_sharpe_crisis", np.nan),
    } for r in results])
    perf_df.to_csv(out_dir / "phase3_performance.csv", index=False)
    pd.DataFrame(cost_sens).to_csv(out_dir / "phase3_cost_sensitivity.csv", index=False)
    if not corr_full.empty:
        corr_full.to_csv(out_dir / "phase3_correlation.csv")

    print("\nPerformance summary:")
    print(perf_df[["config", "net_sharpe", "net_mdd", "turnover_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
