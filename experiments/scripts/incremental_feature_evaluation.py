#!/usr/bin/env python3
"""
Institutional Incremental Feature Evaluation for Macro-Quant Sector Rotation.

STEP 1: Incremental Testing — Add ONE candidate at a time to baseline, run full walk-forward backtest.
STEP 2: Collinearity Check — Full-sample and rolling 252-day correlation.
STEP 3: Turnover Impact — Signal stability, switching frequency, cost drag.
STEP 4: Final Selection — Propose optimal 6–8 feature macro-complete set.

Baseline (LOCKED): volatility_20d, sentiment_dispersion, cpi_all_urban_zscore_lag20
Candidates: treasury_10y_zscore_lag20, yield_curve_10y2y_zscore_lag20, credit_spread_zscore_lag20,
            real_rate_zscore_lag20, treasury_10y_shock_20d_lag20
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
BASELINE_FEATURES = [
    "volatility_20d",
    "sentiment_dispersion",
    "cpi_all_urban_zscore_lag20",
]

CANDIDATE_FEATURES = [
    "treasury_10y_zscore_lag20",
    "yield_curve_10y2y_zscore_lag20",
    "credit_spread_zscore_lag20",
    "real_rate_zscore_lag20",
    "treasury_10y_shock_20d_lag20",
]

COLLINEARITY_THRESHOLD = 0.7
ROLLING_WINDOW = 252
SPLIT_FRACTION = 0.5  # Early vs late for regime IC


def _load_data():
    """Load processed features and raw prices."""
    from src.model_trainer import _load_data as _ld

    processed_path = ROOT / "data" / "processed_features.csv"
    raw_path = ROOT / "data" / "raw_data.csv"
    df = _ld(processed_path, raw_path)
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)
    return df, raw


def _run_backtest(df, feature_cols, use_institutional=True):
    """Run walk-forward backtest with given feature set."""
    from sklearn.preprocessing import StandardScaler
    from src.model_trainer import (
        _load_sentiment,
        _walk_forward_backtest,
        _metrics,
        REBALANCE_DAYS,
    )

    raw_path = ROOT / "data" / "raw_data.csv"
    scaler = StandardScaler()
    sentiment_series = _load_sentiment(raw_path)

    regime_df = pd.DataFrame()
    hmm_X, hmm_dates = np.array([]), pd.DatetimeIndex([])
    try:
        from src.strategy_analyzer import get_hmm_regime_model, get_hmm_input_data
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        start = raw.index.min().strftime("%Y-%m-%d")
        end = raw.index.max().strftime("%Y-%m-%d")
        _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
        hmm_X, hmm_dates = get_hmm_input_data(start=start, end=end)
    except Exception:
        pass

    gross_rets, net_rets, rebal_dates, turnover_list = _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment_series,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=use_institutional,
        show_progress=False,
    )

    gross_m = _metrics(gross_rets)
    net_m = _metrics(net_rets)
    periods_per_year = 252 / REBALANCE_DAYS
    ann_ret_gross = float(np.prod(1 + np.array(gross_rets)) ** (periods_per_year / len(gross_rets)) - 1) if gross_rets else 0
    ann_ret_net = float(np.prod(1 + np.array(net_rets)) ** (periods_per_year / len(net_rets)) - 1) if net_rets else 0
    avg_turnover = float(np.mean(turnover_list)) * 100 if turnover_list else 0

    return {
        "gross_sharpe": gross_m["sharpe"],
        "net_sharpe": net_m["sharpe"],
        "gross_mdd": gross_m["mdd"] * 100,
        "net_mdd": net_m["mdd"] * 100,
        "ann_return_gross": ann_ret_gross * 100,
        "ann_return_net": ann_ret_net * 100,
        "turnover_pct": avg_turnover,
        "n_rebalances": len(rebal_dates),
        "gross_rets": gross_rets,
        "net_rets": net_rets,
        "rebal_dates": rebal_dates,
        "turnover_list": turnover_list,
    }


def _compute_ic(df, feature_cols, target_col="fwd_ret_20d"):
    """Average IC (Spearman) per feature vs forward return."""
    y = df[target_col].values
    results = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        x = df[col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 50:
            results.append({"feature": col, "ic": np.nan, "pvalue": np.nan})
            continue
        r, p = stats.spearmanr(x[valid], y[valid], nan_policy="omit")
        results.append({"feature": col, "ic": float(r) if not np.isnan(r) else 0, "pvalue": float(p) if not np.isnan(p) else 1})
    return pd.DataFrame(results)


def _compute_rolling_ic(df, feature_col, target_col="fwd_ret_20d", window=63):
    """Rolling IC stability (63-day window)."""
    dates = df["date"].unique()
    dates = sorted(dates)
    ics = []
    for i in range(window, len(dates)):
        sub = df[df["date"].isin(dates[i - window : i])]
        if len(sub) < 100:
            continue
        x = sub[feature_col].values
        y = sub[target_col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 50:
            continue
        r, _ = stats.spearmanr(x[valid], y[valid], nan_policy="omit")
        ics.append({"date": dates[i], "ic": float(r) if not np.isnan(r) else 0})
    if not ics:
        return np.nan, np.nan
    ic_df = pd.DataFrame(ics)
    return float(ic_df["ic"].mean()), float(ic_df["ic"].std())


def _compute_regime_ic(df, regime_df, feature_col, target_col="fwd_ret_20d"):
    """IC in Core vs Crisis regimes (P_Crisis < 0.5 = Core, >= 0.5 = Crisis)."""
    if regime_df.empty or "P_Crisis" not in regime_df.columns:
        return {"core": np.nan, "crisis": np.nan}
    regime_df = regime_df.copy()
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    merged = df.merge(regime_df[["date", "P_Crisis"]], on="date", how="left")
    merged = merged.dropna(subset=["P_Crisis"])
    core = merged[merged["P_Crisis"] < 0.5]
    crisis = merged[merged["P_Crisis"] >= 0.5]
    out = {}
    for name, sub in [("core", core), ("crisis", crisis)]:
        if len(sub) < 50:
            out[name] = np.nan
            continue
        x = sub[feature_col].values
        y = sub[target_col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 30:
            out[name] = np.nan
            continue
        r, _ = stats.spearmanr(x[valid], y[valid], nan_policy="omit")
        out[name] = float(r) if not np.isnan(r) else np.nan
    return out


def _correlation_matrix(df, cols):
    """Full-sample correlation between features (one row per date via mean for sector-varying)."""
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    sub = df.groupby("date")[available].mean().dropna()
    if len(sub) < 20:
        return pd.DataFrame()
    return sub.corr()


def _rolling_correlation(df, col1, col2, window=252):
    """Rolling 252-day correlation between two features."""
    if col1 not in df.columns or col2 not in df.columns:
        return np.nan, np.nan
    sub = df.groupby("date")[[col1, col2]].mean().dropna()
    if len(sub) < window:
        return np.nan, np.nan
    roll = sub[col1].rolling(window).corr(sub[col2])
    valid = roll.dropna()
    if len(valid) < 10:
        return np.nan, np.nan
    return float(valid.mean()), float(valid.std())


def run_step1_incremental_testing(df):
    """STEP 1: Run backtest for baseline and baseline + each candidate."""
    all_features = BASELINE_FEATURES + CANDIDATE_FEATURES
    available = [c for c in all_features if c in df.columns]
    missing = [c for c in all_features if c not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing}")

    results = []
    configs = [
        ("baseline", BASELINE_FEATURES),
    ]
    for cand in CANDIDATE_FEATURES:
        if cand in available:
            configs.append((f"baseline+{cand}", BASELINE_FEATURES + [cand]))

    importance_rows = []
    for name, feats in configs:
        feats_avail = [f for f in feats if f in df.columns]
        if len(feats_avail) < len(feats):
            print(f"[SKIP] {name}: missing {set(feats) - set(feats_avail)}")
            continue
        print(f"  Running: {name} ({len(feats_avail)} features)...")
        m = _run_backtest(df, feats_avail)
        results.append({
            "config": name,
            "features": feats_avail,
            **{k: v for k, v in m.items() if k not in ("gross_rets", "net_rets", "rebal_dates", "turnover_list")},
        })
        try:
            imp = _compute_feature_importance(df, feats_avail)
            for f, v in imp.items():
                importance_rows.append({"config": name, "feature": f, "importance": v})
        except Exception as e:
            print(f"    [WARN] Feature importance failed: {e}")

    importance_df = pd.DataFrame(importance_rows) if importance_rows else pd.DataFrame()
    return pd.DataFrame(results), importance_df


def _compute_feature_importance(df, feature_cols):
    """XGBoost feature importance (gain) from full-sample fit."""
    from sklearn.preprocessing import StandardScaler
    from src.model_trainer import train_model

    X = df[feature_cols].values
    y = df["target"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = train_model(pd.DataFrame(X_s, columns=feature_cols), pd.Series(y))
    imp = model.feature_importances_
    return dict(zip(feature_cols, imp))


def run_step1_signal_quality(df, regime_df):
    """STEP 1 (continued): IC, Rolling IC, Regime IC."""
    all_feats = BASELINE_FEATURES + CANDIDATE_FEATURES
    available = [c for c in all_feats if c in df.columns]
    ic_df = _compute_ic(df, available)
    ic_df = ic_df.set_index("feature")

    rows = []
    for feat in available:
        ic_mean, ic_std = _compute_rolling_ic(df, feat)
        regime_ic = _compute_regime_ic(df, regime_df, feat)
        rows.append({
            "feature": feat,
            "ic_avg": ic_df.loc[feat, "ic"] if feat in ic_df.index else np.nan,
            "ic_pvalue": ic_df.loc[feat, "pvalue"] if feat in ic_df.index else np.nan,
            "rolling_ic_mean": ic_mean,
            "rolling_ic_std": ic_std,
            "ic_core": regime_ic.get("core", np.nan),
            "ic_crisis": regime_ic.get("crisis", np.nan),
        })
    return pd.DataFrame(rows)


def run_step2_collinearity(df):
    """STEP 2: Full-sample and rolling correlation."""
    all_feats = BASELINE_FEATURES + CANDIDATE_FEATURES
    available = [c for c in all_feats if c in df.columns]
    corr_full = _correlation_matrix(df, available)

    flags = []
    for i, c1 in enumerate(available):
        for c2 in available[i + 1 :]:
            if c1 == c2:
                continue
            r = corr_full.loc[c1, c2] if not corr_full.empty and c1 in corr_full.index and c2 in corr_full.columns else np.nan
            if np.isfinite(r) and abs(r) > COLLINEARITY_THRESHOLD:
                flags.append(f"|{r:.3f}| > 0.7: {c1} vs {c2}")

    roll_corr = []
    for i, c1 in enumerate(available):
        for c2 in available[i + 1 :]:
            if c1 == c2:
                continue
            mean_r, std_r = _rolling_correlation(df, c1, c2, ROLLING_WINDOW)
            roll_corr.append({"pair": f"{c1} | {c2}", "mean": mean_r, "std": std_r})

    return corr_full, flags, pd.DataFrame(roll_corr)


def run_step3_turnover_impact(step1_results):
    """STEP 3: Turnover impact — compare turnover and net Sharpe across configs."""
    return step1_results[["config", "turnover_pct", "net_sharpe", "net_mdd", "n_rebalances"]].copy()


def run_step4_final_selection(step1_results, step2_flags, step3_df):
    """STEP 4: Propose optimal 6–8 feature set."""
    # Rank candidates by: net_sharpe improvement, low turnover, no collinearity
    baseline_row = step1_results[step1_results["config"] == "baseline"].iloc[0]
    base_sharpe = baseline_row["net_sharpe"]
    base_turnover = baseline_row["turnover_pct"]

    candidates_ranked = []
    for _, row in step1_results[step1_results["config"] != "baseline"].iterrows():
        cand = row["config"].replace("baseline+", "")
        delta_sharpe = row["net_sharpe"] - base_sharpe
        delta_turnover = row["turnover_pct"] - base_turnover
        candidates_ranked.append({
            "candidate": cand,
            "net_sharpe": row["net_sharpe"],
            "delta_sharpe_vs_baseline": delta_sharpe,
            "turnover_pct": row["turnover_pct"],
            "delta_turnover_vs_baseline": delta_turnover,
            "net_mdd": row["net_mdd"],
        })
    return pd.DataFrame(candidates_ranked)


def _render_report(step1_df, step1_ic_df, step1_importance, step2_corr, step2_flags, step2_roll, step3_df, step4_df):
    """Generate institutional PM-style markdown report."""
    lines = [
        "# Incremental Feature Evaluation — Macro-Quant Sector Rotation",
        "",
        "## Executive Summary",
        "",
        "Incremental testing of 5 candidate macro features against locked baseline. "
        "Risk engine, turnover budget, and target definition held identical.",
        "",
        "---",
        "",
        "## STEP 1 — Incremental Testing",
        "",
        "### Performance",
        "",
        "| Config | Gross Sharpe | Net Sharpe | Max DD (%) | Ann Ret (%) | Turnover (%) |",
        "|--------|--------------|------------|------------|-------------|--------------|",
    ]
    for _, row in step1_df.iterrows():
        lines.append(
            f"| {row['config']} | {row['gross_sharpe']:.4f} | {row['net_sharpe']:.4f} | "
            f"{row['net_mdd']:.2f} | {row['ann_return_net']:.2f} | {row['turnover_pct']:.2f} |"
        )
    lines.extend([
        "",
        "### Signal Quality (IC)",
        "",
        "| Feature | IC (avg) | p-value | Rolling IC mean | Rolling IC std | IC (Core) | IC (Crisis) |",
        "|---------|----------|---------|-----------------|----------------|-----------|--------------|",
    ])
    for _, row in step1_ic_df.iterrows():
        ic_core = f"{row['ic_core']:.4f}" if np.isfinite(row.get("ic_core", np.nan)) else "—"
        ic_crisis = f"{row['ic_crisis']:.4f}" if np.isfinite(row.get("ic_crisis", np.nan)) else "—"
        roll_mean = f"{row['rolling_ic_mean']:.4f}" if np.isfinite(row.get("rolling_ic_mean", np.nan)) else "—"
        roll_std = f"{row['rolling_ic_std']:.4f}" if np.isfinite(row.get("rolling_ic_std", np.nan)) else "—"
        lines.append(
            f"| {row['feature']} | {row['ic_avg']:.4f} | {row['ic_pvalue']:.2e} | "
            f"{roll_mean} | {roll_std} | {ic_core} | {ic_crisis} |"
        )
    if not step1_importance.empty:
        lines.extend([
            "",
            "### Model Diagnostics (XGBoost Feature Importance)",
            "",
        ])
        for config in step1_importance["config"].unique():
            sub = step1_importance[step1_importance["config"] == config].sort_values("importance", ascending=False)
            lines.append(f"**{config}:**")
            for _, r in sub.iterrows():
                lines.append(f"  - {r['feature']}: {r['importance']:.4f}")
            lines.append("")
    lines.extend([
        "",
        "---",
        "",
        "## STEP 2 — Collinearity Check",
        "",
        "### Full-Sample Correlation (excerpt)",
        "",
    ])
    if not step2_corr.empty:
        lines.append("```")
        lines.append(step2_corr.round(3).to_string())
        lines.append("```")
    else:
        lines.append("*(No correlation matrix)*")
    lines.extend([
        "",
        "### Flags (|r| > 0.7)",
        "",
    ])
    if step2_flags:
        for f in step2_flags:
            lines.append(f"- {f}")
    else:
        lines.append("None.")
    lines.extend([
        "",
        "### Rolling 252-day Correlation (mean ± std)",
        "",
        "| Pair | Mean | Std |",
        "|------|------|-----|",
    ])
    for _, row in step2_roll.iterrows():
        mean_s = f"{row['mean']:.3f}" if np.isfinite(row.get("mean", np.nan)) else "—"
        std_s = f"{row['std']:.3f}" if np.isfinite(row.get("std", np.nan)) else "—"
        lines.append(f"| {row['pair']} | {mean_s} | {std_s} |")
    lines.extend([
        "",
        "---",
        "",
        "## STEP 3 — Turnover Impact",
        "",
        "| Config | Turnover (%) | Net Sharpe | Net MDD (%) |",
        "|--------|--------------|------------|-------------|",
    ])
    for _, row in step3_df.iterrows():
        lines.append(f"| {row['config']} | {row['turnover_pct']:.2f} | {row['net_sharpe']:.4f} | {row['net_mdd']:.2f} |")
    lines.extend([
        "",
        "---",
        "",
        "## STEP 4 — Final Selection",
        "",
        "### Candidate Ranking (vs Baseline)",
        "",
        "| Candidate | Net Sharpe | Δ Sharpe | Turnover (%) | Δ Turnover | Net MDD (%) |",
        "|-----------|------------|----------|--------------|------------|-------------|",
    ])
    for _, row in step4_df.iterrows():
        d_sharpe = row["delta_sharpe_vs_baseline"]
        d_to = row["delta_turnover_vs_baseline"]
        lines.append(
            f"| {row['candidate']} | {row['net_sharpe']:.4f} | {d_sharpe:+.4f} | "
            f"{row['turnover_pct']:.2f} | {d_to:+.2f} | {row['net_mdd']:.2f} |"
        )
    lines.extend([
        "",
        "### Recommended 6–8 Feature Set",
        "",
        "Prioritize candidates that:",
        "- Improve Net Sharpe",
        "- Maintain or reduce turnover",
        "- Add independent information (low collinearity with baseline)",
        "- Show stable IC across regimes",
        "",
    ])
    # Suggest based on ranking
    best = step4_df.nlargest(3, "delta_sharpe_vs_baseline")
    suggested = BASELINE_FEATURES + best["candidate"].tolist()
    lines.append(f"**Proposed:** `{', '.join(suggested)}`")
    lines.append("")
    return "\n".join(lines)


def main():
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, raw = _load_data()
    regime_df = pd.DataFrame()
    try:
        from src.strategy_analyzer import get_hmm_regime_model
        start = raw.index.min().strftime("%Y-%m-%d")
        end = raw.index.max().strftime("%Y-%m-%d")
        _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
    except Exception:
        pass

    print("\n=== STEP 1: Incremental Testing ===")
    step1_df, step1_importance = run_step1_incremental_testing(df)
    step1_ic_df = run_step1_signal_quality(df, regime_df)
    step1_df.to_csv(out_dir / "incremental_step1_results.csv", index=False)
    step1_ic_df.to_csv(out_dir / "incremental_step1_ic.csv", index=False)
    if not step1_importance.empty:
        step1_importance.to_csv(out_dir / "incremental_step1_importance.csv", index=False)
    print(step1_df[["config", "net_sharpe", "net_mdd", "turnover_pct"]].to_string(index=False))

    print("\n=== STEP 2: Collinearity Check ===")
    step2_corr, step2_flags, step2_roll = run_step2_collinearity(df)
    step2_corr.to_csv(out_dir / "incremental_step2_correlation.csv")
    step2_roll.to_csv(out_dir / "incremental_step2_rolling_corr.csv", index=False)
    if step2_flags:
        print("Flags:", step2_flags)
    else:
        print("No |r| > 0.7 pairs.")

    print("\n=== STEP 3: Turnover Impact ===")
    step3_df = run_step3_turnover_impact(step1_df)
    print(step3_df.to_string(index=False))

    print("\n=== STEP 4: Final Selection ===")
    step4_df = run_step4_final_selection(step1_df, step2_flags, step3_df)
    step4_df.to_csv(out_dir / "incremental_step4_ranking.csv", index=False)
    print(step4_df.to_string(index=False))

    report = _render_report(step1_df, step1_ic_df, step1_importance, step2_corr, step2_flags, step2_roll, step3_df, step4_df)
    report_path = out_dir / "incremental_feature_evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
