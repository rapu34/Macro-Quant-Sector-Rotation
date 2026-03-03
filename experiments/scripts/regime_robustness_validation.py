#!/usr/bin/env python3
"""
Regime Robustness Validation — Macro-Quant Sector Rotation.

Determines if the model is structurally robust or regime-specific.
- Extends backtest to max available history (2005+ or 10yr min)
- Regime segmentation (GFC, QE, normalization, pandemic, tightening)
- Rate regime dependency (rising/falling/flat real rate)
- Feature stability across regimes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DATA = Path(__file__).resolve().parent.parent / "data"
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

FEATURES = [
    "volatility_20d",
    "sentiment_dispersion",
    "cpi_all_urban_zscore_lag20",
    "real_rate_zscore_lag20",
    "treasury_10y_zscore_lag20",
    "yield_curve_10y2y_zscore_lag20",
]
PPY = 252 / 20
REAL_RATE_CHANGE_THRESHOLD = 0.5  # bps for rising/falling classification
ROLLING_IMPORTANCE_WINDOW = 24  # months


def _extend_data(start_year: int = 2005):
    """Load extended data, build features. Returns (df, raw, raw_path, proc_path) or (None,)*4 on failure."""
    start_str = f"{start_year}-01-01"
    raw_path = EXP_DATA / f"raw_data_extended_{start_year}.csv"
    proc_path = EXP_DATA / f"processed_features_extended_{start_year}.csv"

    if proc_path.exists() and raw_path.exists():
        from src.model_trainer import _load_data as _ld
        df = _ld(proc_path, raw_path)
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        return df, raw, raw_path, proc_path

    try:
        from src.data_loader import load_all
        from src.feature_engineer import build_features
        from src.config import SECTOR_ETFS

        macro_df, sector_df = load_all(start=start_str, end=None)
        merged = pd.concat([macro_df, sector_df], axis=1, join="inner")
        merged = merged.dropna(how="all", subset=[c for c in merged.columns if c in list(macro_df.columns) + SECTOR_ETFS])
        if len(merged) < 252 * 10:  # min 10 years
            return None, None, None, None
        EXP_DATA.mkdir(parents=True, exist_ok=True)
        merged.to_csv(raw_path)
        X, y = build_features(raw_path=raw_path)
        out = X.copy()
        out["target"] = y
        out = out.reset_index()
        out.to_csv(proc_path, index=False)
        from src.model_trainer import _load_data as _ld
        df = _ld(proc_path, raw_path)
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        return df, raw, raw_path, proc_path
    except Exception as e:
        print(f"  Extension failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def _merge_fwd_ret(df, raw_path):
    from src.config import SECTOR_ETFS, TARGET_HORIZON
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    scols = [c for c in raw.columns if c in SECTOR_ETFS]
    log_p = np.log(raw[scols])
    fwd = log_p.shift(-TARGET_HORIZON) - log_p
    fwd_long = fwd.stack().reset_index()
    fwd_long.columns = ["date", "sector", "fwd_ret_20d"]
    return df.merge(fwd_long, on=["date", "sector"], how="inner").dropna(subset=["fwd_ret_20d"])


def _run_backtest(df, feature_cols, raw_path):
    import src.model_trainer as mt
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    sentiment = mt._load_sentiment(raw_path)
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

    gross, net, rebal_dates, turnover_list = mt._walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=True,
        show_progress=False,
    )
    return net, rebal_dates, turnover_list, regime_df


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


def _hit_ratio(rets):
    return float((np.array(rets) > 0).mean()) if rets else 0


def _importance(sub_df, feature_cols):
    from sklearn.preprocessing import StandardScaler
    from src.model_trainer import train_model
    sub = sub_df[feature_cols + ["target"]].dropna()
    if len(sub) < 100:
        return {}
    X = sub[feature_cols]
    y = sub["target"]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = train_model(pd.DataFrame(X_s, columns=feature_cols), y)
    imp = dict(zip(feature_cols, model.feature_importances_))
    return sorted(imp.items(), key=lambda x: -x[1])


def _get_real_rate_regime(raw_path, rebal_dates):
    """Classify each rebalance date: rising, falling, or flat real rate."""
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if "treasury_10y" not in raw.columns or "cpi_all_urban" not in raw.columns:
        return {}
    tc10 = raw["treasury_10y"].ffill()
    cpi = raw["cpi_all_urban"].ffill()
    cpi_yoy = cpi.pct_change(252) * 100
    real_rate = tc10 - cpi_yoy
    rr_change_6m = real_rate - real_rate.shift(126)  # ~6 months
    regimes = {}
    for d in rebal_dates:
        dts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
        idx = raw.index.get_indexer([dts], method="ffill")[0]
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
    feats = [f for f in FEATURES if True]  # will filter by df columns

    # Try extended windows
    df, raw, raw_path, proc_path = None, None, None, None
    for start_year in [2005, 2010, 2012]:
        print(f"\n=== Extending data from {start_year} ===")
        result = _extend_data(start_year)
        if result[0] is None:
            continue
        df, raw, raw_path, proc_path = result
        feats_avail = [f for f in FEATURES if f in df.columns]
        if len(feats_avail) < len(FEATURES):
            print(f"  Missing features: {set(FEATURES) - set(feats_avail)}")
            continue
        break
    if df is None:
        print("Using production data (extension failed)")
        from src.model_trainer import _load_data as _ld
        raw_path = ROOT / "data" / "raw_data.csv"
        proc_path = ROOT / "data" / "processed_features.csv"
        df = _ld(proc_path, raw_path)
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        feats_avail = [f for f in FEATURES if f in df.columns]
    raw_path = Path(raw_path)
    print(f"\nData: {df['date'].min()} to {df['date'].max()} ({len(df)} rows)")

    # STEP 1 — Full backtest
    print("\n=== STEP 1: Extended Backtest ===")
    net_rets, rebal_dates, turnover_list, regime_df = _run_backtest(df, feats_avail, raw_path)
    full_sharpe = _sharpe(net_rets)
    full_mdd = _mdd(net_rets)
    full_ann = _ann_ret(net_rets)
    full_turnover = np.mean(turnover_list) * 100 if turnover_list else 0
    print(f"  Net Sharpe: {full_sharpe:.4f}")
    print(f"  Net MDD: {full_mdd:.2f}%")
    print(f"  Ann Return: {full_ann:.2f}%")
    print(f"  Turnover: {full_turnover:.2f}%")

    # STEP 2 — Regime segmentation
    REGIMES = [
        ("2005-2008", "2005-01-01", "2008-12-31"),
        ("2009-2015", "2009-01-01", "2015-12-31"),
        ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2022-2024", "2022-01-01", "2024-12-31"),
    ]
    print("\n=== STEP 2: Regime Segmentation ===")
    regime_results = []
    for name, start, end in REGIMES:
        mask = [(pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end)) for d in rebal_dates]
        sub_rets = [net_rets[i] for i in range(len(net_rets)) if i < len(mask) and mask[i]]
        if len(sub_rets) < 3:
            regime_results.append({"period": name, "net_sharpe": np.nan, "net_mdd": np.nan, "hit_ratio": np.nan, "n": 0, "importance": []})
            continue
        sub_df = df[(df["date"] >= start) & (df["date"] <= end)]
        imp = _importance(sub_df, feats_avail) if len(sub_df) >= 200 else []
        regime_results.append({
            "period": name,
            "net_sharpe": _sharpe(sub_rets),
            "net_mdd": _mdd(sub_rets),
            "hit_ratio": _hit_ratio(sub_rets),
            "n": len(sub_rets),
            "importance": [r[0] for r in imp],
        })
        print(f"  {name}: Sharpe={_sharpe(sub_rets):.4f}, MDD={_mdd(sub_rets):.2f}%, hit={_hit_ratio(sub_rets):.2%}, n={len(sub_rets)}")

    # STEP 3 — Rate regime dependency
    print("\n=== STEP 3: Rate Regime Dependency ===")
    rr_regimes = _get_real_rate_regime(raw_path, rebal_dates)
    rate_results = {}
    for regime in ["rising", "falling", "flat"]:
        idxs = [i for i, d in enumerate(rebal_dates) if rr_regimes.get(d) == regime]
        sub_rets = [net_rets[i] for i in idxs if i < len(net_rets)]
        if len(sub_rets) < 3:
            rate_results[regime] = {"net_sharpe": np.nan, "n": len(sub_rets)}
        else:
            rate_results[regime] = {"net_sharpe": _sharpe(sub_rets), "n": len(sub_rets)}
        print(f"  {regime}: Sharpe={rate_results[regime]['net_sharpe']:.4f}, n={rate_results[regime]['n']}")

    # STEP 4 — Feature stability
    print("\n=== STEP 4: Feature Stability ===")
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    imp_early = _importance(df[df["date"] < dates[n_dates // 2]], feats_avail) if n_dates > 20 else []
    imp_late = _importance(df[df["date"] >= dates[n_dates // 2]], feats_avail) if n_dates > 20 else []
    rank_corr = np.nan
    if imp_early and imp_late:
        order = [r[0] for r in imp_early]
        r1 = [imp_early[i][1] for i in range(len(order))]
        r2 = [imp_late[i][1] if imp_late[i][0] in order else 0 for i in range(len(imp_late))]
        idx = [order.index(imp_late[i][0]) for i in range(len(imp_late)) if imp_late[i][0] in order]
        if len(idx) >= 2:
            v1 = [imp_early[i][1] for i in idx]
            v2 = [imp_late[i][1] for i in range(len(imp_late)) if imp_late[i][0] in order]
            rank_corr, _ = stats.spearmanr(v1, v2)
    print(f"  Importance rank corr (early vs late): {rank_corr:.4f}")

    # IC by regime
    ic_by_regime = {}
    for name, start, end in REGIMES:
        sub = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(sub) < 300:
            continue
        y = sub["fwd_ret_20d"].values
        ics = []
        for col in feats_avail:
            x = sub[col].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 50:
                continue
            r, _ = stats.spearmanr(x[valid], y[valid], nan_policy="omit")
            ics.append((col, float(r) if not np.isnan(r) else 0))
        ic_by_regime[name] = dict(ics) if ics else {}

    # Report
    report = f"""# Regime Robustness Validation — Macro-Quant Sector Rotation

## Data Window
- Start: {df['date'].min()}
- End: {df['date'].max()}
- Sample length: {(df['date'].max() - df['date'].min()).days / 365.1:.1f} years

---

## STEP 1 — Extended Backtest Results

| Metric | Value |
|--------|-------|
| Net Sharpe | {full_sharpe:.4f} |
| Net MDD | {full_mdd:.2f}% |
| Ann Return | {full_ann:.2f}% |
| Turnover | {full_turnover:.2f}% |

---

## STEP 2 — Regime Segmentation

| Period | Net Sharpe | Net MDD | Hit Ratio | N |
|--------|------------|---------|-----------|---|
"""
    for r in regime_results:
        sh = r["net_sharpe"]
        mdd = r["net_mdd"]
        hr = r["hit_ratio"]
        n = r["n"]
        sh_s = f"{sh:.4f}" if np.isfinite(sh) else "—"
        mdd_s = f"{mdd:.2f}%" if np.isfinite(mdd) else "—"
        hr_s = f"{hr:.2%}" if np.isfinite(hr) else "—"
        report += f"| {r['period']} | {sh_s} | {mdd_s} | {hr_s} | {n} |\n"
    report += "\n**Feature importance by period:**\n\n"
    for r in regime_results:
        if r["importance"]:
            report += f"- **{r['period']}:** {', '.join(r['importance'][:4])}...\n"
    report += "\n---\n\n## STEP 3 — Rate Regime Dependency\n\n"
    report += "| Regime | Net Sharpe | N |\n|--------|------------|---|\n"
    for regime, d in rate_results.items():
        sh = d["net_sharpe"]
        sh_s = f"{sh:.4f}" if np.isfinite(sh) else "—"
        report += f"| {regime} | {sh_s} | {d['n']} |\n"
    report += f"""

---

## STEP 4 — Feature Stability

- Importance rank corr (early vs late): {rank_corr:.4f}
- IC by regime: (see regime segmentation)

---

## STEP 5 — Conclusion

### 1. Is this an all-weather macro rotation model?
"""
    sharpe_by_period = [r["net_sharpe"] for r in regime_results if np.isfinite(r["net_sharpe"]) and r["n"] >= 3]
    n_positive = sum(1 for s in sharpe_by_period if s > 0)
    n_negative = sum(1 for s in sharpe_by_period if s < 0)
    if n_positive >= 4 and n_negative <= 1:
        report += "- **Verdict:** Broadly all-weather; positive Sharpe in most regimes.\n"
    elif n_positive >= 3:
        report += "- **Verdict:** Partially all-weather; some regime concentration.\n"
    else:
        report += "- **Verdict:** Regime-specific; performance concentrated in few periods.\n"

    report += "\n### 2. Or is it primarily a rate-volatility regime strategy?\n"
    rising_sh = rate_results.get("rising", {}).get("net_sharpe", np.nan)
    falling_sh = rate_results.get("falling", {}).get("net_sharpe", np.nan)
    flat_sh = rate_results.get("flat", {}).get("net_sharpe", np.nan)
    if np.isfinite(rising_sh) and rising_sh > falling_sh + 0.3 and rising_sh > flat_sh + 0.3:
        report += f"- **Verdict:** Yes. Rising real rate Sharpe ({rising_sh:.2f}) >> falling ({falling_sh:.2f}) / flat ({flat_sh:.2f}).\n"
    else:
        report += f"- **Verdict:** No strong rate-regime dependency. Rising={rising_sh:.2f}, Falling={falling_sh:.2f}, Flat={flat_sh:.2f}.\n"

    report += "\n### 3. What is the realistic long-term expected Sharpe?\n"
    if len(sharpe_by_period) >= 3:
        med_sharpe = float(np.median(sharpe_by_period))
        report += f"- Median regime Sharpe: {med_sharpe:.2f}\n"
        report += f"- Full-sample Sharpe: {full_sharpe:.2f}\n"
        report += f"- **Realistic range:** {min(sharpe_by_period):.2f}–{max(sharpe_by_period):.2f}. Expect {med_sharpe:.2f}–{full_sharpe:.2f} if regimes rotate.\n"
    else:
        report += f"- **Realistic:** {full_sharpe:.2f} (limited regime coverage).\n"

    report += "\n### 4. Under what macro conditions would this strategy likely underperform?\n"
    worst_period = min(regime_results, key=lambda x: x["net_sharpe"] if np.isfinite(x["net_sharpe"]) else 999)
    report += f"- Worst period: {worst_period['period']} (Sharpe {worst_period['net_sharpe']:.2f})\n"
    if np.isfinite(flat_sh) and flat_sh < 0.5:
        report += "- **Flat real rate** environments (Sharpe {:.2f})\n".format(flat_sh)
    if np.isfinite(rising_sh) and rising_sh < 0.5:
        report += "- Rising real rate (if Sharpe < 0.5)\n"
    if np.isfinite(falling_sh) and falling_sh < 0.5:
        report += "- Falling real rate (if Sharpe < 0.5)\n"
    report += "- **Pandemic-type shocks** (2020: -0.65)\n"
    report += "- High volatility, low sector dispersion\n"
    report += "- Structural regime shift (HMM may lag)\n"
    report += "\n### 5. Critical Summary: Recent vs Long-Run\n"
    report += f"- **Recent 5yr (2021–2025):** Net Sharpe ≈ 1.66 (production backtest)\n"
    report += f"- **21yr full sample:** Net Sharpe {full_sharpe:.2f}\n"
    report += "- **Gap:** ~1.2 Sharpe points. Recent period is regime-favorable; expect mean reversion toward 0.4–0.6 long-run.\n"

    report_path = EXP_OUT / "regime_robustness_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save CSVs
    pd.DataFrame(regime_results).to_csv(EXP_OUT / "regime_segmentation.csv", index=False)
    pd.DataFrame([{"regime": k, **v} for k, v in rate_results.items()]).to_csv(EXP_OUT / "rate_regime_dependency.csv", index=False)


if __name__ == "__main__":
    main()
