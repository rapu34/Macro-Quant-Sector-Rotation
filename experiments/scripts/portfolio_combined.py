#!/usr/bin/env python3
"""
Combined Portfolio: Block 1 + Block 2 with full risk decomposition.

A) Equal Weight: 50% Block 1, 50% Block 2
B) Inverse Volatility: Rolling 36M vol, weight ∝ 1/vol, rebalance monthly

Goal: Evaluate structural diversification and risk contribution (no weight tuning).
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

ROLLING_36M_DAYS = 36 * 21   # ~756 trading days
ROLLING_12M_DAYS = 252
REBALANCE_DAYS = 21          # monthly for InvVol
PPY_DAILY = 252
CVAR_ALPHA = 0.95


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


def _get_block1_returns(raw_path, proc_path):
    """Run Block 1 backtest and return daily_rets."""
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
    )
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    daily_rets = _period_to_daily_returns(net, rebal_dates, raw.index, 20)
    return daily_rets


def _get_block2_returns(raw_path):
    """Run Block 2 backtest and return daily_rets."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "block2", Path(__file__).resolve().parent / "block2_momentum_extended.py"
    )
    block2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(block2)
    _, _, _, daily_rets = block2._run_block2_backtest(raw_path)
    return daily_rets


def _align_daily_returns(b1, b2):
    """Align Block 1 and Block 2 daily returns on common dates."""
    common = b1.index.intersection(b2.index)
    b1_a = b1.reindex(common).ffill().bfill().fillna(0)
    b2_a = b2.reindex(common).ffill().bfill().fillna(0)
    return b1_a, b2_a


def _build_inverse_vol_weights(b1, b2):
    """Build inverse volatility weights, rebalanced monthly."""
    roll_vol_b1 = b1.rolling(ROLLING_36M_DAYS, min_periods=ROLLING_36M_DAYS).std() * np.sqrt(PPY_DAILY)
    roll_vol_b2 = b2.rolling(ROLLING_36M_DAYS, min_periods=ROLLING_36M_DAYS).std() * np.sqrt(PPY_DAILY)
    inv_vol_b1 = 1.0 / roll_vol_b1.replace(0, np.nan)
    inv_vol_b2 = 1.0 / roll_vol_b2.replace(0, np.nan)
    total = inv_vol_b1.fillna(0) + inv_vol_b2.fillna(0)
    w1 = (inv_vol_b1 / total).fillna(0.5)
    w2 = (inv_vol_b2 / total).fillna(0.5)
    # Rebalance monthly: use weights at month-end
    w1_rebal = w1.resample("ME").last().reindex(b1.index).ffill()
    w2_rebal = w2.resample("ME").last().reindex(b2.index).ffill()
    w1_rebal = w1_rebal.fillna(0.5)
    w2_rebal = w2_rebal.fillna(0.5)
    return w1_rebal, w2_rebal


def _sharpe(rets):
    arr = np.array(rets)
    if len(arr) < 2 or arr.std() < 1e-10:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(PPY_DAILY))


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
    n = len(arr)
    return float(np.prod(1 + arr) ** (PPY_DAILY / n) - 1) * 100 if n > 0 else 0


def _volatility(rets):
    arr = np.array(rets)
    return float(arr.std() * np.sqrt(PPY_DAILY)) * 100 if len(arr) > 1 else 0


def _cvar95(rets):
    arr = np.array(rets)
    n_tail = max(1, int(len(arr) * (1 - CVAR_ALPHA)))
    worst = np.partition(arr, n_tail - 1)[:n_tail]
    return float(np.mean(worst)) * 100


def _rolling_12m_sharpe(rets):
    roll = rets.rolling(ROLLING_12M_DAYS, min_periods=ROLLING_12M_DAYS)
    return roll.apply(lambda x: _sharpe(x.tolist()) if len(x) >= 2 and x.std() > 1e-10 else np.nan)


def _vol_decomposition(w1, w2, cov_matrix):
    """Variance decomposition: total var, covariance term, % contribution."""
    var1 = cov_matrix.iloc[0, 0]
    var2 = cov_matrix.iloc[1, 1]
    cov12 = cov_matrix.iloc[0, 1]
    var_p = w1**2 * var1 + w2**2 * var2 + 2 * w1 * w2 * cov12
    contrib1 = w1**2 * var1 + w1 * w2 * cov12
    contrib2 = w2**2 * var2 + w1 * w2 * cov12
    pct1 = 100 * contrib1 / var_p if var_p > 1e-12 else 50
    pct2 = 100 * contrib2 / var_p if var_p > 1e-12 else 50
    return var_p, cov12, pct1, pct2


def _mcr(w, cov_vector, var_p):
    """Marginal contribution to risk: w_i * Cov(R_i, R_p) / Var(R_p)."""
    return w * cov_vector / var_p if var_p > 1e-12 else 0.5


def _load_vix(raw_path):
    """Load VIX if available (for stress correlation)."""
    try:
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        start = raw.index.min().strftime("%Y-%m-%d")
        end = raw.index.max().strftime("%Y-%m-%d")
        import yfinance as yf
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if vix.empty:
            return None
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix["Close"] if "Close" in vix.columns.get_level_values(1) else vix.iloc[:, 0]
        else:
            vix = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0]
        return vix.squeeze()
    except Exception:
        return None


def main():
    EXP_OUT.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "raw_data_extended_2005.csv"
    proc_path = DATA_DIR / "processed_features_extended_2005.csv"
    if not raw_path.exists() and (EXP_DATA / "raw_data_extended_2005.csv").exists():
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(EXP_DATA / "raw_data_extended_2005.csv", raw_path)
        shutil.copy(EXP_DATA / "processed_features_extended_2005.csv", proc_path)

    print("\n=== Combined Portfolio: Block 1 + Block 2 ===\n")

    # Load daily returns
    print("Loading Block 1 and Block 2 returns...")
    b1 = _get_block1_returns(raw_path, proc_path)
    b2 = _get_block2_returns(raw_path)
    b1, b2 = _align_daily_returns(b1, b2)

    # STEP 1 — Portfolio construction
    print("STEP 1: Portfolio Construction")
    ew_ret = 0.5 * b1 + 0.5 * b2

    w1_inv, w2_inv = _build_inverse_vol_weights(b1, b2)
    inv_vol_ret = w1_inv * b1 + w2_inv * b2

    print("  Equal Weight: 50% Block 1, 50% Block 2")
    print("  Inverse Vol: Rolling 36M vol, rebalance monthly")

    # STEP 2 — Portfolio performance
    print("\nSTEP 2: Portfolio Performance")
    strategies = {
        "Block 1": b1,
        "Block 2": b2,
        "Equal Weight": ew_ret,
        "Inverse Vol": inv_vol_ret,
    }
    perf = {}
    for name, rets in strategies.items():
        rets_clean = rets.dropna()
        perf[name] = {
            "sharpe": _sharpe(rets_clean),
            "mdd": _mdd(rets_clean),
            "ann_ret": _ann_ret(rets_clean),
            "vol": _volatility(rets_clean),
            "cvar95": _cvar95(rets_clean),
        }
        roll_sh = _rolling_12m_sharpe(rets_clean)
        roll_valid = roll_sh.dropna()
        perf[name]["roll_sh_mean"] = float(roll_valid.mean()) if len(roll_valid) > 0 else np.nan
        perf[name]["roll_sh_std"] = float(roll_valid.std()) if len(roll_valid) > 1 else np.nan

    for name, p in perf.items():
        print(f"  {name}: Sharpe={p['sharpe']:.4f}, MDD={p['mdd']:.2f}%, AnnRet={p['ann_ret']:.2f}%, Vol={p['vol']:.2f}%, CVaR95={p['cvar95']:.2f}%")

    # STEP 3 — Risk decomposition
    print("\nSTEP 3: Risk Decomposition")
    df_ret = pd.DataFrame({"b1": b1, "b2": b2})
    cov_full = df_ret.cov() * PPY_DAILY  # annualized cov

    # Equal Weight
    w1_ew, w2_ew = 0.5, 0.5
    ret_ew = w1_ew * b1 + w2_ew * b2
    var_ew = cov_full.iloc[0, 0] * w1_ew**2 + cov_full.iloc[1, 1] * w2_ew**2 + 2 * w1_ew * w2_ew * cov_full.iloc[0, 1]
    cov_b1_ew = np.cov(b1.values, ret_ew.values)[0, 1] * PPY_DAILY
    cov_b2_ew = np.cov(b2.values, ret_ew.values)[1, 0] * PPY_DAILY
    var_ew_emp = np.var(ret_ew.values) * PPY_DAILY
    _, cov_term_ew, pct1_ew, pct2_ew = _vol_decomposition(w1_ew, w2_ew, cov_full)
    mcr1_ew = 100 * _mcr(w1_ew, cov_b1_ew, var_ew_emp)
    mcr2_ew = 100 * _mcr(w2_ew, cov_b2_ew, var_ew_emp)

    # Inverse Vol (use average weights for decomposition)
    w1_iv_mean = float(w1_inv.mean())
    w2_iv_mean = float(w2_inv.mean())
    ret_iv = w1_inv * b1 + w2_inv * b2
    cov_b1_iv = np.cov(b1.values, ret_iv.values)[0, 1] * PPY_DAILY
    cov_b2_iv = np.cov(b2.values, ret_iv.values)[1, 0] * PPY_DAILY
    var_iv_emp = np.var(ret_iv.values) * PPY_DAILY
    _, cov_term_iv, pct1_iv, pct2_iv = _vol_decomposition(w1_iv_mean, w2_iv_mean, cov_full)
    mcr1_iv = 100 * _mcr(w1_iv_mean, cov_b1_iv, var_iv_emp)
    mcr2_iv = 100 * _mcr(w2_iv_mean, cov_b2_iv, var_iv_emp)

    risk_decomp = [
        {"portfolio": "Equal Weight", "total_var": var_ew_emp, "cov_term": cov_full.iloc[0, 1], "pct_b1": pct1_ew, "pct_b2": pct2_ew, "mcr_b1": mcr1_ew, "mcr_b2": mcr2_ew},
        {"portfolio": "Inverse Vol", "total_var": var_iv_emp, "cov_term": cov_full.iloc[0, 1], "pct_b1": pct1_iv, "pct_b2": pct2_iv, "mcr_b1": mcr1_iv, "mcr_b2": mcr2_iv},
    ]

    # STEP 4 — Stress & conditional correlation
    print("\nSTEP 4: Stress & Conditional Correlation")
    corr_2008 = float(b1.loc["2008-01-01":"2008-12-31"].corr(b2.loc["2008-01-01":"2008-12-31"])) if len(b1.loc["2008-01-01":"2008-12-31"]) > 10 else np.nan
    corr_2020 = float(b1.loc["2020-01-01":"2020-12-31"].corr(b2.loc["2020-01-01":"2020-12-31"])) if len(b1.loc["2020-01-01":"2020-12-31"]) > 10 else np.nan
    corr_2022 = float(b1.loc["2022-01-01":"2022-12-31"].corr(b2.loc["2022-01-01":"2022-12-31"])) if len(b1.loc["2022-01-01":"2022-12-31"]) > 10 else np.nan

    roll_corr = b1.rolling(ROLLING_36M_DAYS, min_periods=ROLLING_36M_DAYS).corr(b2)
    roll_corr_valid = roll_corr.dropna()
    roll_corr_mean = float(roll_corr_valid.mean()) if len(roll_corr_valid) > 0 else np.nan
    roll_corr_max = float(roll_corr_valid.max()) if len(roll_corr_valid) > 0 else np.nan

    # Top 5% VIX days
    vix = _load_vix(raw_path)
    corr_vix_top5 = np.nan
    if vix is not None and not vix.empty:
        common = b1.index.intersection(vix.index)
        b1_v = b1.reindex(common).ffill().bfill()
        b2_v = b2.reindex(common).ffill().bfill()
        vix_a = vix.reindex(common).ffill().bfill()
        thresh = vix_a.quantile(0.95)
        mask = vix_a >= thresh
        if mask.sum() > 10:
            corr_vix_top5 = float(b1_v.loc[mask].corr(b2_v.loc[mask]))

    # Worst 10% portfolio return days (use EW)
    worst_10_pct = ew_ret.quantile(0.10)
    mask_worst = ew_ret <= worst_10_pct
    corr_worst10 = float(b1.loc[mask_worst].corr(b2.loc[mask_worst])) if mask_worst.sum() > 10 else np.nan

    print(f"  2008: {corr_2008:.4f}")
    print(f"  2020: {corr_2020:.4f}")
    print(f"  2022: {corr_2022:.4f}")
    print(f"  Top 5% VIX days: {corr_vix_top5:.4f}" if not np.isnan(corr_vix_top5) else "  Top 5% VIX days: N/A")
    print(f"  Worst 10% portfolio days: {corr_worst10:.4f}")
    print(f"  Rolling 36M: mean={roll_corr_mean:.4f}, max={roll_corr_max:.4f}")

    # STEP 5 — Crisis performance comparison
    print("\nSTEP 5: Crisis Performance Comparison")
    crises = [("2008", "2008-01-01", "2008-12-31"), ("2020", "2020-01-01", "2020-12-31"), ("2022", "2022-01-01", "2022-12-31")]
    crisis_results = {name: [] for name in strategies}
    for name, start, end in crises:
        for strat_name, rets in strategies.items():
            sub = rets.loc[start:end].dropna()
            if len(sub) >= 2:
                crisis_results[strat_name].append({
                    "period": name,
                    "sharpe": _sharpe(sub),
                    "mdd": _mdd(sub),
                    "ret": _ann_ret(sub),
                    "n": len(sub),
                })
            else:
                crisis_results[strat_name].append({"period": name, "sharpe": np.nan, "mdd": np.nan, "ret": np.nan, "n": len(sub)})

    corr_vix_str = f"{corr_vix_top5:.4f}" if not np.isnan(corr_vix_top5) else "N/A"
    corr_2008_str = f"{corr_2008:.4f}" if not np.isnan(corr_2008) else "N/A"

    # STEP 6 — Deliverables
    # portfolio_combined_returns.csv
    ret_df = pd.DataFrame({
        "date": b1.index,
        "block1": b1.values,
        "block2": b2.values,
        "equal_weight": ew_ret.values,
        "inverse_vol": inv_vol_ret.values,
    })
    ret_df.to_csv(EXP_OUT / "portfolio_combined_returns.csv", index=False)

    # risk_decomposition.csv
    pd.DataFrame(risk_decomp).to_csv(EXP_OUT / "risk_decomposition.csv", index=False)

    # rolling_correlation.csv
    roll_corr_df = pd.DataFrame({"date": roll_corr.index, "rolling_36m_corr": roll_corr.values}).dropna()
    roll_corr_df.to_csv(EXP_OUT / "rolling_correlation.csv", index=False)

    # Report
    report = f"""# Combined Portfolio Report

> Block 1 + Block 2. Structural diversification and risk contribution analysis.

## Setup

- **Data**: 2005-01-03 ~ 2026-01-28 (extended)
- **A) Equal Weight**: 50% Block 1, 50% Block 2
- **B) Inverse Volatility**: Rolling 36M vol, weight ∝ 1/vol, rebalance monthly
- **Portfolio-layer cost**: Ignored (included in block returns)

---

## STEP 2 — Portfolio Performance

| Strategy | Sharpe | MDD | Ann Return | Vol | CVaR (95%) | Roll 12M Sharpe (mean) | Roll 12M Sharpe (std) |
|----------|--------|-----|------------|-----|------------|------------------------|----------------------|
"""
    for name, p in perf.items():
        report += f"| {name} | {p['sharpe']:.4f} | {p['mdd']:.2f}% | {p['ann_ret']:.2f}% | {p['vol']:.2f}% | {p['cvar95']:.2f}% | {p['roll_sh_mean']:.4f} | {p['roll_sh_std']:.4f} |\n"

    report += """
## STEP 3 — Risk Decomposition

### Volatility Decomposition

| Portfolio | Total Var (ann) | Cov Term | % Block 1 | % Block 2 |
|-----------|-----------------|----------|-----------|-----------|
"""
    for r in risk_decomp:
        report += f"| {r['portfolio']} | {r['total_var']:.6f} | {r['cov_term']:.6f} | {r['pct_b1']:.1f}% | {r['pct_b2']:.1f}% |\n"

    report += """
### Marginal Contribution to Risk (MCR)
MCR_i = w_i * Cov(R_i, R_p) / Var(R_p). Reported as % risk contribution.

| Portfolio | MCR Block 1 (%) | MCR Block 2 (%) |
|-----------|-----------------|-----------------|
"""
    for r in risk_decomp:
        report += f"| {r['portfolio']} | {r['mcr_b1']:.1f}% | {r['mcr_b2']:.1f}% |\n"

    report += f"""
## STEP 4 — Stress & Conditional Correlation

| Condition | Correlation |
|-----------|-------------|
| 2008 | {corr_2008_str} |
| 2020 | {corr_2020:.4f} |
| 2022 | {corr_2022:.4f} |
| Top 5% VIX days | {corr_vix_str} |
| Worst 10% portfolio return days | {corr_worst10:.4f} |
| Rolling 36M (mean) | {roll_corr_mean:.4f} |
| Rolling 36M (max) | {roll_corr_max:.4f} |

*Top 5% VIX days: N/A if VIX data unavailable.*

---

## STEP 5 — Crisis Performance Comparison

### 2008

| Strategy | Sharpe | MDD | Return |
|----------|--------|-----|--------|
"""
    for strat_name in strategies:
        r = next((x for x in crisis_results[strat_name] if x["period"] == "2008"), {})
        sh = f"{r.get('sharpe', np.nan):.4f}" if not np.isnan(r.get('sharpe', np.nan)) else "—"
        mdd = f"{r.get('mdd', np.nan):.2f}%" if not np.isnan(r.get('mdd', np.nan)) else "—"
        ret = f"{r.get('ret', np.nan):.2f}%" if not np.isnan(r.get('ret', np.nan)) else "—"
        report += f"| {strat_name} | {sh} | {mdd} | {ret} |\n"

    report += """
### 2020

| Strategy | Sharpe | MDD | Return |
|----------|--------|-----|--------|
"""
    for strat_name in strategies:
        r = next((x for x in crisis_results[strat_name] if x["period"] == "2020"), {})
        sh = f"{r.get('sharpe', np.nan):.4f}" if not np.isnan(r.get('sharpe', np.nan)) else "—"
        mdd = f"{r.get('mdd', np.nan):.2f}%" if not np.isnan(r.get('mdd', np.nan)) else "—"
        ret = f"{r.get('ret', np.nan):.2f}%" if not np.isnan(r.get('ret', np.nan)) else "—"
        report += f"| {strat_name} | {sh} | {mdd} | {ret} |\n"

    report += """
### 2022

| Strategy | Sharpe | MDD | Return |
|----------|--------|-----|--------|
"""
    for strat_name in strategies:
        r = next((x for x in crisis_results[strat_name] if x["period"] == "2022"), {})
        sh = f"{r.get('sharpe', np.nan):.4f}" if not np.isnan(r.get('sharpe', np.nan)) else "—"
        mdd = f"{r.get('mdd', np.nan):.2f}%" if not np.isnan(r.get('mdd', np.nan)) else "—"
        ret = f"{r.get('ret', np.nan):.2f}%" if not np.isnan(r.get('ret', np.nan)) else "—"
        report += f"| {strat_name} | {sh} | {mdd} | {ret} |\n"

    report += """
---
*Generated by experiments/scripts/portfolio_combined.py*
*No weight tuning. Goal: evaluate structural diversification and risk contribution.*
"""

    with open(EXP_OUT / "portfolio_combined_report.md", "w") as f:
        f.write(report)
    print(f"\nSaved: {EXP_OUT / 'portfolio_combined_report.md'}")
    print(f"Saved: {EXP_OUT / 'portfolio_combined_returns.csv'}")
    print(f"Saved: {EXP_OUT / 'risk_decomposition.csv'}")
    print(f"Saved: {EXP_OUT / 'rolling_correlation.csv'}")


if __name__ == "__main__":
    main()
