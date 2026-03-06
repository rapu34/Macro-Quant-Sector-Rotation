#!/usr/bin/env python3
"""
Factor Exposure Regression for final portfolio (30% Block1 + 70% Block2_HMM_REBAL_ONLY).

Run: python experiments/scripts/factor_regression.py

True Daily PnL. OLS regression.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

_suffix = "_refresh" if os.environ.get("PIPELINE_REFRESH_MODE") == "1" else ""
EXP_OUT = Path(__file__).resolve().parent.parent / f"outputs{_suffix}"
PPY = 252
ROLLING_36M_DAYS = 36 * 21
W_B1 = 0.3
W_B2 = 0.7


def _fetch_factor_returns(start: str, end: str) -> pd.DataFrame:
    """Fetch factor data: SPY, IWM, MTUM, TLT, VIX."""
    import yfinance as yf

    tickers = ["SPY", "IWM", "MTUM", "TLT", "^VIX"]
    out = {}
    for t in tickers:
        try:
            d = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if d.empty:
                continue
            c = d["Close"] if "Close" in d.columns else d.iloc[:, 0]
            if isinstance(c, pd.DataFrame):
                c = c.squeeze()
            out[t] = c
        except Exception:
            continue

    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    df = df.ffill().dropna(how="all")

    # Returns
    factors = pd.DataFrame(index=df.index)
    if "SPY" in df.columns:
        factors["r_mkt"] = df["SPY"].pct_change()
    if "IWM" in df.columns:
        factors["r_size"] = df["IWM"].pct_change()
    if "MTUM" in df.columns:
        factors["r_mom"] = df["MTUM"].pct_change()
    elif "SPY" in df.columns:
        factors["r_mom"] = df["SPY"].pct_change()
    if "TLT" in df.columns:
        factors["r_tlt"] = df["TLT"].pct_change()
    if "^VIX" in df.columns:
        factors["delta_vix"] = df["^VIX"].pct_change()

    return factors.dropna(how="all")


def main():
    print("\n=== Factor Exposure Regression ===\n")

    block1 = pd.read_csv(EXP_OUT / "true_daily_block1.csv", index_col=0, parse_dates=True).squeeze()
    block2 = pd.read_csv(EXP_OUT / "block2_hmm_expanding_rebalonly.csv", index_col=0, parse_dates=True)
    block2 = block2.iloc[:, 0].squeeze()

    common = block1.index.intersection(block2.index)
    r_b1 = block1.reindex(common).ffill().bfill().fillna(0)
    r_b2 = block2.reindex(common).ffill().bfill().fillna(0)
    r_p = W_B1 * r_b1 + W_B2 * r_b2

    start = r_p.index.min().strftime("%Y-%m-%d")
    end = r_p.index.max().strftime("%Y-%m-%d")
    print(f"Portfolio period: {start} ~ {end}")

    print("Fetching factor data...")
    factors = _fetch_factor_returns(start, end)
    factor_cols = [c for c in ["r_mkt", "r_size", "r_mom", "r_tlt", "delta_vix"] if c in factors.columns]
    if len(factor_cols) < 2:
        raise RuntimeError("Insufficient factor data. Need at least r_mkt.")

    merged = pd.concat([r_p.rename("r_p"), factors[factor_cols]], axis=1, sort=True).dropna()
    X = merged[factor_cols].values
    y = merged["r_p"].values
    n, k = X.shape

    # OLS via numpy (avoid statsmodels/scipy compatibility issues)
    X_const = np.column_stack([np.ones(n), X])
    xtx_inv = np.linalg.inv(X_const.T @ X_const)
    coef = xtx_inv @ X_const.T @ y
    y_hat = X_const @ coef
    resid = y - y_hat
    mse = np.sum(resid ** 2) / (n - k - 1) if n > k + 1 else 0
    se = np.sqrt(np.diag(xtx_inv) * mse)
    tstats = coef / se if np.all(se > 1e-12) else np.zeros_like(coef)
    r2 = 1 - np.var(resid) / np.var(y) if np.var(y) > 1e-12 else 0
    resid_vol = np.std(resid)
    resid_vol_ann = resid_vol * np.sqrt(PPY)

    alpha = coef[0]
    alpha_ann = alpha * PPY
    betas = {factor_cols[i]: coef[i + 1] for i in range(len(factor_cols))}
    tstat_dict = {factor_cols[i]: tstats[i + 1] for i in range(len(factor_cols))}
    tstat_alpha = tstats[0]

    summary_rows = [
        {"factor": "alpha", "coef": alpha, "t_stat": tstat_alpha, "alpha_ann_pct": alpha_ann * 100},
    ]
    for c in factor_cols:
        summary_rows.append({"factor": c, "coef": betas[c], "t_stat": tstat_dict[c]})
    summary_rows.append({"factor": "R2", "coef": r2, "t_stat": np.nan})
    summary_rows.append({"factor": "resid_vol_ann", "coef": resid_vol_ann, "t_stat": np.nan})

    df_summary = pd.DataFrame(summary_rows)

    # Rolling 36M beta_mkt
    r_mkt_col = "r_mkt" if "r_mkt" in factor_cols else factor_cols[0]
    roll_betas = []

    def _ols_simple(y_arr, x_arr):
        n = len(y_arr)
        X = np.column_stack([np.ones(n), x_arr])
        try:
            coef = np.linalg.lstsq(X, y_arr, rcond=None)[0]
            y_hat = X @ coef
            r2 = 1 - np.var(y_arr - y_hat) / np.var(y_arr) if np.var(y_arr) > 1e-12 else 0
            return coef[0], coef[1], r2
        except Exception:
            return np.nan, np.nan, np.nan

    for i in range(ROLLING_36M_DAYS, len(merged) + 1):
        sub = merged.iloc[i - ROLLING_36M_DAYS : i]
        if len(sub) < int(ROLLING_36M_DAYS * 0.8):
            continue
        alpha_r, beta_r, r2_r = _ols_simple(sub["r_p"].values, sub[r_mkt_col].values)
        roll_betas.append({"date": sub.index[-1], "beta_mkt": beta_r, "alpha": alpha_r, "r2": r2_r})
    df_roll = pd.DataFrame(roll_betas)

    # Crisis period regressions
    crisis_results = []
    for name, c_start, c_end in [("2020", "2020-01-01", "2020-12-31"), ("2022", "2022-01-01", "2022-12-31")]:
        mask = (merged.index >= c_start) & (merged.index <= c_end)
        sub = merged.loc[mask]
        if len(sub) < 50:
            crisis_results.append({"period": name, "beta_mkt": np.nan, "alpha_ann": np.nan, "r2": np.nan, "n": len(sub)})
            continue
        X_c = np.column_stack([np.ones(len(sub)), sub[factor_cols].values])
        try:
            coef_c = np.linalg.lstsq(X_c, sub["r_p"].values, rcond=None)[0]
            y_hat_c = X_c @ coef_c
            r2_c = 1 - np.var(sub["r_p"].values - y_hat_c) / np.var(sub["r_p"].values) if np.var(sub["r_p"].values) > 1e-12 else 0
            idx_mkt = factor_cols.index("r_mkt") + 1 if "r_mkt" in factor_cols else 1
            beta_mkt_c = coef_c[idx_mkt]
            crisis_results.append({
                "period": name,
                "beta_mkt": beta_mkt_c,
                "alpha_ann": coef_c[0] * PPY * 100,
                "r2": r2_c,
                "n": len(sub),
            })
        except Exception:
            crisis_results.append({"period": name, "beta_mkt": np.nan, "alpha_ann": np.nan, "r2": np.nan, "n": len(sub)})
    df_crisis = pd.DataFrame(crisis_results)

    # Interpretations
    beta_mkt = betas.get("r_mkt", np.nan)
    market_below_one = beta_mkt < 1.0 if not np.isnan(beta_mkt) else None
    beta_dur = betas.get("r_tlt", np.nan)
    beta_vol = betas.get("delta_vix", np.nan)
    has_alpha = abs(tstat_alpha) > 1.96

    # Save
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(EXP_OUT / "factor_regression_summary.csv", index=False)
    df_roll.to_csv(EXP_OUT / "factor_rolling_betas.csv", index=False)
    df_crisis.to_csv(EXP_OUT / "factor_regression_crisis.csv", index=False)

    report = f"""# Factor Exposure Regression Report

Run: `python experiments/scripts/factor_regression.py`

Portfolio: 30% Block1 + 70% Block2_HMM_REBAL_ONLY. True Daily PnL. OLS.

## Model

r_p = alpha + beta_mkt * r_mkt + beta_size * r_size + beta_mom * r_mom + beta_dur * r_tlt + beta_vol * ΔVIX + ε

## Full-Sample Regression (2005–2026)

| Factor | Coefficient | t-stat |
|--------|-------------|--------|
| alpha | {alpha:.6f} | {tstat_alpha:.2f} |
"""
    for c in factor_cols:
        report += f"| {c} | {betas[c]:.4f} | {tstat_dict[c]:.2f} |\n"

    report += f"""
- **R²**: {r2:.4f}
- **Alpha (annualized)**: {alpha_ann*100:.2f}%
- **Residual volatility (annualized)**: {resid_vol_ann*100:.2f}%

## Main Beta Interpretation

- **Market (r_mkt)**: beta = {beta_mkt:.4f}. {'Market beta < 1: portfolio has lower systematic market exposure than the market.' if market_below_one else 'Market beta ≥ 1: portfolio has at least full market exposure.'}
- **Duration (r_tlt)**: beta = {beta_dur:.4f}. {'Negative: portfolio tends to fall when bonds rally (rate-sensitive).' if not np.isnan(beta_dur) and beta_dur < 0 else 'Positive: portfolio tends to rise with bonds.' if not np.isnan(beta_dur) else 'N/A'}
- **Volatility (ΔVIX)**: beta = {beta_vol:.4f}. {'Negative: portfolio tends to fall when VIX rises (typical for long equity).' if not np.isnan(beta_vol) and beta_vol < 0 else 'Positive: unusual.' if not np.isnan(beta_vol) else 'N/A'}

## Market Beta < 1?

**{'Yes' if market_below_one else 'No'}** — beta_mkt = {beta_mkt:.4f}

## Duration / Vol Sensitivity

| Factor | Beta | Interpretation |
|--------|------|-----------------|
| TLT (duration) | {beta_dur:.4f} | {'Rate sensitivity' if not np.isnan(beta_dur) else 'N/A'} |
| ΔVIX (vol) | {beta_vol:.4f} | {'Vol sensitivity' if not np.isnan(beta_vol) else 'N/A'} |

## Residual Alpha Existence?

**{'Yes' if has_alpha else 'No'}** — alpha t-stat = {tstat_alpha:.2f} (|t| > 1.96 implies significant alpha)

## Crisis Period Betas

| Period | beta_mkt | alpha_ann (%) | R² | N |
|--------|----------|---------------|-----|---|
"""
    for _, r in df_crisis.iterrows():
        report += f"| {r['period']} | {r['beta_mkt']:.4f} | {r['alpha_ann']:.2f} | {r['r2']:.4f} | {r['n']} |\n"

    report += """
## Rolling 36M Market Beta

See `factor_rolling_betas.csv` for time series of beta_mkt.

---
*Generated by factor_regression.py*
"""

    with open(EXP_OUT / "factor_regression_report.md", "w") as f:
        f.write(report)

    print("\nRegression summary:")
    print(df_summary.to_string(index=False))
    print(f"\nAlpha (ann): {alpha_ann*100:.2f}%")
    print(f"Resid vol (ann): {resid_vol_ann*100:.2f}%")
    print(f"Market beta < 1: {market_below_one}")
    print(f"Significant alpha: {has_alpha}")
    print(f"\nCrisis betas:")
    print(df_crisis.to_string(index=False))
    print(f"\nSaved: {EXP_OUT / 'factor_regression_summary.csv'}")
    print(f"Saved: {EXP_OUT / 'factor_rolling_betas.csv'}")
    print(f"Saved: {EXP_OUT / 'factor_regression_report.md'}")


if __name__ == "__main__":
    main()
