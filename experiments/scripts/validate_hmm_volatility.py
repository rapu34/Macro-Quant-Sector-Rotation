"""
HMM P_Crisis Volatility Predictive Power Validation

Test whether P_crisis predicts forward 20-day realized volatility.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

REBALANCE_DAYS = 20
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU"]


def main():
    pc_path = ROOT / "outputs" / "p_crisis_log.csv"
    raw_path = ROOT / "data" / "raw_data.csv"

    if not pc_path.exists():
        print(f"[ERROR] {pc_path} not found. Run model_trainer first.")
        return
    if not raw_path.exists():
        print(f"[ERROR] {raw_path} not found.")
        return

    pc_df = pd.read_csv(pc_path)
    pc_df["date"] = pd.to_datetime(pc_df["date"])

    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    sector_cols = [c for c in raw.columns if c in SECTOR_ETFS]
    if not sector_cols:
        print("[ERROR] No sector columns in raw_data.")
        return

    # Daily log returns (equal-weighted sectors)
    prices = raw[sector_cols]
    log_prices = np.log(prices)
    daily_ret = log_prices.diff().dropna()
    ew_daily = daily_ret.mean(axis=1)
    dates_arr = ew_daily.index

    # For each rebalance date: forward 20-day realized vol = std(daily returns) * sqrt(252)
    forward_vols = []
    for d in pc_df["date"]:
        idx = dates_arr.get_indexer([pd.Timestamp(d)], method="nearest")[0]
        if idx < 0 or idx + REBALANCE_DAYS >= len(ew_daily):
            forward_vols.append(np.nan)
            continue
        window = ew_daily.iloc[idx + 1 : idx + REBALANCE_DAYS + 1]
        if len(window) < REBALANCE_DAYS:
            forward_vols.append(np.nan)
            continue
        vol = np.std(window) * np.sqrt(252)  # annualized
        forward_vols.append(vol)

    pc_df["forward_20d_vol"] = forward_vols
    pc_df = pc_df.dropna(subset=["forward_20d_vol"])

    if len(pc_df) < 10:
        print("[ERROR] Too few valid rows for analysis.")
        return

    X = pc_df["p_crisis"].values.reshape(-1, 1)
    y = pc_df["forward_20d_vol"].values
    n = len(y)

    # --- 1. Regression: Forward_20d_Vol ~ P_crisis ---
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    y_pred = reg.predict(X)
    resid = y - y_pred
    x_var = np.var(X.ravel())
    se_slope = np.sqrt(np.sum(resid**2) / max(1, n - 2)) / np.sqrt(max(1e-12, n * x_var))
    t_slope = slope / se_slope if se_slope > 1e-12 else 0
    pval_slope = 2 * (1 - stats.t.cdf(abs(t_slope), max(1, n - 2)))

    # --- 2. Spearman correlation ---
    rho, p_rho = stats.spearmanr(pc_df["p_crisis"], pc_df["forward_20d_vol"])

    # --- 3. Median split ---
    med = pc_df["p_crisis"].median()
    high = pc_df[pc_df["p_crisis"] > med]["forward_20d_vol"]
    low = pc_df[pc_df["p_crisis"] <= med]["forward_20d_vol"]
    t_stat, p_ttest = stats.ttest_ind(high, low)

    # --- Report ---
    print("\n" + "=" * 60)
    print("HMM P_Crisis vs Forward 20-Day Realized Volatility")
    print("=" * 60)
    print("\nForward vol: std(daily returns) * sqrt(252), annualized")
    print(f"Sample: {n} rebalance periods")
    print(f"P_crisis range: [{pc_df['p_crisis'].min():.4f}, {pc_df['p_crisis'].max():.4f}]")
    print(f"Forward vol mean: {y.mean()*100:.2f}%, std: {y.std()*100:.2f}%")

    print("\n--- 1. Regression: Forward_20d_Vol ~ P_crisis ---")
    print(f"  Coefficient (β): {slope:.4f}  (positive = predictive)")
    print(f"  t-stat: {t_slope:.3f}, p-value: {pval_slope:.4f}")
    print("  → Significant at 5%" if pval_slope < 0.05 else "  → NOT significant at 5%")

    print("\n--- 2. Spearman Correlation ---")
    print(f"  ρ = {rho:.4f}, p-value: {p_rho:.4f}")
    print("  → Significant" if p_rho < 0.05 else "  → NOT significant")

    print("\n--- 3. Median Split (High vs Low P_crisis) ---")
    print(f"  High P_crisis (>{med:.3f}): mean vol = {high.mean()*100:.2f}%, n={len(high)}")
    print(f"  Low  P_crisis (≤{med:.3f}): mean vol = {low.mean()*100:.2f}%, n={len(low)}")
    print(f"  Diff: {(high.mean() - low.mean())*100:.2f}%, p={p_ttest:.4f}")

    print("\n" + "=" * 60)
    verdict = (
        "P_crisis predicts forward volatility (crisis → higher vol)."
        if (pval_slope < 0.05 or p_rho < 0.05) and slope > 0
        else "P_crisis does NOT significantly predict forward volatility."
    )
    print("VERDICT:", verdict)
    print("=" * 60 + "\n")

    out_path = EXP_OUT / "hmm_volatility_validation.csv"
    pc_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
