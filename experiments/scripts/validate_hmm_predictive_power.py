"""
HMM P_Crisis Predictive Power Validation

핵심 질문: Crisis regime에서 실제로 다음 1개월 수익률이 유의미하게 낮았는가?

- 예측력 있음 → P_crisis는 리스크 예측 변수 (justified)
- 예측력 없음 → 단순 리스크 축소 장치 (risk reduction device)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

# Config
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

    # Equal-weighted sector forward 20d return (log return)
    prices = raw[sector_cols]
    log_prices = np.log(prices)
    fwd_20d = log_prices.shift(-REBALANCE_DAYS) - log_prices
    ew_ret = fwd_20d.mean(axis=1)

    # Align: for each rebalance date, get forward return from that date
    forward_rets = []
    for d in pc_df["date"]:
        idx = raw.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]
        if 0 <= idx < len(ew_ret):
            val = ew_ret.iloc[idx]
            forward_rets.append(val if not np.isnan(val) else np.nan)
        else:
            forward_rets.append(np.nan)

    pc_df["forward_ret"] = forward_rets
    pc_df = pc_df.dropna(subset=["forward_ret"])

    if len(pc_df) < 10:
        print("[ERROR] Too few valid rows for analysis.")
        return

    X = pc_df["p_crisis"].values.reshape(-1, 1)
    y = pc_df["forward_ret"].values

    # --- 1. Regression: forward_ret ~ p_crisis ---
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    n = len(y)
    y_pred = reg.predict(X)
    resid = y - y_pred
    x_var = np.var(X.ravel())
    se_slope = np.sqrt(np.sum(resid**2) / max(1, n - 2)) / np.sqrt(max(1e-12, n * x_var))
    t_slope = slope / se_slope if se_slope > 1e-12 else 0
    pval_slope = 2 * (1 - stats.t.cdf(abs(t_slope), max(1, n - 2)))

    # --- 2. Spearman correlation ---
    rho, p_rho = stats.spearmanr(pc_df["p_crisis"], pc_df["forward_ret"])

    # --- 3. Median split ---
    med = pc_df["p_crisis"].median()
    high = pc_df[pc_df["p_crisis"] > med]["forward_ret"]
    low = pc_df[pc_df["p_crisis"] <= med]["forward_ret"]
    t_stat, p_ttest = stats.ttest_ind(high, low)

    # --- 4. Crisis threshold 0.2 ---
    crisis_thresh = 0.2
    crisis_periods = pc_df[pc_df["p_crisis"] > crisis_thresh]["forward_ret"]
    calm_periods = pc_df[pc_df["p_crisis"] <= crisis_thresh]["forward_ret"]
    t_crisis, p_crisis_ttest = stats.ttest_ind(crisis_periods, calm_periods)

    # --- Report ---
    print("\n" + "=" * 60)
    print("HMM P_Crisis Predictive Power Validation")
    print("=" * 60)
    print("\nForward return: Equal-weighted 9-sector, 20-day (monthly)")
    print(f"Sample: {n} rebalance periods")
    print(f"P_crisis range: [{pc_df['p_crisis'].min():.4f}, {pc_df['p_crisis'].max():.4f}]")
    print(f"P_crisis median: {med:.4f}")
    print(f"Forward return mean: {y.mean()*100:.2f}%, std: {y.std()*100:.2f}%")

    print("\n--- 1. Regression: Forward_Ret ~ P_Crisis ---")
    print(f"  Slope (β): {slope:.4f}  (negative = predictive)")
    print(f"  t-stat: {t_slope:.3f}, p-value: {pval_slope:.4f}")
    print("  → Significant at 5%" if pval_slope < 0.05 else "  → NOT significant at 5%")

    print("\n--- 2. Spearman Correlation ---")
    print(f"  ρ = {rho:.4f}, p-value: {p_rho:.4f}")
    print("  → Significant" if p_rho < 0.05 else "  → NOT significant")

    print("\n--- 3. Median Split (High vs Low P_crisis) ---")
    print(f"  High P_crisis (>{med:.3f}): mean = {high.mean()*100:.2f}%, n={len(high)}")
    print(f"  Low  P_crisis (≤{med:.3f}): mean = {low.mean()*100:.2f}%, n={len(low)}")
    print(f"  Diff: {(high.mean() - low.mean())*100:.2f}%, p={p_ttest:.4f}")

    print("\n--- 4. Crisis Regime (P_crisis > 0.2) vs Calm ---")
    print(f"  Crisis: mean = {crisis_periods.mean()*100:.2f}%, n={len(crisis_periods)}")
    print(f"  Calm:   mean = {calm_periods.mean()*100:.2f}%, n={len(calm_periods)}")
    print(f"  Diff: {(crisis_periods.mean() - calm_periods.mean())*100:.2f}%, p={p_crisis_ttest:.4f}")

    print("\n" + "=" * 60)
    verdict = (
        "P_crisis shows PREDICTIVE power: crisis regime → lower next-month returns."
        if (pval_slope < 0.05 or p_rho < 0.05) and slope < 0
        else "P_crisis does NOT show significant predictive power. It acts as a risk reduction device."
    )
    print("VERDICT:", verdict)
    print("=" * 60 + "\n")

    out_path = EXP_OUT / "hmm_predictive_validation.csv"
    pc_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
