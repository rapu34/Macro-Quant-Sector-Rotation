"""
HMM P_Crisis Drawdown Predictive Power Validation

Test whether high P_crisis increases probability of future drawdowns.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

REBALANCE_DAYS = 20
DRAWDOWN_THRESHOLD = -0.05  # -5%
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU"]


def _forward_mdd(daily_rets: np.ndarray) -> float:
    """Compute max drawdown from daily log returns. Returns negative value (e.g. -0.05 = -5%)."""
    if len(daily_rets) < 2:
        return 0.0
    wealth = np.exp(np.cumsum(daily_rets))
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return float(np.min(dd))


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

    # Forward 20-day max drawdown for each rebalance date
    forward_mdds = []
    for d in pc_df["date"]:
        idx = dates_arr.get_indexer([pd.Timestamp(d)], method="nearest")[0]
        if idx < 0 or idx + REBALANCE_DAYS >= len(ew_daily):
            forward_mdds.append(np.nan)
            continue
        window = ew_daily.iloc[idx + 1 : idx + REBALANCE_DAYS + 1].values
        if len(window) < REBALANCE_DAYS:
            forward_mdds.append(np.nan)
            continue
        mdd = _forward_mdd(window)
        forward_mdds.append(mdd)

    pc_df["forward_20d_mdd"] = forward_mdds
    pc_df["drawdown_event"] = (pc_df["forward_20d_mdd"] < DRAWDOWN_THRESHOLD).astype(int)
    pc_df = pc_df.dropna(subset=["forward_20d_mdd"])

    if len(pc_df) < 10:
        print("[ERROR] Too few valid rows for analysis.")
        return

    med = pc_df["p_crisis"].median()
    high_pc = pc_df[pc_df["p_crisis"] > med]
    low_pc = pc_df[pc_df["p_crisis"] <= med]

    # --- 3. Conditional probabilities ---
    p_dd_given_high = high_pc["drawdown_event"].mean()
    p_dd_given_low = low_pc["drawdown_event"].mean()
    n_high, n_low = len(high_pc), len(low_pc)
    n_dd_high = high_pc["drawdown_event"].sum()
    n_dd_low = low_pc["drawdown_event"].sum()

    # --- 4. Logistic regression: Drawdown_Event ~ P_crisis ---
    X = pc_df[["p_crisis"]].values
    y = pc_df["drawdown_event"].values
    logit = LogisticRegression(random_state=42, max_iter=500).fit(X, y)
    coef = logit.coef_[0, 0]
    odds_ratio = np.exp(coef)

    # Wald test for coefficient (p-value)
    from scipy.special import expit
    pred_prob = expit(X @ logit.coef_.T + logit.intercept_).ravel()
    W = pred_prob * (1 - pred_prob)
    X_const = np.column_stack([np.ones(len(X)), X[:, 0]])
    try:
        cov = np.linalg.inv(X_const.T @ (W.reshape(-1, 1) * X_const))
        se_coef = np.sqrt(cov[1, 1])
        z_stat = coef / se_coef if se_coef > 1e-12 else 0
        pval_logit = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    except np.linalg.LinAlgError:
        pval_logit = np.nan
        z_stat = np.nan

    # --- Report ---
    print("\n" + "=" * 60)
    print("HMM P_Crisis vs Forward 20-Day Drawdown Probability")
    print("=" * 60)
    print(f"\nDrawdown event: Forward_MDD < {DRAWDOWN_THRESHOLD*100:.0f}%")
    print(f"Sample: {len(pc_df)} rebalance periods")
    print(f"P_crisis median: {med:.4f}")
    print(f"Forward MDD range: [{pc_df['forward_20d_mdd'].min()*100:.2f}%, {pc_df['forward_20d_mdd'].max()*100:.2f}%]")
    print(f"Total drawdown events: {pc_df['drawdown_event'].sum()}")

    print("\n--- 3. Conditional Probabilities ---")
    print(f"  P(Drawdown | P_crisis > median): {p_dd_given_high:.2%}  (n={n_high}, events={n_dd_high})")
    print(f"  P(Drawdown | P_crisis ≤ median): {p_dd_given_low:.2%}  (n={n_low}, events={n_dd_low})")
    print(f"  Difference: {(p_dd_given_high - p_dd_given_low):.2%}")

    print("\n--- 4. Logistic Regression: Drawdown_Event ~ P_crisis ---")
    print(f"  Coefficient (β): {coef:.4f}")
    print(f"  Odds ratio (exp(β)): {odds_ratio:.4f}")
    print(f"  p-value: {pval_logit:.4f}")
    print("  → Significant at 5%" if pval_logit < 0.05 else "  → NOT significant at 5%")

    print("\n" + "=" * 60)
    verdict = (
        "High P_crisis increases probability of future drawdowns."
        if pval_logit < 0.05 and coef > 0
        else "P_crisis does NOT significantly predict drawdown probability."
    )
    print("VERDICT:", verdict)
    print("=" * 60 + "\n")

    out_path = EXP_OUT / "hmm_drawdown_validation.csv"
    pc_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
