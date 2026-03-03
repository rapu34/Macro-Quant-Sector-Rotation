"""
HMM P_Crisis Left-Tail Risk Validation

Test whether crisis regime increases left-tail return frequency.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

REBALANCE_DAYS = 20
TAIL_PERCENTILE = 10  # Bottom 10%
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

    # --- 1. Identify bottom 10% monthly returns ---
    p10 = np.percentile(pc_df["forward_ret"], TAIL_PERCENTILE)
    pc_df["bottom_10_event"] = (pc_df["forward_ret"] <= p10).astype(int)

    # --- 2. Split by median P_crisis ---
    med = pc_df["p_crisis"].median()
    pc_df["high_p_crisis"] = (pc_df["p_crisis"] > med).astype(int)

    high = pc_df[pc_df["high_p_crisis"] == 1]
    low = pc_df[pc_df["high_p_crisis"] == 0]

    freq_high = high["bottom_10_event"].mean()
    freq_low = low["bottom_10_event"].mean()
    n_bottom_high = high["bottom_10_event"].sum()
    n_bottom_low = low["bottom_10_event"].sum()

    # --- 3. Chi-square test (2x2 contingency table) ---
    # Rows: High P_crisis, Low P_crisis
    # Cols: Bottom 10%, Not Bottom 10%
    tbl = np.array([
        [n_bottom_high, len(high) - n_bottom_high],
        [n_bottom_low, len(low) - n_bottom_low],
    ])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(tbl)

    # --- Report ---
    print("\n" + "=" * 60)
    print("HMM P_Crisis vs Left-Tail Return Frequency")
    print("=" * 60)
    print(f"\nBottom {TAIL_PERCENTILE}%: forward_ret <= {p10*100:.2f}% (10th percentile)")
    print(f"Sample: {len(pc_df)} rebalance periods")
    print(f"P_crisis median: {med:.4f}")
    print(f"Total bottom-10% events: {pc_df['bottom_10_event'].sum()}")

    print("\n--- 2. Frequency of Bottom 10% by P_crisis State ---")
    print(f"  High P_crisis (>{med:.3f}): {freq_high:.2%}  (n={len(high)}, events={n_bottom_high})")
    print(f"  Low  P_crisis (≤{med:.3f}): {freq_low:.2%}  (n={len(low)}, events={n_bottom_low})")
    print(f"  Difference: {(freq_high - freq_low):.2%}")

    print("\n--- 3. Chi-Square Test (2x2 contingency) ---")
    print(f"  Contingency table:")
    print(f"                    Bottom 10%  Not Bottom 10%")
    print(f"  High P_crisis         {tbl[0,0]:2d}            {tbl[0,1]:2d}")
    print(f"  Low  P_crisis         {tbl[1,0]:2d}            {tbl[1,1]:2d}")
    print(f"  χ² = {chi2:.4f}, p-value = {p_chi2:.4f}")
    print("  → Significant at 5%" if p_chi2 < 0.05 else "  → NOT significant at 5%")

    print("\n" + "=" * 60)
    verdict = (
        "Crisis regime increases left-tail return frequency."
        if p_chi2 < 0.05 and freq_high > freq_low
        else "P_crisis does NOT significantly increase left-tail frequency."
    )
    print("VERDICT:", verdict)
    print("=" * 60 + "\n")

    out_path = EXP_OUT / "hmm_tailrisk_validation.csv"
    pc_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
