"""
HMM Regime-based CVaR Comparison

Compare CVaR(95%) between high and low P_crisis states.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

REBALANCE_DAYS = 20
CVAR_ALPHA = 0.95
N_BOOTSTRAP = 1000
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU"]


def _cvar95(returns: np.ndarray) -> float:
    """CVaR(95%) = mean of worst 5% of returns."""
    arr = np.asarray(returns)
    if len(arr) < 2:
        return 0.0
    n_tail = max(1, int(len(arr) * (1 - CVAR_ALPHA)))
    worst = np.partition(arr, n_tail - 1)[:n_tail]
    return float(np.mean(worst))


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

    # --- 1. Split by median P_crisis ---
    med = pc_df["p_crisis"].median()
    high_rets = pc_df[pc_df["p_crisis"] > med]["forward_ret"].values
    low_rets = pc_df[pc_df["p_crisis"] <= med]["forward_ret"].values

    # --- 2. Compute CVaR(95%) for each group ---
    cvar_high = _cvar95(high_rets)
    cvar_low = _cvar95(low_rets)

    # --- 3. Bootstrap 1000 times for confidence intervals ---
    np.random.seed(42)
    cvar_high_boot = []
    cvar_low_boot = []
    for _ in range(N_BOOTSTRAP):
        resamp_high = np.random.choice(high_rets, size=len(high_rets), replace=True)
        resamp_low = np.random.choice(low_rets, size=len(low_rets), replace=True)
        cvar_high_boot.append(_cvar95(resamp_high))
        cvar_low_boot.append(_cvar95(resamp_low))

    cvar_high_lo, cvar_high_hi = np.percentile(cvar_high_boot, [2.5, 97.5])
    cvar_low_lo, cvar_low_hi = np.percentile(cvar_low_boot, [2.5, 97.5])

    # --- Report ---
    print("\n" + "=" * 60)
    print("Regime-based CVaR(95%) Comparison")
    print("=" * 60)
    print(f"\nSplit: median P_crisis = {med:.4f}")
    print(f"Bootstrap: {N_BOOTSTRAP} iterations, 95% CI")

    print("\n--- CVaR(95%) by Regime ---")
    print(f"  High P_crisis (>{med:.3f}): {cvar_high*100:.2f}%  [{cvar_high_lo*100:.2f}%, {cvar_high_hi*100:.2f}%]  (n={len(high_rets)})")
    print(f"  Low  P_crisis (≤{med:.3f}): {cvar_low*100:.2f}%  [{cvar_low_lo*100:.2f}%, {cvar_low_hi*100:.2f}%]  (n={len(low_rets)})")
    print(f"  Difference (High - Low): {(cvar_high - cvar_low)*100:.2f}%")

    print("\n" + "=" * 60 + "\n")

    # --- 4. Save results ---
    out_rows = [
        {"group": "high_p_crisis", "n": len(high_rets), "cvar_95": cvar_high, "ci_lo": cvar_high_lo, "ci_hi": cvar_high_hi},
        {"group": "low_p_crisis", "n": len(low_rets), "cvar_95": cvar_low, "ci_lo": cvar_low_lo, "ci_hi": cvar_low_hi},
    ]
    out_df = pd.DataFrame(out_rows)
    out_path = EXP_OUT / "hmm_regime_cvar.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
