#!/usr/bin/env python3
"""
CVaR Contribution Decomposition for final portfolio (30% Block1 + 70% Block2_HMM_REBAL_ONLY).

Run: python experiments/scripts/cvar_decomposition.py

True Daily PnL only. No smoothing. No additional scaling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"

W_B1 = 0.3
W_B2 = 0.7


def main():
    print("\n=== CVaR Contribution Decomposition ===\n")

    block1 = pd.read_csv(EXP_OUT / "true_daily_block1.csv", index_col=0, parse_dates=True).squeeze()
    block2 = pd.read_csv(EXP_OUT / "block2_hmm_expanding_rebalonly.csv", index_col=0, parse_dates=True)
    block2 = block2.iloc[:, 0].squeeze()

    common = block1.index.intersection(block2.index)
    r_b1 = block1.reindex(common).ffill().bfill().fillna(0)
    r_b2 = block2.reindex(common).ffill().bfill().fillna(0)
    r_p = W_B1 * r_b1 + W_B2 * r_b2

    # 1) Portfolio CVaR(95%)
    q05 = r_p.quantile(0.05)
    q01 = r_p.quantile(0.01)
    cvar_95 = r_p[r_p <= q05].mean()
    cvar_99 = r_p[r_p <= q01].mean()

    # 2) Tail sets
    tail_5pct = r_p <= q05
    tail_1pct = r_p <= q01
    n_tail_5 = tail_5pct.sum()
    n_tail_1 = tail_1pct.sum()

    # 3) Component contribution: contrib_i = mean(w_i * r_i | tail)
    contrib_b1_5 = (W_B1 * r_b1).loc[tail_5pct].mean()
    contrib_b2_5 = (W_B2 * r_b2).loc[tail_5pct].mean()
    contrib_b1_1 = (W_B1 * r_b1).loc[tail_1pct].mean()
    contrib_b2_1 = (W_B2 * r_b2).loc[tail_1pct].mean()

    # 4) CVaR contribution %
    cvar_contrib_b1_5_pct = (contrib_b1_5 / cvar_95 * 100) if abs(cvar_95) > 1e-12 else np.nan
    cvar_contrib_b2_5_pct = (contrib_b2_5 / cvar_95 * 100) if abs(cvar_95) > 1e-12 else np.nan
    cvar_contrib_b1_1_pct = (contrib_b1_1 / cvar_99 * 100) if abs(cvar_99) > 1e-12 else np.nan
    cvar_contrib_b2_1_pct = (contrib_b2_1 / cvar_99 * 100) if abs(cvar_99) > 1e-12 else np.nan

    # Tail correlation (Block1 vs Block2 in worst 5%)
    tail_corr_5 = r_b1.loc[tail_5pct].corr(r_b2.loc[tail_5pct]) if n_tail_5 >= 2 else np.nan
    tail_corr_1 = r_b1.loc[tail_1pct].corr(r_b2.loc[tail_1pct]) if n_tail_1 >= 2 else np.nan

    # Which block drives extreme loss?
    if abs(contrib_b1_5) >= abs(contrib_b2_5):
        driver_5 = "Block1"
    else:
        driver_5 = "Block2"
    if abs(contrib_b1_1) >= abs(contrib_b2_1):
        driver_1 = "Block1"
    else:
        driver_1 = "Block2"

    # Decomposition table
    decomp_rows = [
        {
            "tail_percentile": "5%",
            "n_days": n_tail_5,
            "portfolio_cvar": cvar_95,
            "contrib_block1": contrib_b1_5,
            "contrib_block2": contrib_b2_5,
            "cvar_contrib_b1_pct": cvar_contrib_b1_5_pct,
            "cvar_contrib_b2_pct": cvar_contrib_b2_5_pct,
            "tail_corr": tail_corr_5,
            "driver": driver_5,
        },
        {
            "tail_percentile": "1%",
            "n_days": n_tail_1,
            "portfolio_cvar": cvar_99,
            "contrib_block1": contrib_b1_1,
            "contrib_block2": contrib_b2_1,
            "cvar_contrib_b1_pct": cvar_contrib_b1_1_pct,
            "cvar_contrib_b2_pct": cvar_contrib_b2_1_pct,
            "tail_corr": tail_corr_1,
            "driver": driver_1,
        },
    ]
    df_decomp = pd.DataFrame(decomp_rows)

    # Tail details: dates and returns in worst 5%
    tail_dates_5 = r_p.loc[tail_5pct].sort_values().index
    tail_details = pd.DataFrame({
        "date": tail_dates_5,
        "r_portfolio": r_p.loc[tail_dates_5].values,
        "r_block1": r_b1.loc[tail_dates_5].values,
        "r_block2": r_b2.loc[tail_dates_5].values,
        "contrib_block1": (W_B1 * r_b1).loc[tail_dates_5].values,
        "contrib_block2": (W_B2 * r_b2).loc[tail_dates_5].values,
    })

    # Save
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    df_decomp.to_csv(EXP_OUT / "cvar_decomposition.csv", index=False)
    tail_details.to_csv(EXP_OUT / "cvar_tail_details.csv", index=False)

    # Report
    report = f"""# CVaR Contribution Decomposition Report

Run: `python experiments/scripts/cvar_decomposition.py`

Portfolio: 30% Block1 + 70% Block2_HMM_REBAL_ONLY. True Daily PnL. No smoothing.

## Portfolio CVaR

| Percentile | Portfolio CVaR | N (tail days) |
|------------|---------------|---------------|
| 95% (worst 5%) | {cvar_95:.4f} ({cvar_95*100:.2f}%) | {n_tail_5} |
| 99% (worst 1%) | {cvar_99:.4f} ({cvar_99*100:.2f}%) | {n_tail_1} |

## Block-wise CVaR Contribution (%)

| Tail | Block1 Contrib (%) | Block2 Contrib (%) | Sum |
|------|--------------------|--------------------|-----|
| Worst 5% | {cvar_contrib_b1_5_pct:.1f}% | {cvar_contrib_b2_5_pct:.1f}% | {cvar_contrib_b1_5_pct + cvar_contrib_b2_5_pct:.1f}% |
| Worst 1% | {cvar_contrib_b1_1_pct:.1f}% | {cvar_contrib_b2_1_pct:.1f}% | {cvar_contrib_b1_1_pct + cvar_contrib_b2_1_pct:.1f}% |

*Contrib_i = mean(w_i × r_i | r_p in tail). Contribution % = Contrib_i / CVaR_p × 100.*

## Tail Period Correlation (Block1 vs Block2)

| Tail | Correlation |
|------|-------------|
| Worst 5% | {tail_corr_5:.4f} |
| Worst 1% | {tail_corr_1:.4f} |

## Which Block Drives Extreme Loss?

| Tail | Primary Driver |
|------|-----------------|
| Worst 5% | **{driver_5}** (larger absolute contribution to loss) |
| Worst 1% | **{driver_1}** |

Interpretation: On the worst 5% of days, {driver_5} contributes more to portfolio loss. On the worst 1% of days, {driver_1} is the primary driver.

---
*Generated by cvar_decomposition.py*
"""

    with open(EXP_OUT / "cvar_report.md", "w") as f:
        f.write(report)

    print("CVaR Decomposition:")
    print(df_decomp.to_string(index=False))
    print(f"\nTail correlation (worst 5%): {tail_corr_5:.4f}")
    print(f"Primary driver (worst 5%): {driver_5}")
    print(f"Primary driver (worst 1%): {driver_1}")
    print(f"\nSaved: {EXP_OUT / 'cvar_decomposition.csv'}")
    print(f"Saved: {EXP_OUT / 'cvar_tail_details.csv'}")
    print(f"Saved: {EXP_OUT / 'cvar_report.md'}")


if __name__ == "__main__":
    main()
