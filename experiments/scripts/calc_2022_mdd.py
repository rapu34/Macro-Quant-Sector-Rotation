#!/usr/bin/env python3
"""
2022-only Max Drawdown calculation.
Uses same backtest logic (HMM expanding, risk_mult, costs). Sandbox only.
Output: outputs/2022_subperiod_performance.csv, outputs/2022_subperiod_summary.csv

Delete outputs: python experiments/scripts/calc_2022_mdd.py --delete
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

REBALANCE_DAYS = 20
Y2022_START = "2022-01-01"
Y2022_END = "2022-12-31"


def _delete_outputs() -> bool:
    """Remove 2022 subperiod outputs. Returns True if any file was deleted."""
    out_dir = ROOT / "outputs"
    files = ["2022_subperiod_performance.csv", "2022_subperiod_summary.csv"]
    deleted = False
    for f in files:
        p = out_dir / f
        if p.exists():
            p.unlink()
            print(f"Deleted: {p}")
            deleted = True
    if not deleted:
        print("No 2022 subperiod files to delete.")
    return deleted


def _cagr(returns: list[float]) -> float:
    arr = np.array(returns)
    if len(arr) < 1:
        return 0.0
    n_per = len(arr)
    periods_per_year = 252 / REBALANCE_DAYS
    total = np.prod(1 + arr)
    return float(total ** (periods_per_year / n_per) - 1) * 100 if n_per > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Delete 2022 subperiod outputs")
    args = parser.parse_args()
    if args.delete:
        _delete_outputs()
        return

    from src.model_trainer import (
        _load_data,
        _get_feature_cols,
        _load_sentiment,
        _walk_forward_backtest,
        _metrics,
    )
    from src.strategy_analyzer import get_hmm_regime_model, get_hmm_input_data

    processed_path = ROOT / "data" / "processed_features.csv"
    raw_path = ROOT / "data" / "raw_data.csv"
    selected_path = ROOT / "outputs" / "selected_features.json"

    df = _load_data(processed_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sentiment = _load_sentiment(raw_path)

    # Run backtest for 2022 period only (same pattern as robustness_oos_evaluation)
    mask = (df["date"] >= Y2022_START) & (df["date"] <= Y2022_END)
    df_2022_sub = df[mask].copy()
    if len(df_2022_sub) < 20:
        print("[ERROR] Insufficient 2022 data. Need at least 20 rows.")
        return

    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
    hmm_X, hmm_dates = get_hmm_input_data(start, end)

    p_crisis_log = []
    gross_rets, net_rets, rebal_dates, _ = _walk_forward_backtest(
        df_2022_sub, feature_cols, scaler,
        sentiment_series=sentiment,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=True,
        p_crisis_log=p_crisis_log,
        show_progress=False,
    )

    # All rebalance dates are in 2022 (we ran on 2022 subset)
    t_start = pd.Timestamp(Y2022_START)
    t_end = pd.Timestamp(Y2022_END)
    rows_2022 = []
    for i, rd in enumerate(rebal_dates):
        rts = pd.Timestamp(rd)
        if t_start <= rts <= t_end:
            rows_2022.append({
                "date": str(rd)[:10],
                "period_net_return": net_rets[i],
                "period_gross_return": gross_rets[i],
                "p_crisis": p_crisis_log[i]["p_crisis"] if i < len(p_crisis_log) else np.nan,
                "risk_mult": p_crisis_log[i]["risk_mult"] if i < len(p_crisis_log) else np.nan,
            })

    # Verify no dates outside 2022
    for rd in rebal_dates:
        rts = pd.Timestamp(rd)
        if rts < t_start or rts > t_end:
            print(f"[WARN] Rebalance date outside 2022: {rd}")

    if not rows_2022:
        print("[ERROR] No rebalance dates in 2022. Check data range.")
        return

    df_2022 = pd.DataFrame(rows_2022)
    net_2022 = df_2022["period_net_return"].values

    # Cumulative return, peak, drawdown (period-level)
    wealth = np.ones(len(net_2022) + 1)
    for i, r in enumerate(net_2022):
        wealth[i + 1] = max(0.0, wealth[i] * (1.0 + r))
    wealth = wealth[1:]
    peak = np.maximum.accumulate(wealth)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 1e-12, (wealth - peak) / peak, 0.0)

    df_2022["cumulative_return"] = wealth - 1.0
    df_2022["drawdown"] = drawdown

    mdd_2022 = float(np.min(drawdown)) * 100
    cagr_2022 = _cagr(net_2022.tolist())
    m = _metrics(net_2022.tolist())
    sharpe_2022 = m["sharpe"]
    crisis_on = 100 * sum(1 for p in p_crisis_log if pd.Timestamp(p["date"]) >= t_start and pd.Timestamp(p["date"]) <= t_end and p["p_crisis"] >= 0.8 - 1e-6) / max(1, len(rows_2022))

    # Save detailed CSV (period-level; cost included in period_net_return)
    out_path = ROOT / "outputs" / "2022_subperiod_performance.csv"
    df_out = df_2022[["date", "period_net_return", "cumulative_return", "drawdown", "p_crisis", "risk_mult"]].copy()
    df_out = df_out.rename(columns={"period_net_return": "daily_net_return"})  # user-requested column name
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("  Note: daily_net_return = period net return (20-day rebalance); cost included.")

    # Summary
    summary = {
        "metric": [
            "2022_only_MDD",
            "2022_CAGR",
            "2022_Sharpe",
            "2022_Crisis_ON_ratio",
            "SP500_2022_return_ref",
        ],
        "value": [
            f"{mdd_2022:.2f}%",
            f"{cagr_2022:.2f}%",
            f"{sharpe_2022:.2f}",
            f"{crisis_on:.1f}%",
            "-19% (reference)",
        ],
    }
    summary_path = ROOT / "outputs" / "2022_subperiod_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Report
    print("\n" + "=" * 60)
    print("2022 SUBPERIOD RESULTS (Net, cost included)")
    print("=" * 60)
    print(f"  2022_only_MDD:        {mdd_2022:.2f}%")
    print(f"  2022_CAGR:            {cagr_2022:.2f}%")
    print(f"  2022_Sharpe:          {sharpe_2022:.2f}")
    print(f"  2022_Crisis_ON_ratio: {crisis_on:.1f}%")
    print(f"  Rebalances in 2022:   {len(rows_2022)}")
    print("=" * 60)
    print(f'\nReport: "In 2022 alone, the system experienced a maximum drawdown of {mdd_2022:.1f}%, while S&P 500 declined approximately -19%."')
    print("=" * 60)


if __name__ == "__main__":
    main()
