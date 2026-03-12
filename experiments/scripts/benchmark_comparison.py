#!/usr/bin/env python3
"""
Benchmark Comparison — Portfolio vs SPY. Produces benchmark_daily.csv, benchmark_metrics.json.
Run: python experiments/scripts/benchmark_comparison.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

_suffix = "_refresh" if os.environ.get("PIPELINE_REFRESH_MODE") == "1" else ""
EXP_OUT = Path(__file__).resolve().parent.parent / f"outputs{_suffix}"
OUT = ROOT / f"outputs{_suffix}"
W_B1, W_B2 = 0.3, 0.7
PPY = 252


def _load_from_factor_data() -> pd.DataFrame | None:
    """Load r_port, r_spy from factor_regression output (avoids yfinance)."""
    p = EXP_OUT / "benchmark_factor_data.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if "r_p" in df.columns and "r_spy" in df.columns:
            return df[["r_p", "r_spy"]].dropna(how="any")
    except Exception:
        pass
    return None


def main():
    print("\n=== Benchmark Comparison ===\n")
    if not (EXP_OUT / "true_daily_block1.csv").exists():
        print("[WARN] true_daily_block1.csv not found. Run pipeline first.")
        return 1
    b1 = pd.read_csv(EXP_OUT / "true_daily_block1.csv", parse_dates=["date"]).set_index("date")["block1"].squeeze()
    b2_path = EXP_OUT / "block2_hmm_expanding_rebalonly.csv"
    b2 = pd.read_csv(b2_path if b2_path.exists() else EXP_OUT / "true_daily_block2.csv", parse_dates=["date"]).set_index("date").iloc[:, 0].squeeze()
    common = b1.index.intersection(b2.index)
    r_p = W_B1 * b1.reindex(common).ffill().bfill().fillna(0) + W_B2 * b2.reindex(common).ffill().bfill().fillna(0)
    r_p = r_p.dropna()
    # Prefer factor_regression output (no yfinance)
    merged = _load_from_factor_data()
    if merged is None or len(merged) < 10:
        print("[WARN] No benchmark data. Run factor_regression first (or ensure benchmark_factor_data.csv exists).")
        return 0
    cum_p = (1 + merged["r_p"]).cumprod()
    cum_s = (1 + merged["r_spy"]).cumprod()
    excess = merged["r_p"] - merged["r_spy"]
    n = len(merged)
    ann_ret_p = float((cum_p.iloc[-1] / cum_p.iloc[0]) ** (PPY / n) - 1) if n > 0 else 0
    ann_ret_s = float((cum_s.iloc[-1] / cum_s.iloc[0]) ** (PPY / n) - 1) if n > 0 else 0
    excess_ann = ann_ret_p - ann_ret_s
    te = float(excess.std() * np.sqrt(PPY)) if excess.std() > 1e-12 else 0
    ir = float(excess_ann / te) if te > 1e-6 else 0
    downside = merged["r_p"][merged["r_p"] < 0]
    dd_vol = float(downside.std() * np.sqrt(PPY)) if len(downside) > 1 and downside.std() > 1e-12 else 0
    sortino = float(ann_ret_p / dd_vol) if dd_vol > 1e-6 else 0
    pd.DataFrame({"date": merged.index, "r_port": merged["r_p"].values, "r_spy": merged["r_spy"].values, "cum_port": cum_p.values, "cum_spy": cum_s.values}).to_csv(EXP_OUT / "benchmark_daily.csv", index=False)
    with open(EXP_OUT / "benchmark_metrics.json", "w") as f:
        json.dump({"excess_return_ann": excess_ann, "tracking_error": te, "information_ratio": ir, "sortino_ratio": sortino, "downside_deviation": dd_vol, "ann_ret_portfolio": ann_ret_p, "ann_ret_spy": ann_ret_s}, f, indent=2)
    print("Saved: benchmark_daily.csv, benchmark_metrics.json")
    # Regime performance
    regime_path = OUT / "hmm_regime.csv"
    if regime_path.exists():
        try:
            regime_df = pd.read_csv(regime_path, parse_dates=["date"]).set_index("date")
            if "P_Crisis" in regime_df.columns:
                regime_df["regime"] = np.where(regime_df["P_Crisis"] >= 0.6, "Crisis", np.where(regime_df["P_Crisis"] >= 0.4, "Elevated", "Core"))
                joined = merged[["r_p"]].join(regime_df[["regime"]], how="inner").dropna()
                rows = []
                for reg in ["Core", "Elevated", "Crisis"]:
                    sub = joined[joined["regime"] == reg]["r_p"]
                    if len(sub) > 20:
                        ann_r = float((1 + sub).prod() ** (PPY / len(sub)) - 1)
                        vol = float(sub.std() * np.sqrt(PPY))
                        sharpe = float(ann_r / vol) if vol > 1e-8 else 0
                        rows.append({"Regime": reg, "Avg Return": ann_r, "Volatility": vol, "Sharpe": sharpe})
                if rows:
                    pd.DataFrame(rows).to_csv(EXP_OUT / "regime_performance.csv", index=False)
                    print("Saved: regime_performance.csv")
        except Exception as e:
            print(f"[WARN] Regime: {e}")
    # Alert history
    worst_path = EXP_OUT / "stress_top5_worst_days.csv"
    alert_rows = [{"Date": d, "Alert": lab} for d, lab in [("2020-03-09", "SPY crash (COVID)"), ("2020-03-12", "SPY crash (COVID)"), ("2020-03-16", "SPY crash (COVID)"), ("2022-06-13", "VIX spike (Rate shock)"), ("2025-04-04", "Volatility shock (Tariff)")] if pd.Timestamp(d) in merged.index]
    if worst_path.exists():
        try:
            w5 = pd.read_csv(worst_path, index_col=0)
            for idx in w5.index[:5]:
                d = str(idx)[:10]
                if not any(r["Date"] == d for r in alert_rows):
                    alert_rows.append({"Date": d, "Alert": "VIX spike" if w5.loc[idx, "delta_vix"] * 100 > 30 else "SPY crash"})
        except Exception:
            pass
    if alert_rows:
        pd.DataFrame(alert_rows).sort_values("Date", ascending=False).drop_duplicates("Date").head(20).to_csv(EXP_OUT / "alert_history.csv", index=False)
        print("Saved: alert_history.csv")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
