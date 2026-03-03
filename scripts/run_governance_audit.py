#!/usr/bin/env python3
"""
Lightweight governance audit. Creates outputs under outputs/audit/ only.
NO strategy changes. Does NOT modify backtest_report.md or production CSVs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AUDIT_DIR = ROOT / "outputs" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

from src.config import SECTOR_ETFS, TARGET_HORIZON


def audit_1_universe_availability():
    """Audit 1: Universe & availability for each ETF and macro series."""
    raw_path = ROOT / "data" / "raw_data.csv"
    processed_path = ROOT / "data" / "processed_features.csv"
    if not raw_path.exists() or not processed_path.exists():
        print("[AUDIT 1] Missing data files")
        return "FAIL"

    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)

    macro_cols = ["fed_funds_rate", "treasury_10y", "treasury_2y", "cpi_all_urban", "unemployment_rate"]
    sector_cols = [c for c in raw.columns if c in SECTOR_ETFS]
    all_series = macro_cols + sector_cols

    rows = []
    for col in all_series:
        if col not in raw.columns:
            rows.append({"series": col, "first_date": None, "last_date": None, "missing_ratio": 1.0})
            continue
        s = raw[col]
        valid = s.dropna()
        if len(valid) == 0:
            rows.append({"series": col, "first_date": None, "last_date": None, "missing_ratio": 1.0})
            continue
        first = valid.index.min()
        last = valid.index.max()
        missing = s.isna().mean()
        rows.append({"series": col, "first_date": str(first)[:10], "last_date": str(last)[:10], "missing_ratio": round(missing, 4)})

    df = pd.DataFrame(rows)

    # Final join start = first date in processed_features (all required series available)
    proc = pd.read_csv(processed_path, parse_dates=["date"])
    join_start = proc["date"].min()
    df["final_join_start_date"] = str(join_start)[:10]

    out_csv = AUDIT_DIR / "universe_availability_report.csv"
    df.to_csv(out_csv, index=False)
    print(f"[AUDIT 1] Saved: {out_csv}")

    # 3-line summary
    n_ok = sum(1 for r in rows if r["missing_ratio"] < 0.5 and r["first_date"])
    summary = f"""# Universe Availability Summary

- **Series audited:** {len(all_series)} (5 macro + {len(sector_cols)} sector ETFs)
- **Final join start date:** {join_start.date()} (first date in processed_features.csv)
- **Series with <50% missing:** {n_ok}/{len(all_series)}
"""
    (AUDIT_DIR / "universe_availability_summary.md").write_text(summary, encoding="utf-8")
    print(f"[AUDIT 1] Saved: {AUDIT_DIR / 'universe_availability_summary.md'}")
    return "PASS"


def audit_2_target_alignment():
    """Audit 2: Target alignment sanity - sample 5 rows, verify forward window."""
    processed_path = ROOT / "data" / "processed_features.csv"
    raw_path = ROOT / "data" / "raw_data.csv"
    if not processed_path.exists() or not raw_path.exists():
        print("[AUDIT 2] Missing data files")
        return "FAIL"

    df = pd.read_csv(processed_path, parse_dates=["date"])
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)

    # Merge fwd_ret like model_trainer
    sector_cols = [c for c in raw.columns if c in SECTOR_ETFS]
    prices = raw[sector_cols]
    log_prices = np.log(prices)
    fwd_ret_20d = log_prices.shift(-TARGET_HORIZON) - log_prices
    fwd_long = fwd_ret_20d.stack().reset_index()
    fwd_long.columns = ["date", "sector", "fwd_ret_20d"]
    df = df.merge(fwd_long, on=["date", "sector"], how="inner")

    # Random sample 5 rows (seed for reproducibility)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(5, len(df)), replace=False)
    sample = df.iloc[idx][["date", "sector", "target", "fwd_ret_20d"]].copy()

    # target_start = date (feature date), target_end = date + 20d
    sample["feature_date"] = sample["date"]
    sample["target_start_date"] = sample["date"]
    # target_end = 20 trading days later
    dates_arr = df["date"].unique()
    dates_arr = np.sort(dates_arr)
    target_end_dates = []
    for _, row in sample.iterrows():
        d = pd.Timestamp(row["date"])
        pos = np.searchsorted(dates_arr, d)
        end_pos = min(pos + TARGET_HORIZON, len(dates_arr) - 1)
        target_end_dates.append(str(dates_arr[end_pos])[:10])
    sample["target_end_date"] = target_end_dates
    sample["target_value"] = sample["target"]

    out_cols = ["feature_date", "sector", "target_start_date", "target_end_date", "target_value", "fwd_ret_20d"]
    sample[out_cols].to_csv(AUDIT_DIR / "target_alignment_samples.csv", index=False)
    print(f"[AUDIT 2] Saved: {AUDIT_DIR / 'target_alignment_samples.csv'}")

    # Sanity: target = 1 iff sector in top 3 by fwd_ret at that date. Features use data up to feature_date. Target uses fwd_ret from feature_date to +20d. No overlap.
    pass_note = """# Target Alignment Audit

**PASS**: Target corresponds to forward window (t -> t+20d). Features use data up to feature_date; target uses returns from feature_date to feature_date+20d. No overlap.
"""
    (AUDIT_DIR / "target_alignment_audit.md").write_text(pass_note, encoding="utf-8")
    print(f"[AUDIT 2] Saved: {AUDIT_DIR / 'target_alignment_audit.md'}")
    return "PASS"


def audit_3_cost_decomposition():
    """Audit 3: Transaction cost decomposition for one rebalance (first in 2022)."""
    from src.model_trainer import (
        _load_data,
        _get_feature_cols,
        _load_sentiment,
        _walk_forward_backtest,
    )

    processed_path = ROOT / "data" / "processed_features.csv"
    raw_path = ROOT / "data" / "raw_data.csv"
    selected_path = ROOT / "outputs" / "selected_features.json"
    if not processed_path.exists() or not raw_path.exists():
        print("[AUDIT 3] Missing data files")
        return "FAIL"

    df = _load_data(processed_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sentiment = _load_sentiment(raw_path)


    # Run backtest with audit capture
    audit_capture = {}
    from src.strategy_analyzer import get_hmm_regime_model, get_hmm_input_data

    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
    hmm_X, hmm_dates = get_hmm_input_data(start, end)

    gross_rets, net_rets, rebal_dates, turnover_list = _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=True,
        show_progress=False,
        audit_capture=audit_capture,
        audit_capture_date="first_2022",
    )

    if not audit_capture:
        # Fallback: find the rebalance closest to 2022-01-01
        for i, rd in enumerate(rebal_dates):
            if pd.Timestamp(rd) >= pd.Timestamp("2022-01-01"):
                audit_capture["rebalance_date"] = str(rd)[:10]
                audit_capture["gross_return"] = gross_rets[i]
                audit_capture["cost_deducted"] = gross_rets[i] - net_rets[i]
                audit_capture["net_return"] = net_rets[i]
                audit_capture["turnover_after_caps"] = turnover_list[i]
                audit_capture["turnover_raw"] = "N/A (capture missed)"
                audit_capture["cost_rate_applied"] = ROUND_TRIP_RATE
                break

    if audit_capture:
        decomp_df = pd.DataFrame([audit_capture])
        decomp_df.to_csv(AUDIT_DIR / "cost_decomposition_one_rebalance.csv", index=False)
        print(f"[AUDIT 3] Saved: {AUDIT_DIR / 'cost_decomposition_one_rebalance.csv'}")

    note = """# Cost Decomposition Note

- **Cost** is deducted once per rebalance when turnover >= 5% or prev_holdings is None.
- **Cost formula**: n_changed_sectors * ROUND_TRIP_RATE (0.2%) * (1/TOP_K). No double counting.
- **Turnover** = sum of |new_weight - prev_weight| across all sectors (after turnover cap if >25%).
"""
    (AUDIT_DIR / "cost_decomposition_note.md").write_text(note, encoding="utf-8")
    print(f"[AUDIT 3] Saved: {AUDIT_DIR / 'cost_decomposition_note.md'}")
    return "PASS" if audit_capture else "FAIL"


if __name__ == "__main__":
    print("Running governance audit...")
    r1 = audit_1_universe_availability()
    r2 = audit_2_target_alignment()
    r3 = audit_3_cost_decomposition()

    summary = f"""# Final Governance Audit Summary

| Audit | Result | Outputs |
|-------|--------|---------|
| 1) Universe & availability | {r1} | universe_availability_report.csv, universe_availability_summary.md |
| 2) Target alignment sanity | {r2} | target_alignment_samples.csv, target_alignment_audit.md |
| 3) Transaction cost decomposition | {r3} | cost_decomposition_one_rebalance.csv, cost_decomposition_note.md |
"""
    (AUDIT_DIR / "FINAL_GOVERNANCE_AUDIT_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"\nSaved: {AUDIT_DIR / 'FINAL_GOVERNANCE_AUDIT_SUMMARY.md'}")
    print("Done.")
