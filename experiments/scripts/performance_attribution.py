#!/usr/bin/env python3
"""
Performance attribution: Model vs EqualWeight.
Decomposes underperformance into gross alpha gap vs cost gap.
Outputs: performance_attribution_summary.csv, performance_attribution_timeseries.csv, performance_attribution.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

REBALANCE_DAYS = 20


def _metrics(returns: list[float]) -> dict:
    """Sharpe, MDD, CVaR (95%)."""
    from src.model_trainer import _metrics
    return _metrics(returns)


def _cagr(returns: list[float]) -> float:
    arr = np.array(returns)
    if len(arr) < 1:
        return 0.0
    n_per = len(arr)
    periods_per_year = 252 / REBALANCE_DAYS
    total = np.prod(1 + arr)
    return float(total ** (periods_per_year / n_per) - 1) * 100 if n_per > 0 else 0.0


def main():
    from src.model_trainer import (
        _load_data,
        _get_feature_cols,
        _load_sentiment,
        _walk_forward_backtest,
    )
    from src.strategy_analyzer import get_hmm_regime_model, get_hmm_input_data

    processed_path = ROOT / "data" / "processed_features.csv"
    raw_path = ROOT / "data" / "raw_data.csv"
    selected_path = ROOT / "outputs" / "selected_features.json"
    baseline_path = ROOT / "outputs" / "baseline_equal_weight.csv"

    if not baseline_path.exists():
        print("[ERROR] Run baseline_equal_weight.py first to create baseline_equal_weight.csv")
        return

    df = _load_data(processed_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sentiment = _load_sentiment(raw_path)
    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
    hmm_X, hmm_dates = get_hmm_input_data(start, end)

    p_crisis_log = []
    gross_model, net_model, dates_model, turnover_model = _walk_forward_backtest(
        df, feature_cols, scaler,
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
    cost_model = [g - n for g, n in zip(gross_model, net_model)]

    df_ew = pd.read_csv(baseline_path)


    df_model = pd.DataFrame({
        "date": [str(d)[:10] for d in dates_model],
        "gross_return": gross_model,
        "cost_deducted": cost_model,
        "net_return": net_model,
        "turnover_after_caps": turnover_model,
    })

    df_merged = df_model.merge(
        df_ew[["date", "gross_return", "cost_deducted", "net_return", "turnover_after_caps"]],
        on="date",
        how="inner",
        suffixes=("_model", "_ew"),
    )

    if len(df_merged) < 2:
        print("[ERROR] Insufficient overlapping dates for attribution.")
        return

    g_m = df_merged["gross_return_model"].values
    c_m = df_merged["cost_deducted_model"].values
    n_m = df_merged["net_return_model"].values
    c_ew = df_merged["cost_deducted_ew"].values
    n_ew = df_merged["net_return_ew"].values
    g_ew = df_merged["gross_return_ew"].values

    m_gross = _metrics(g_m.tolist())
    m_net_m = _metrics(n_m.tolist())
    m_net_ew = _metrics(n_ew.tolist())
    m_gross_ew = _metrics(g_ew.tolist())

    n_periods = len(df_merged)
    periods_per_year = 252 / REBALANCE_DAYS
    n_years = n_periods / periods_per_year

    avg_cost_model = np.mean(c_m)
    avg_cost_ew = np.mean(c_ew)
    total_cost_model = np.sum(c_m)
    total_cost_ew = np.sum(c_ew)
    cost_gap = total_cost_model - total_cost_ew
    annualized_cost_drag_model = avg_cost_model * periods_per_year * 100
    annualized_cost_drag_ew = avg_cost_ew * periods_per_year * 100

    cagr_gross_m = _cagr(g_m.tolist())
    cagr_gross_ew = _cagr(g_ew.tolist())
    cagr_net_m = _cagr(n_m.tolist())
    cagr_net_ew = _cagr(n_ew.tolist())

    net_cagr_gap = cagr_net_m - cagr_net_ew
    gross_cagr_gap = cagr_gross_m - cagr_gross_ew

    cost_gap_pct = cost_gap * 100
    if abs(cost_gap_pct) > abs(gross_cagr_gap):
        driver = "Cost-driven"
    else:
        driver = "Alpha-driven"

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "Model_Sharpe_gross": m_gross["sharpe"],
        "Model_Sharpe_net": m_net_m["sharpe"],
        "Model_CAGR_gross": cagr_gross_m,
        "Model_CAGR_net": cagr_net_m,
        "Model_MDD_gross": m_gross["mdd"] * 100,
        "Model_MDD_net": m_net_m["mdd"] * 100,
        "Model_CVaR95_gross": m_gross["cvar"] * 100,
        "Model_CVaR95_net": m_net_m["cvar"] * 100,
        "Model_AvgTurnover": np.mean(df_merged["turnover_after_caps_model"]) * 100,
        "Model_avg_cost_per_period": avg_cost_model,
        "Model_annualized_cost_drag_pct": annualized_cost_drag_model,
        "Model_total_cost": total_cost_model,
        "EW_Sharpe_gross": m_gross_ew["sharpe"],
        "EW_Sharpe_net": m_net_ew["sharpe"],
        "EW_CAGR_gross": cagr_gross_ew,
        "EW_CAGR_net": cagr_net_ew,
        "EW_MDD_gross": m_gross_ew["mdd"] * 100,
        "EW_MDD_net": m_net_ew["mdd"] * 100,
        "EW_CVaR95_gross": m_gross_ew["cvar"] * 100,
        "EW_CVaR95_net": m_net_ew["cvar"] * 100,
        "EW_AvgTurnover": np.mean(df_merged["turnover_after_caps_ew"]) * 100,
        "EW_avg_cost_per_period": avg_cost_ew,
        "EW_annualized_cost_drag_pct": annualized_cost_drag_ew,
        "EW_total_cost": total_cost_ew,
        "Net_CAGR_gap": net_cagr_gap,
        "Gross_CAGR_gap": gross_cagr_gap,
        "Cost_gap": cost_gap,
        "Cost_gap_pct": cost_gap_pct,
        "Underperformance_driver": driver,
        "n_periods": n_periods,
        "n_years": n_years,
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(out_dir / "performance_attribution_summary.csv", index=False)
    print(f"Saved: {out_dir / 'performance_attribution_summary.csv'}")

    df_ts = pd.DataFrame({
        "date": df_merged["date"],
        "model_gross": df_merged["gross_return_model"],
        "model_cost": df_merged["cost_deducted_model"],
        "model_net": df_merged["net_return_model"],
        "ew_gross": df_merged["gross_return_ew"],
        "ew_cost": df_merged["cost_deducted_ew"],
        "ew_net": df_merged["net_return_ew"],
        "model_turnover": df_merged["turnover_after_caps_model"],
        "ew_turnover": df_merged["turnover_after_caps_ew"],
    })
    df_ts.to_csv(out_dir / "performance_attribution_timeseries.csv", index=False)
    print(f"Saved: {out_dir / 'performance_attribution_timeseries.csv'}")

    md_lines = [
        "# Performance Attribution: Model vs EqualWeight",
        "",
        "## 핵심 결론 (5줄)",
        "",
        f"1. **Net CAGR gap**: Model {cagr_net_m:.2f}% vs EW {cagr_net_ew:.2f}% → **Δ = {net_cagr_gap:+.2f}pp**" + (" (Model underperforms)" if net_cagr_gap < 0 else ""),
        f"2. **Gross alpha gap**: {gross_cagr_gap:+.2f}pp (Model gross {cagr_gross_m:.2f}% vs EW {cagr_gross_ew:.2f}%) — alpha 차이 " + ("미미" if abs(gross_cagr_gap) < 0.5 else "유의"),
        f"3. **Cost gap**: Model 총 비용 {total_cost_model*100:.2f}% vs EW {total_cost_ew*100:.2f}% → **Δ = {cost_gap_pct:+.2f}pp**",
        f"4. **Annualized cost drag**: Model {annualized_cost_drag_model:.2f}% vs EW {annualized_cost_drag_ew:.2f}%",
        f"5. **Driver**: **{driver}** — 비용 격차({abs(cost_gap_pct):.2f}pp) vs alpha 격차({abs(gross_cagr_gap):.2f}pp)",
        "",
        "## Summary",
        "",
        f"- **Net CAGR gap**: Model {cagr_net_m:.2f}% vs EW {cagr_net_ew:.2f}% → **Δ = {net_cagr_gap:+.2f}pp**",
        f"- **Gross CAGR gap**: Model {cagr_gross_m:.2f}% vs EW {cagr_gross_ew:.2f}% → **Δ = {gross_cagr_gap:+.2f}pp** (alpha gap)",
        f"- **Cost gap**: Model total cost {total_cost_model*100:.2f}% vs EW {total_cost_ew*100:.2f}% → **Δ = {cost_gap_pct:+.2f}pp**",
        f"- **Cost drag (annualized)**: Model {annualized_cost_drag_model:.2f}% vs EW {annualized_cost_drag_ew:.2f}%",
        f"- **Driver**: **{driver}**",
        "",
        "## Conclusion",
        "",
    ]

    if net_cagr_gap < 0:
        if driver == "Cost-driven":
            conclusion = f"Model underperforms EW because **higher turnover (Model {summary['Model_AvgTurnover']:.1f}% vs EW {summary['EW_AvgTurnover']:.1f}%)** drives higher transaction costs ({cost_gap_pct:.2f}pp total gap), which more than offsets the gross alpha gap ({gross_cagr_gap:+.2f}pp)."
        else:
            conclusion = f"Model underperforms EW because **gross alpha is lower** ({gross_cagr_gap:+.2f}pp). Cost contributes {cost_gap_pct:+.2f}pp but is secondary to the alpha gap."
    else:
        conclusion = f"Model outperforms EW by {net_cagr_gap:+.2f}pp net CAGR. Gross alpha gap: {gross_cagr_gap:+.2f}pp; cost gap: {cost_gap_pct:+.2f}pp."

    md_lines.append(conclusion)
    md_lines.append("")

    (out_dir / "performance_attribution.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {out_dir / 'performance_attribution.md'}")

    print("\n" + "=" * 60)
    print("PERFORMANCE ATTRIBUTION")
    print("=" * 60)
    print(f"  Net CAGR gap: {net_cagr_gap:+.2f}pp")
    print(f"  Gross CAGR gap: {gross_cagr_gap:+.2f}pp")
    print(f"  Cost gap: {cost_gap_pct:+.2f}pp")
    print(f"  Driver: {driver}")
    print("=" * 60)
    print(f"\nConclusion: {conclusion}")
    print("=" * 60)


if __name__ == "__main__":
    main()
