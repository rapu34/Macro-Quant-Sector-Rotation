#!/usr/bin/env python3
"""
Cost sensitivity analysis: 10bps (base) vs 20bps stress scenario.
Uses production net return + turnover. No model retraining, no model.pkl changes.
Output: outputs/cost_sensitivity_analysis.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

BASE_COST = 0.001   # 10 bps (one-way)
STRESS_COST = 0.002  # 20 bps (one-way)
REBALANCE_DAYS = 20


def main():
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

    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
    hmm_X, hmm_dates = get_hmm_input_data(start, end)

    p_crisis_log = []
    gross_rets, net_rets, rebal_dates, turnover_list = _walk_forward_backtest(
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

    # Stress: additional_cost = turnover * (STRESS_COST - BASE_COST)
    additional_cost = np.array(turnover_list) * (STRESS_COST - BASE_COST)
    stress_net_rets = np.array(net_rets) - additional_cost

    # Build output DataFrame
    rows = []
    for i, rd in enumerate(rebal_dates):
        rows.append({
            "date": str(rd)[:10],
            "original_net_return": net_rets[i],
            "stress_net_return": float(stress_net_rets[i]),
            "turnover": turnover_list[i],
            "additional_cost": float(additional_cost[i]),
        })
    df_out = pd.DataFrame(rows)

    out_path = ROOT / "outputs" / "cost_sensitivity_analysis.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Metrics
    m_base = _metrics(net_rets)
    m_stress = _metrics(stress_net_rets.tolist())

    def _cagr(returns: list) -> float:
        arr = np.array(returns)
        if len(arr) < 1:
            return 0.0
        n_per = len(arr)
        periods_per_year = 252 / REBALANCE_DAYS
        total = np.prod(1 + arr)
        return float(total ** (periods_per_year / n_per) - 1) * 100 if n_per > 0 else 0.0

    sharpe_base = m_base["sharpe"]
    sharpe_stress = m_stress["sharpe"]
    sharpe_diff = sharpe_stress - sharpe_base

    print("\n" + "=" * 60)
    print("COST ASSUMPTION")
    print("=" * 60)
    print(f"  BASE_COST:   {BASE_COST*10000:.0f} bps (one-way)")
    print(f"  STRESS_COST: {STRESS_COST*10000:.0f} bps (one-way)")
    print("=" * 60)
    print("COST SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"  BASE Sharpe:  {sharpe_base:.4f}")
    print(f"  STRESS Sharpe: {sharpe_stress:.4f}")
    print(f"  Sharpe diff:  {sharpe_diff:.4f}")
    print("-" * 60)
    print(f"  BASE CAGR:    {_cagr(net_rets):.2f}%")

    print(f"  STRESS CAGR:  {_cagr(stress_net_rets.tolist()):.2f}%")
    print("-" * 60)
    print(f"  BASE MDD:     {m_base['mdd']*100:.2f}%")
    print(f"  STRESS MDD:   {m_stress['mdd']*100:.2f}%")
    print("-" * 60)
    print(f"  BASE CVaR:    {m_base['cvar']*100:.2f}%")
    print(f"  STRESS CVaR:  {m_stress['cvar']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
