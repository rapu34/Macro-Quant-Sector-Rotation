#!/usr/bin/env python3
"""
9-sector equal-weight baseline. Same backtest rules as production:
rebalance period (20d), turnover skip/cap, cost (10bps one-way).
Outputs: baseline_equal_weight.csv, baseline_equal_weight_summary.csv, model_vs_equal_weight.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _run_equal_weight_backtest(df: pd.DataFrame) -> tuple[list, list, list, list, list, list]:
    """Run equal-weight backtest. Returns (gross, cost, net, turnover_raw, turnover_caps, dates)."""
    from src.config import SECTOR_ETFS
    from src.model_trainer import (
        COST_RATE,
        ROUND_TRIP_RATE,
        TURNOVER_THRESHOLD,
        TURNOVER_CAP_THRESHOLD,
        TURNOVER_CAP_RATE,
        REBALANCE_DAYS,
        MIN_TRAIN_PCT,
        _turnover,
        _apply_turnover_cap,
    )

    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    n_dates = len(dates)
    min_train_idx = max(1, int(n_dates * MIN_TRAIN_PCT))
    step = max(1, REBALANCE_DAYS)
    EW_K = len(SECTOR_ETFS)

    gross_rets = []
    cost_list = []
    turnover_raw_list = []
    turnover_caps_list = []
    rebal_dates = []

    prev_holdings = None
    prev_weights = None

    for test_start_idx in range(min_train_idx, n_dates, step):
        test_date = dates[test_start_idx]
        test_df = df[df["date"] == test_date]

        if len(test_df) < len(SECTOR_ETFS):
            continue

        holdings = set(SECTOR_ETFS)
        target_weights = {s: 1.0 / EW_K for s in SECTOR_ETFS}
        prev_weights_full = prev_weights or {s: 0.0 for s in SECTOR_ETFS}

        turnover_raw = _turnover(prev_weights_full, target_weights)
        new_weights_full = _apply_turnover_cap(prev_weights_full, target_weights)
        turnover_after_caps = _turnover(prev_weights_full, new_weights_full)

        sector_rebalance_cost = 0.0
        if turnover_after_caps >= TURNOVER_THRESHOLD or prev_holdings is None:
            if prev_holdings is None:
                sector_rebalance_cost = EW_K * COST_RATE * (1.0 / EW_K)
            elif holdings != prev_holdings:
                n_changed = len(holdings.symmetric_difference(prev_holdings))
                sector_rebalance_cost = n_changed * ROUND_TRIP_RATE * (1.0 / EW_K)

        if turnover_after_caps < TURNOVER_THRESHOLD and prev_holdings is not None:
            weights_held = prev_weights or {s: 1.0 / EW_K for s in holdings}
        else:
            weights_held = {s: new_weights_full.get(s, 0) for s in SECTOR_ETFS}

        active_holdings = {s for s in SECTOR_ETFS if weights_held.get(s, 0) > 1e-8}
        if not active_holdings:
            active_holdings = holdings
        period_ret = sum(
            test_df[test_df["sector"] == s]["fwd_ret_20d"].values[0] * weights_held.get(s, 0)
            for s in active_holdings if s in test_df["sector"].values
        )

        gross_rets.append(period_ret)
        cost_list.append(sector_rebalance_cost)
        turnover_raw_list.append(turnover_raw)
        turnover_caps_list.append(turnover_after_caps)
        rebal_dates.append(test_date)

        prev_holdings = holdings
        prev_weights = {s: new_weights_full.get(s, 0) for s in SECTOR_ETFS}

    net_rets = [g - c for g, c in zip(gross_rets, cost_list)]
    return gross_rets, cost_list, net_rets, turnover_raw_list, turnover_caps_list, rebal_dates


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

    # 1) Equal-weight baseline
    gross_ew, cost_ew, net_ew, to_raw, to_caps, dates_ew = _run_equal_weight_backtest(df)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_baseline = pd.DataFrame({
        "date": [str(d)[:10] for d in dates_ew],
        "gross_return": gross_ew,
        "cost_deducted": cost_ew,
        "net_return": net_ew,
        "turnover_raw": to_raw,
        "turnover_after_caps": to_caps,
    })
    df_baseline.to_csv(out_dir / "baseline_equal_weight.csv", index=False)
    print(f"Saved: {out_dir / 'baseline_equal_weight.csv'}")

    m_ew = _metrics(net_ew)

    def _cagr(returns: list) -> float:
        arr = np.array(returns)
        if len(arr) < 1:
            return 0.0
        n_per = len(arr)
        periods_per_year = 252 / 20
        total = np.prod(1 + arr)
        return float(total ** (periods_per_year / n_per) - 1) * 100 if n_per > 0 else 0.0

    avg_turnover_ew = np.mean(to_caps) * 100 if to_caps else np.nan

    df_summary = pd.DataFrame([{
        "Sharpe": m_ew["sharpe"],
        "CAGR": _cagr(net_ew),
        "MDD": m_ew["mdd"] * 100,
        "CVaR95": m_ew["cvar"] * 100,
        "AvgTurnover": avg_turnover_ew,
    }])
    df_summary.to_csv(out_dir / "baseline_equal_weight_summary.csv", index=False)
    print(f"Saved: {out_dir / 'baseline_equal_weight_summary.csv'}")

    # 2) Production model net return (same source as cost_sensitivity)
    feature_cols = _get_feature_cols(df, selected_path)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sentiment = _load_sentiment(raw_path)
    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
    hmm_X, hmm_dates = get_hmm_input_data(start, end)

    p_crisis_log = []
    _, net_model, dates_model, turnover_model = _walk_forward_backtest(
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

    m_model = _metrics(net_model)
    avg_turnover_model = np.mean(turnover_model) * 100 if turnover_model else np.nan

    # 3) Compare: rows=Model_Net/EqualWeight_Net, cols=Sharpe,CAGR,MDD,CVaR95,AvgTurnover
    df_compare = pd.DataFrame(
        {
            "Sharpe": [m_model["sharpe"], m_ew["sharpe"]],
            "CAGR": [_cagr(net_model), _cagr(net_ew)],
            "MDD": [m_model["mdd"] * 100, m_ew["mdd"] * 100],
            "CVaR95": [m_model["cvar"] * 100, m_ew["cvar"] * 100],
            "AvgTurnover": [avg_turnover_model, avg_turnover_ew],
        },
        index=["Model_Net", "EqualWeight_Net"],
    )
    df_compare.to_csv(out_dir / "model_vs_equal_weight.csv")
    print(f"Saved: {out_dir / 'model_vs_equal_weight.csv'}")

    # Console summary
    print("\n" + "=" * 60)
    print("EQUAL-WEIGHT BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Sharpe:     {m_ew['sharpe']:.4f}")
    print(f"  CAGR:       {_cagr(net_ew):.2f}%")
    print(f"  MDD:        {m_ew['mdd']*100:.2f}%")
    print(f"  CVaR95:     {m_ew['cvar']*100:.2f}%")
    print(f"  AvgTurnover: {avg_turnover_ew:.2f}%")
    print("=" * 60)
    print("MODEL vs EQUAL-WEIGHT")
    print("=" * 60)
    print(f"  {'Metric':<12} {'Model_Net':>12} {'EqualWeight_Net':>16}")
    print("-" * 60)
    for col in ["Sharpe", "CAGR", "MDD", "CVaR95", "AvgTurnover"]:
        v1 = df_compare.loc["Model_Net", col]
        v2 = df_compare.loc["EqualWeight_Net", col]
        if col in ("CAGR", "MDD", "CVaR95", "AvgTurnover"):
            print(f"  {col:<12} {v1:>12.2f} {v2:>16.2f}")
        else:
            print(f"  {col:<12} {v1:>12.4f} {v2:>16.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
