"""
Risk Efficiency Optimization: Reduce Crisis ON ratio while maintaining MDD/CVaR improvement.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

REBALANCE_DAYS = 20
CRISIS_ON_THRESHOLD = 0.8  # P_crisis >= this = Crisis ON


def _run_backtest(
    regime_df: pd.DataFrame,
    risk_mult_min: float = 0.01,
    risk_mult_step=None,
    use_institutional: bool = True,
) -> tuple[dict, list, list, list, list]:
    """Run backtest with given params. Returns (metrics, gross_rets, net_rets, rebal_dates, turnover_list)."""
    from src.model_trainer import (
        _get_feature_cols,
        _load_data,
        _load_sentiment,
        _metrics,
        _walk_forward_backtest,
    )
    from sklearn.preprocessing import StandardScaler

    processed_path = ROOT / "data" / "processed_features.csv"
    raw_path = ROOT / "data" / "raw_data.csv"
    selected_path = ROOT / "outputs" / "selected_features.json"

    df = _load_data(processed_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    sentiment_series = _load_sentiment(raw_path)

    gross_rets, net_rets, rebal_dates, turnover_list = _walk_forward_backtest(
        df, feature_cols, StandardScaler(),
        sentiment_series=sentiment_series,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df if use_institutional else pd.DataFrame(),
        hmm_X=None,
        hmm_dates=None,
        use_institutional=use_institutional,
        show_progress=False,
        risk_mult_min=risk_mult_min,
        risk_mult_step=risk_mult_step,
    )
    m = _metrics(net_rets)
    n_periods = len(net_rets)
    years = n_periods * REBALANCE_DAYS / 252
    cagr = (1 + m["cum_return"]) ** (1 / max(0.01, years)) - 1 if years > 0 else 0
    m["cagr"] = cagr
    m["turnover"] = np.mean(turnover_list) if turnover_list else 0
    return m, gross_rets, net_rets, rebal_dates, turnover_list


def _crisis_diagnostics(p_crisis_log: pd.DataFrame, net_rets: list, rebal_dates: list) -> dict:
    """Compute crisis activation diagnostics."""
    if p_crisis_log.empty or len(net_rets) != len(rebal_dates):
        return {}
    pc = p_crisis_log.copy()
    pc["date"] = pd.to_datetime(pc["date"])
    pc["crisis_on"] = pc["p_crisis"] >= CRISIS_ON_THRESHOLD
    pc["period_ret"] = net_rets

    total = len(pc)
    on_ratio = pc["crisis_on"].mean()

    # ON duration: consecutive ON periods
    runs = []
    in_run = False
    run_len = 0
    for v in pc["crisis_on"].values:
        if v:
            run_len += 1
            in_run = True
        else:
            if in_run:
                runs.append(run_len)
            run_len = 0
            in_run = False
    if in_run:
        runs.append(run_len)
    avg_duration = np.mean(runs) if runs else 0

    years = total * REBALANCE_DAYS / 252
    annual_on_count = len(runs) / max(0.01, years) if years > 0 else 0

    on_rets = pc[pc["crisis_on"]]["period_ret"]
    off_rets = pc[~pc["crisis_on"]]["period_ret"]
    avg_ret_on = on_rets.mean() if len(on_rets) > 0 else np.nan
    avg_ret_off = off_rets.mean() if len(off_rets) > 0 else np.nan

    return {
        "crisis_on_ratio_pct": on_ratio * 100,
        "avg_on_duration": avg_duration,
        "annual_on_count": annual_on_count,
        "avg_ret_on": avg_ret_on,
        "avg_ret_off": avg_ret_off,
    }


def main():
    from src.strategy_analyzer import get_hmm_regime_model

    EXP_OUT.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "raw_data.csv", index_col=0, parse_dates=True)
    start = raw.index.min().strftime("%Y-%m-%d")
    end = raw.index.max().strftime("%Y-%m-%d")

    # --- 1) Crisis Activation Diagnostics (Base) ---
    print("\n[1] Crisis Activation Diagnostics...")
    model, regime_df, _ = get_hmm_regime_model(start=start, end=end)
    if regime_df.empty:
        print("ERROR: No regime data")
        return

    m_base, gross, net, rebal, turnover = _run_backtest(regime_df)
    # Align: rebal_dates to p_crisis (asof each rebalance date)
    pc_aligned = []
    for d in rebal:
        sub = regime_df[regime_df["date"] <= pd.Timestamp(d)]
        if not sub.empty:
            pc_aligned.append({"date": d, "p_crisis": sub.iloc[-1]["P_Crisis"], "risk_mult": 1 - sub.iloc[-1]["P_Crisis"]})
    pc_df = pd.DataFrame(pc_aligned) if pc_aligned else pd.DataFrame(columns=["date", "p_crisis", "risk_mult"])

    diag = _crisis_diagnostics(pc_df, net, rebal)
    diag_rows = [{"metric": k, "value": v} for k, v in diag.items()]
    pd.DataFrame(diag_rows).to_csv(EXP_OUT / "crisis_activation_diagnostics.csv", index=False)
    print(f"  Crisis ON ratio: {diag.get('crisis_on_ratio_pct', 0):.1f}%")
    print(f"  Avg ON duration: {diag.get('avg_on_duration', 0):.2f}")
    print(f"  Annual ON count: {diag.get('annual_on_count', 0):.1f}")
    print(f"  Avg ret ON: {diag.get('avg_ret_on', 0)*100:.2f}%")
    print(f"  Avg ret OFF: {diag.get('avg_ret_off', 0)*100:.2f}%")

    # --- 2) Threshold Sensitivity ---
    print("\n[2] Threshold Sensitivity Test...")
    hyst_results = []
    for name, enter, exit_ in [("A", 0.85, 0.65), ("B", 0.9, 0.7), ("C", 0.95, 0.75)]:
        _, regime_h, _ = get_hmm_regime_model(start=start, end=end, hysteresis_enter=enter, hysteresis_exit=exit_)
        if regime_h.empty:
            continue
        m, _, _, _, _ = _run_backtest(regime_h)
        hyst_results.append({
            "case": name,
            "hysteresis": f"{enter}/{exit_}",
            "cagr": m["cagr"] * 100,
            "sharpe": m["sharpe"],
            "mdd_pct": m["mdd"] * 100,
            "cvar_pct": m["cvar"] * 100,
            "turnover": m["turnover"],
        })
    pd.DataFrame(hyst_results).to_csv(EXP_OUT / "hysteresis_sensitivity_test.csv", index=False)

    # --- 3) Risk Mult Sensitivity ---
    print("\n[3] Risk Mult Sensitivity...")
    risk_results = []
    # Base (min=0.01)
    m0, _, _, _, _ = _run_backtest(regime_df, risk_mult_min=0.01)
    risk_results.append({
        "case": "base",
        "config": "min=0.01",
        "cagr": m0["cagr"] * 100,
        "sharpe": m0["sharpe"],
        "mdd_pct": m0["mdd"] * 100,
        "cvar_pct": m0["cvar"] * 100,
        "turnover": m0["turnover"],
    })
    # Case 1: min=0.5
    m1, _, _, _, _ = _run_backtest(regime_df, risk_mult_min=0.5)
    risk_results.append({
        "case": "min_0.5",
        "config": "min=0.5",
        "cagr": m1["cagr"] * 100,
        "sharpe": m1["sharpe"],
        "mdd_pct": m1["mdd"] * 100,
        "cvar_pct": m1["cvar"] * 100,
        "turnover": m1["turnover"],
    })
    # Case 2: min=0.6
    m2, _, _, _, _ = _run_backtest(regime_df, risk_mult_min=0.6)
    risk_results.append({
        "case": "min_0.6",
        "config": "min=0.6",
        "cagr": m2["cagr"] * 100,
        "sharpe": m2["sharpe"],
        "mdd_pct": m2["mdd"] * 100,
        "cvar_pct": m2["cvar"] * 100,
        "turnover": m2["turnover"],
    })
    # Case 3: step (P>0.8 -> 0.7, else 1.0)
    m3, _, _, _, _ = _run_backtest(regime_df, risk_mult_step=(0.8, 0.7))
    risk_results.append({
        "case": "step_0.8_0.7",
        "config": "P>0.8->0.7 else 1.0",
        "cagr": m3["cagr"] * 100,
        "sharpe": m3["sharpe"],
        "mdd_pct": m3["mdd"] * 100,
        "cvar_pct": m3["cvar"] * 100,
        "turnover": m3["turnover"],
    })
    pd.DataFrame(risk_results).to_csv(EXP_OUT / "risk_mult_sensitivity.csv", index=False)

    # --- 4) Crisis Definition ---
    print("\n[4] Crisis Definition Test...")
    crisis_results = []
    for sel, name in [("worst_cvar", "Alt1_worst_cvar"), ("stress_intersection", "Alt2_stress_intersection")]:
        try:
            _, regime_alt, _ = get_hmm_regime_model(start=start, end=end, crisis_selector=sel)
            if regime_alt.empty:
                continue
            m_alt, _, _, _, _ = _run_backtest(regime_alt)
            crisis_results.append({
                "case": name,
                "cagr": m_alt["cagr"] * 100,
                "sharpe": m_alt["sharpe"],
                "mdd_pct": m_alt["mdd"] * 100,
                "cvar_pct": m_alt["cvar"] * 100,
                "turnover": m_alt["turnover"],
            })
        except Exception as e:
            print(f"  {name} failed: {e}")
    if crisis_results:
        pd.DataFrame(crisis_results).to_csv(EXP_OUT / "crisis_definition_sensitivity.csv", index=False)

    # --- 5) Final Report ---
    print("\n" + "=" * 70)
    print("RISK EFFICIENCY OPTIMIZATION — FINAL REPORT")
    print("=" * 70)

    m_no_rm, _, _, _, _ = _run_backtest(pd.DataFrame(), use_institutional=False)

    cagr_no = m_no_rm.get("cagr", m_no_rm["cum_return"]) * 100
    cagr_base = m_base.get("cagr", m_base["cum_return"]) * 100
    print("\n| Scenario              | CAGR   | Sharpe | MDD    | CVaR   |")
    print("|-----------------------|--------|--------|--------|--------|")
    print(f"| Net (no Risk Mgmt)    | {cagr_no:5.1f}% | {m_no_rm['sharpe']:.3f}  | {m_no_rm['mdd']*100:5.1f}% | {m_no_rm['cvar']*100:5.1f}% |")
    print(f"| Base 4-state          | {cagr_base:5.1f}% | {m_base['sharpe']:.3f}  | {m_base['mdd']*100:5.1f}% | {m_base['cvar']*100:5.1f}% |")

    best_hyst = max(hyst_results, key=lambda x: x["sharpe"]) if hyst_results else None
    if best_hyst:
        print(f"| Threshold best ({best_hyst['case']})   | {best_hyst['cagr']:5.1f}% | {best_hyst['sharpe']:.3f}  | {best_hyst['mdd_pct']:5.1f}% | {best_hyst['cvar_pct']:5.1f}% |")

    best_risk = max(risk_results, key=lambda x: x["sharpe"]) if risk_results else None
    if best_risk:
        print(f"| Risk_mult best ({best_risk['case'][:8]}) | {best_risk['cagr']:5.1f}% | {best_risk['sharpe']:.3f}  | {best_risk['mdd_pct']:5.1f}% | {best_risk['cvar_pct']:5.1f}% |")

    if crisis_results:
        best_crisis = max(crisis_results, key=lambda x: x["sharpe"])
        print(f"| Crisis def best        | {best_crisis['cagr']:5.1f}% | {best_crisis['sharpe']:.3f}  | {best_crisis['mdd_pct']:5.1f}% | {best_crisis['cvar_pct']:5.1f}% |")

    # Verdict
    sharpe_improve = m_base["sharpe"] > m_no_rm["sharpe"]
    mdd_improve = (m_no_rm["mdd"] - m_base["mdd"]) >= 0.02
    sharpe_flat = abs(m_base["sharpe"] - m_no_rm["sharpe"]) <= 0.05
    verdict = (
        "Risk engine improves risk-adjusted efficiency"
        if sharpe_improve or (sharpe_flat and mdd_improve)
        else "Risk engine economically inefficient"
    )
    print("\n" + "=" * 70)
    print("VERDICT:", verdict)
    print("=" * 70)

    print(f"\nSaved: {EXP_OUT / 'crisis_activation_diagnostics.csv'}")
    print(f"Saved: {EXP_OUT / 'hysteresis_sensitivity_test.csv'}")
    print(f"Saved: {EXP_OUT / 'risk_mult_sensitivity.csv'}")
    if crisis_results:
        print(f"Saved: {EXP_OUT / 'crisis_definition_sensitivity.csv'}")


if __name__ == "__main__":
    main()
