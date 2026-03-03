#!/usr/bin/env python3
"""
Risk-Consistent Exposure Elasticity Refinement
Sandbox-only: does not modify production files.
Improves exposure elasticity in mid-probability range while keeping tail protection (risk_mult_min=0.5).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

print("Refining Risk Response Curve: Improving Exposure Elasticity under Fixed Tail Protection.\n")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
from src.model_trainer import (
    REBALANCE_DAYS,
    _load_data,
    _get_feature_cols,
    _load_sentiment,
    _metrics,
    _walk_forward_backtest,
)
from src.strategy_analyzer import get_hmm_regime_model, get_hmm_input_data, HYSTERESIS_ENTER

processed_path = ROOT / "data" / "processed_features.csv"
raw_path = ROOT / "data" / "raw_data.csv"
selected_path = ROOT / "outputs" / "selected_features.json"

if not processed_path.exists() or not raw_path.exists():
    print("[ERROR] Data files not found.")
    sys.exit(1)

df = _load_data(processed_path, raw_path)
df = df[df["date"] >= "2021-01-01"].copy()
if len(df) < 20:
    print("[ERROR] Insufficient data for 2021+.")
    sys.exit(1)

feature_cols = _get_feature_cols(df, selected_path)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sentiment = _load_sentiment(raw_path)

start = df["date"].min().strftime("%Y-%m-%d")
end = df["date"].max().strftime("%Y-%m-%d")


def _cagr(rets):
    arr = np.array(rets)
    if len(arr) < 1:
        return 0.0
    ppy = 252 / REBALANCE_DAYS
    return float(np.prod(1 + arr) ** (ppy / len(arr)) - 1)


def run_scenario(
    use_nonlinear: bool,
    turnover_threshold: float,
) -> dict:
    """Run backtest with linear or nonlinear risk response, and turnover threshold."""
    import src.model_trainer as mt

    # Expanding HMM: use hmm_X/hmm_dates for get_p_crisis_expanding (no lookahead)
    hmm_X, hmm_dates = get_hmm_input_data(start, end)
    if len(hmm_X) == 0 or len(hmm_dates) == 0:
        # Fallback to regime_df if expanding data unavailable
        _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
        regime_df, hmm_X, hmm_dates = regime_df, None, None
        if regime_df.empty:
            return {}
    else:
        regime_df = None

    orig_turnover = mt.TURNOVER_THRESHOLD
    mt.TURNOVER_THRESHOLD = turnover_threshold

    # Patch _compute_vol_scaled_weights for nonlinear scenario
    if use_nonlinear:
        _original = mt._compute_vol_scaled_weights

        def _patched_nonlinear(test_df, holdings, p_crisis, target_vol=None, kelly_cap=1.0,
                               exposure_diag=None, cov_matrix=None, gross_exposure_cap=None,
                               risk_mult_min=0.5, risk_mult_step=None):
            if risk_mult_step is not None:
                thresh, mult_val = risk_mult_step
                risk_mult = mult_val if p_crisis > thresh else 1.0
            else:
                base_mult = 1.0 - p_crisis ** 1.3
                risk_mult = max(risk_mult_min, base_mult)
            # Inject: pass p_effective so original's linear formula yields our risk_mult
            p_effective = 1.0 - risk_mult
            return _original(test_df, holdings, p_effective, target_vol=target_vol, kelly_cap=kelly_cap,
                            exposure_diag=exposure_diag, cov_matrix=cov_matrix,
                            gross_exposure_cap=gross_exposure_cap, risk_mult_min=risk_mult_min,
                            risk_mult_step=risk_mult_step)

        mt._compute_vol_scaled_weights = _patched_nonlinear

    try:
        p_log = []
        gross, net, dates, turnover_list = _walk_forward_backtest(
            df, feature_cols, scaler,
            sentiment_series=sentiment,
            use_risk_mgmt=True,
            raw_path=raw_path,
            regime_df=regime_df,
            hmm_X=hmm_X,
            hmm_dates=hmm_dates,
            use_institutional=True,
            p_crisis_log=p_log,
            show_progress=False,
        )
    finally:
        mt.TURNOVER_THRESHOLD = orig_turnover
        if use_nonlinear:
            mt._compute_vol_scaled_weights = _original

    if len(net) < 2:
        return {}

    m = _metrics(net)
    crisis_on = 100 * sum(1 for p in p_log if p["p_crisis"] >= HYSTERESIS_ENTER - 1e-6) / max(1, len(p_log))
    avg_turnover = np.mean(turnover_list) * 100 if turnover_list else np.nan

    return {
        "sharpe": m["sharpe"],
        "cagr": _cagr(net) * 100,
        "mdd": m["mdd"] * 100,
        "cvar": m["cvar"] * 100,
        "crisis_on": crisis_on,
        "turnover": avg_turnover,
    }


# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------
scenarios = [
    ("Base_linear", False, 0.05),
    ("Nonlinear_p1.3", True, 0.05),
    ("Nonlinear_p1.3_turn7", True, 0.07),
]

results = []
for name, use_nl, t_thresh in scenarios:
    r = run_scenario(use_nonlinear=use_nl, turnover_threshold=t_thresh)
    if r:
        results.append({
            "Scenario": name,
            "Sharpe": r["sharpe"],
            "CAGR": r["cagr"],
            "MDD": r["mdd"],
            "CVaR(95%)": r["cvar"],
            "Crisis_ON": r["crisis_on"],
            "Turnover": r["turnover"],
        })
        print(f"  {name}: Sharpe={r['sharpe']:.4f}  CAGR={r['cagr']:.2f}%  MDD={r['mdd']:.1f}%  CVaR={r['cvar']:.1f}%  Crisis_ON={r['crisis_on']:.1f}%  Turnover={r['turnover']:.2f}%")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out_df = pd.DataFrame(results)
out_path = EXP_OUT / "risk_elasticity_refinement.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

SHARPE_BASELINE = 1.04
if len(results) >= 1:
    base = results[0]
    sharpe_base = base["Sharpe"]
    mdd_base = base["MDD"]
    cvar_base = base["CVaR(95%)"]

    best_sharpe = max(results, key=lambda x: x["Sharpe"])
    worst_mdd = min(results, key=lambda x: x["MDD"])
    worst_cvar = min(results, key=lambda x: x["CVaR(95%)"])

    sharpe_improve = best_sharpe["Sharpe"] - SHARPE_BASELINE
    print(f"  Sharpe 개선 폭 (vs {SHARPE_BASELINE}): {sharpe_improve:+.4f} (Best: {best_sharpe['Scenario']})")
    print(f"  MDD 변화: Base {mdd_base:.1f}% → Worst {worst_mdd['MDD']:.1f}% ({worst_mdd['Scenario']})")
    print(f"  CVaR 변화: Base {cvar_base:.1f}% → Worst {worst_cvar['CVaR(95%)']:.1f}% ({worst_cvar['Scenario']})")
    tail_ok = worst_mdd["MDD"] >= mdd_base - 2 and worst_cvar["CVaR(95%)"] >= cvar_base - 2
    print(f"  Tail protection 유지: {'Yes' if tail_ok else 'Check MDD/CVaR degradation'}")

print("=" * 70)

# ---------------------------------------------------------------------------
# 해석 질문
# ---------------------------------------------------------------------------
print("\n[해석 질문]")
print("1. Sharpe 개선이 중간 확률 구간에서의 노출 증가 때문인가?")
if len(results) >= 2:
    nl = next((r for r in results if "Nonlinear" in r["Scenario"] and "turn7" not in r["Scenario"]), None)
    if nl and nl["Sharpe"] > base["Sharpe"]:
        print("   → Nonlinear(p^1.3)가 Base보다 Sharpe 높으면, 0~0.5 구간 완만한 감소로 FP 구간 수익 회수 기여 가능.")
    else:
        print("   → Sharpe 차이 없거나 Base 우위 시, 현재 구간에서 노출 탄력성 효과 제한적.")

print("2. Tail metric(CVaR, MDD)이 훼손되었는가?")
if len(results) >= 2:
    worst = max(results, key=lambda x: abs(x["MDD"]))
    print(f"   → 최악 MDD {worst['MDD']:.1f}%. Base 대비 2%p 이상 악화 시 tail 훼손.")

print("3. 개선이 구조적이며 과최적화 위험이 낮은가?")
print("   → p^1.3은 단일 파라미터, 해석 가능. 과최적화 위험은 부트스트랩/out-of-sample 검증으로 확인 필요.")
print("=" * 70 + "\n")
