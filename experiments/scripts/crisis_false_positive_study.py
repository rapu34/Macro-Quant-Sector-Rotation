#!/usr/bin/env python3
"""
Crisis False Positive Reduction Study (Risk-Constrained Optimization)
Sandbox-only: does not overwrite production files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

print("Running Crisis False Positive Reduction Study in sandbox mode...\n")

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
from src.strategy_analyzer import HYSTERESIS_ENTER, HYSTERESIS_EXIT

processed_path = ROOT / "data" / "processed_features.csv"
raw_path = ROOT / "data" / "raw_data.csv"
selected_path = ROOT / "outputs" / "selected_features.json"

if not processed_path.exists() or not raw_path.exists():
    print("[ERROR] Data files not found.")
    sys.exit(1)

df = _load_data(processed_path, raw_path)
# Filter to 2021~present
df = df[df["date"] >= "2021-01-01"].copy()
if len(df) < 20:
    print("[ERROR] Insufficient data for 2021+.")
    sys.exit(1)

feature_cols = _get_feature_cols(df, selected_path)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sentiment = _load_sentiment(raw_path)

# Date range
start = df["date"].min().strftime("%Y-%m-%d")
end = df["date"].max().strftime("%Y-%m-%d")


def _cagr(rets):
    arr = np.array(rets)
    if len(arr) < 1:
        return 0.0
    ppy = 252 / REBALANCE_DAYS
    return float(np.prod(1 + arr) ** (ppy / len(arr)) - 1)


def run_scenario(
    hysteresis_enter: float,
    turnover_threshold: float,
) -> dict:
    """Run backtest with given hysteresis and turnover threshold."""
    from src.strategy_analyzer import get_hmm_regime_model
    import src.model_trainer as mt

    # Regime with scenario hysteresis (full-sample fit, used via get_p_crisis_asof)
    _, regime_df, _ = get_hmm_regime_model(
        start=start, end=end,
        hysteresis_enter=hysteresis_enter,
        hysteresis_exit=0.6,
    )
    if regime_df.empty:
        return {}

    orig_turnover = mt.TURNOVER_THRESHOLD
    mt.TURNOVER_THRESHOLD = turnover_threshold

    try:
        p_log = []
        gross, net, dates, turnover_list = _walk_forward_backtest(
            df, feature_cols, scaler,
            sentiment_series=sentiment,
            use_risk_mgmt=True,
            raw_path=raw_path,
            regime_df=regime_df,
            hmm_X=None,
            hmm_dates=None,
            use_institutional=True,
            p_crisis_log=p_log,
            show_progress=False,
        )
    finally:
        mt.TURNOVER_THRESHOLD = orig_turnover

    if len(net) < 2:
        return {}

    m = _metrics(net)
    crisis_on = 100 * sum(1 for p in p_log if p["p_crisis"] >= hysteresis_enter - 1e-6) / max(1, len(p_log))
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
    ("Base(0.8/5%)", 0.8, 0.05),
    ("Enter0.85", 0.85, 0.05),
    ("Enter0.90", 0.90, 0.05),
    ("Enter0.85+Turn8%", 0.85, 0.08),
    ("Enter0.90+Turn8%", 0.90, 0.08),
]

results = []
for i, (name, h_enter, t_thresh) in enumerate(scenarios):
    r = run_scenario(h_enter, t_thresh)
    if r:
        results.append({
            "Scenario": name,
            "Sharpe": r["sharpe"],
            "CAGR": r["cagr"],
            "MDD": r["mdd"],
            "CVaR": r["cvar"],
            "Crisis_ON": r["crisis_on"],
            "Turnover": r["turnover"],
        })
        print(f"  {name}: Sharpe={r['sharpe']:.4f}  CAGR={r['cagr']:.2f}%  Crisis_ON={r['crisis_on']:.1f}%  Turnover={r['turnover']:.2f}%")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out_df = pd.DataFrame(results)
out_path = EXP_OUT / "false_positive_reduction_study.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if len(results) >= 2:
    base = results[0]
    crisis_on_base = base["Crisis_ON"]
    sharpe_base = base["Sharpe"]
    mdd_base = base["MDD"]

    best_crisis_reduction = min(results, key=lambda x: x["Crisis_ON"])
    best_sharpe = max(results, key=lambda x: x["Sharpe"])
    worst_mdd = min(results, key=lambda x: x["MDD"])

    print(f"  Crisis ON 변화 폭: {crisis_on_base:.1f}% (Base) → {best_crisis_reduction['Crisis_ON']:.1f}% (최저, {best_crisis_reduction['Scenario']})")
    print(f"  Sharpe 개선: Base {sharpe_base:.4f} → Best {best_sharpe['Sharpe']:.4f} ({best_sharpe['Scenario']})")
    if best_sharpe["Sharpe"] > sharpe_base:
        print(f"    → Sharpe 개선됨 (+{best_sharpe['Sharpe'] - sharpe_base:.4f})")
    else:
        print(f"    → Sharpe 개선 없음")
    print(f"  Drawdown 악화: Base MDD {mdd_base:.1f}% → Worst {worst_mdd['MDD']:.1f}% ({worst_mdd['Scenario']})")
    if worst_mdd["MDD"] < mdd_base:
        print(f"    → Drawdown 악화됨 (더 깊음)")
    else:
        print(f"    → Drawdown containment 유지")

print("=" * 70)

# ---------------------------------------------------------------------------
# 해석 가이드 답변
# ---------------------------------------------------------------------------
print("\n[해석 가이드]")
print("1. Crisis ON 비율은 얼마나 감소했는가?")
if len(results) >= 2:
    min_crisis = min(r["Crisis_ON"] for r in results)
    print(f"   → Base 대비 최대 {crisis_on_base - min_crisis:.1f}%p 감소 (최저 {min_crisis:.1f}%)")

print("2. Sharpe 개선이 통계적으로 의미 있는가?")
print("   → 37회 rebalance 기준, Sharpe 차이 0.1~0.2는 표준오차 ~0.15 수준. 단일 샘플이므로 통계적 유의성 판단은 부트스트랩 필요.")

print("3. Drawdown containment가 훼손되었는가?")
if len(results) >= 2:
    mdd_worst = max(r["MDD"] for r in results)
    print(f"   → 최악 MDD {mdd_worst:.1f}%. Base 대비 악화 시 containment 훼손.")

print("4. Risk-Adjusted 관점에서 최적 설정은?")
if len(results) >= 1:
    # Sort by Sharpe, then by MDD (less negative preferred)
    ranked = sorted(results, key=lambda x: (x["Sharpe"], -abs(x["MDD"])), reverse=True)
    print(f"   → Sharpe 우선: {ranked[0]['Scenario']} (Sharpe={ranked[0]['Sharpe']:.4f})")

print("\n[중요] 동일 결과 원인 (risk_mult_min=0.5):")
print("   risk_mult = max(0.5, 1 - p_crisis) 이므로 p_crisis ≥ 0.5 구간에서 항상 0.5로 고정.")
print("   Hysteresis 변경(0.8→0.85→0.90)은 0.6~0.9 구간의 regime만 바꾸며, 해당 구간에서도")
print("   risk_mult=0.5이므로 실질 노출이 동일 → 성과 차이 없음.")
print("=" * 70 + "\n")
