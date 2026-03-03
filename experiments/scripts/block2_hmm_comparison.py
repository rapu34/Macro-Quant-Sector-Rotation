#!/usr/bin/env python3
"""
Block2 Raw vs Block2 + HMM comparison (True Daily PnL 기준).

1) Block2 Raw vs Block2 + HMM 단독 비교
2) (Block1 + Block2 Raw) vs (Block1 + Block2 HMM) 결합 비교

주의: Target vol scaling 추가 금지, Double scaling 금지, Daily smoothing 금지
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS = ROOT / "outputs"

CRISIS_THRESHOLD = 0.5  # P_Crisis >= 0.5 -> risk_mult = 0.5


def _load_hmm_risk_mult() -> pd.Series:
    """HMM P_Crisis에서 risk_mult 유도. crisis면 0.5, 아니면 1.0."""
    for p in [EXP_OUT / "hmm_regime.csv", OUTPUTS / "hmm_regime.csv"]:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            if "P_Crisis" not in df.columns:
                continue
            risk_mult = pd.Series(
                np.where(df["P_Crisis"] >= CRISIS_THRESHOLD, 0.5, 1.0),
                index=df.index,
            )
            return risk_mult
    raise FileNotFoundError("hmm_regime.csv not found in experiments/outputs or outputs")


def compute_metrics(r: pd.Series) -> tuple:
    r = r.dropna()
    if len(r) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    ann_return = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.nan
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax()).min() - 1
    cvar = r[r <= r.quantile(0.05)].mean()
    return ann_return, ann_vol, sharpe, mdd, cvar


def stress_corr(x: pd.Series, y: pd.Series) -> float:
    """Portfolio worst 10% days에서 x vs y 상관계수."""
    worst_days = x.nsmallest(max(1, int(len(x) * 0.1))).index
    common = worst_days.intersection(y.index)
    if len(common) < 2:
        return np.nan
    return x.loc[common].corr(y.loc[common])


def main():
    # STEP 1 — 데이터 로드
    block1 = pd.read_csv(EXP_OUT / "true_daily_block1.csv", index_col=0, parse_dates=True).squeeze()
    block2_raw = pd.read_csv(EXP_OUT / "true_daily_block2.csv", index_col=0, parse_dates=True).squeeze()

    # STEP 2 — HMM risk multiplier
    risk_mult = _load_hmm_risk_mult()
    risk_mult = risk_mult.reindex(block2_raw.index).ffill().bfill().fillna(1.0)
    block2_hmm = block2_raw * risk_mult.shift(1)

    # STEP 3 — 단독 성과 비교
    metrics = {}
    for name, series in [("Block2_Raw", block2_raw), ("Block2_HMM", block2_hmm)]:
        metrics[name] = compute_metrics(series)

    single_block_df = pd.DataFrame(metrics, index=["AnnReturn", "AnnVol", "Sharpe", "MDD", "CVaR"]).T

    # STEP 4 — 결합 포트폴리오
    equal_raw = 0.5 * block1 + 0.5 * block2_raw
    equal_hmm = 0.5 * block1 + 0.5 * block2_hmm

    portfolio_df = {}
    for name, series in [("Equal_Raw", equal_raw), ("Equal_Block2_HMM", equal_hmm)]:
        portfolio_df[name] = compute_metrics(series)

    portfolio_metrics_df = pd.DataFrame(portfolio_df, index=["AnnReturn", "AnnVol", "Sharpe", "MDD", "CVaR"]).T

    # STEP 5 — Stress Correlation
    corr_results = {
        "Full_corr_raw": block1.corr(block2_raw),
        "Full_corr_hmm": block1.corr(block2_hmm),
        "Stress_corr_raw": stress_corr(equal_raw, block2_raw),
        "Stress_corr_hmm": stress_corr(equal_hmm, block2_hmm),
    }
    corr_df = pd.Series(corr_results)

    # STEP 6 — 저장
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    single_block_df.to_csv(EXP_OUT / "block2_single_comparison.csv")
    portfolio_metrics_df.to_csv(EXP_OUT / "block2_portfolio_comparison.csv")
    corr_df.to_csv(EXP_OUT / "block2_corr_comparison.csv")

    print("Block2 단독 비교:")
    print(single_block_df)
    print("\n포트폴리오 비교:")
    print(portfolio_metrics_df)
    print("\n상관 비교:")
    print(corr_df)


if __name__ == "__main__":
    main()
