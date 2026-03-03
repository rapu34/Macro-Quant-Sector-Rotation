#!/usr/bin/env python3
"""
Block 2 — Standalone Cross-Sectional Sector Momentum Strategy.

Signal: 12M-1M (skip-month) momentum. Top 3 equal weight. Monthly rebalance.
No risk management. For risk-source diversification analysis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
EXP_DATA = Path(__file__).resolve().parent.parent / "data"
EXP_OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(ROOT))

from src.config import SECTOR_ETFS

COST_RATE = 0.001  # 0.1% per side
TOP_K = 3
MOMENTUM_LOOKBACK = 252   # 12 months
MOMENTUM_SKIP = 21        # skip last 1 month (21 trading days)
REBALANCE_DAYS = 21       # monthly (~21 trading days)
REAL_RATE_CHANGE_THRESHOLD = 0.5
ROLLING_12M_PERIODS = 12
ROLLING_CORR_MONTHS = 36

PPY = 252 / REBALANCE_DAYS  # periods per year for Block 2


def _load_raw_prices(raw_path: Path) -> pd.DataFrame:
    """Load sector prices from raw data."""
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    cols = [c for c in SECTOR_ETFS if c in raw.columns]
    return raw[cols].ffill().dropna(how="all")


def _compute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    12M-1M momentum: log return from t-252 to t-21 (exclude last 21 days).
    momentum_t = log(P_{t-21} / P_{t-252})
    """
    log_p = np.log(prices)
    ret_12m = log_p - log_p.shift(MOMENTUM_LOOKBACK)
    ret_1m = log_p - log_p.shift(MOMENTUM_SKIP)
    mom = ret_12m - ret_1m  # 12M minus 1M
    return mom


def _run_block2_backtest(raw_path: Path) -> tuple:
    """Run Block 2 momentum backtest. Returns (net_rets, rebal_dates, turnover_list, daily_rets)."""
    prices = _load_raw_prices(raw_path)
    mom = _compute_momentum(prices)

    dates = mom.dropna(how="all").index
    # First valid momentum at t requires data from t-252 to t-21
    first_idx = MOMENTUM_LOOKBACK
    dates = dates[first_idx:]

    # Rebalance every 21 days (monthly)
    rebal_dates = []
    net_rets = []
    turnover_list = []
    prev_holdings = None
    prev_weights = None

    for i in range(0, len(dates), REBALANCE_DAYS):
        rebal_date = dates[i]
        mom_row = mom.loc[rebal_date]
        if mom_row.isna().all():
            continue

        # Rank by momentum, pick top 3
        rank = mom_row.rank(ascending=False, method="min")
        top3 = rank[rank <= TOP_K].index.tolist()
        if len(top3) < TOP_K:
            top3 = rank.nsmallest(TOP_K).index.tolist()
        top3 = top3[:TOP_K]
        holdings = set(top3)
        weights = {s: 1.0 / len(top3) for s in top3}

        # Period return: from this rebal to next (or end)
        end_idx = min(i + REBALANCE_DAYS, len(dates))
        period_end_date = dates[end_idx - 1] if end_idx > i else rebal_date
        start_prices = prices.loc[rebal_date]
        end_prices = prices.loc[period_end_date] if period_end_date != rebal_date else prices.shift(-1).loc[rebal_date]

        if prev_holdings is None:
            # First period: use current holdings
            period_ret = 0.0
            for s in holdings:
                p0 = start_prices.get(s, np.nan)
                p1 = end_prices.get(s, np.nan) if hasattr(end_prices, "get") else end_prices[s]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    period_ret += weights[s] * (np.log(p1 / p0) if p1 > 0 else 0)
            cost = len(holdings) * COST_RATE * (1.0 / TOP_K)  # one-way to enter
        else:
            # Compute gross return
            period_ret = 0.0
            for s in holdings:
                p0 = start_prices.get(s, np.nan)
                p1 = end_prices.get(s, np.nan) if hasattr(end_prices, "get") else end_prices[s]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    period_ret += weights[s] * (np.log(p1 / p0) if p1 > 0 else 0)

            # Turnover and cost
            sold = prev_holdings - holdings
            bought = holdings - prev_holdings
            turnover = sum(prev_weights.get(s, 0) for s in sold) + sum(weights.get(s, 0) for s in bought)
            turnover_list.append(turnover)
            cost = turnover * COST_RATE

        net_rets.append(period_ret - cost)
        rebal_dates.append(rebal_date)
        prev_holdings = holdings
        prev_weights = weights

    # Build daily return series for correlation
    daily_rets = _period_to_daily_returns(net_rets, rebal_dates, prices.index, REBALANCE_DAYS)
    return net_rets, rebal_dates, turnover_list, daily_rets


def _period_to_daily_returns(
    period_rets: list, rebal_dates: list, all_dates: pd.DatetimeIndex, default_period_days: int
) -> pd.Series:
    """Convert period returns to daily returns (for correlation)."""
    all_dates = pd.DatetimeIndex(all_dates).sort_values()
    daily = pd.Series(0.0, index=all_dates)
    for i, (r, d) in enumerate(zip(period_rets, rebal_dates)):
        start = pd.Timestamp(d)
        if i + 1 < len(rebal_dates):
            end = pd.Timestamp(rebal_dates[i + 1])
        else:
            end = all_dates[-1]
        mask = (all_dates >= start) & (all_dates <= end)
        n_days = mask.sum()
        if n_days > 0:
            daily_ret = (1 + r) ** (1 / n_days) - 1
            daily.loc[mask] = daily_ret
    return daily


def _get_block1_returns(raw_path: Path, proc_path: Path) -> tuple:
    """Run Block 1 backtest and return (net_rets, rebal_dates, daily_rets)."""
    from src.model_trainer import _load_data, _get_feature_cols, _load_sentiment, _walk_forward_backtest
    from sklearn.preprocessing import StandardScaler

    selected_path = ROOT / "outputs" / "selected_features.json"
    df = _load_data(proc_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols) < 3:
        feature_cols = [c for c in df.columns if c not in {"date", "sector", "target", "fwd_ret_20d"}][:6]

    scaler = StandardScaler()
    sentiment = _load_sentiment(raw_path)
    regime_df = pd.DataFrame()
    hmm_X, hmm_dates = np.array([]), pd.DatetimeIndex([])
    try:
        from src.strategy_analyzer import get_hmm_regime_model, get_hmm_input_data
        start = df["date"].min().strftime("%Y-%m-%d")
        end = df["date"].max().strftime("%Y-%m-%d")
        _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
        hmm_X, hmm_dates = get_hmm_input_data(start=start, end=end)
    except Exception:
        pass

    gross, net, rebal_dates, _ = _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=True,
        show_progress=False,
    )
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    daily_rets = _period_to_daily_returns(net, rebal_dates, raw.index, 20)
    return net, rebal_dates, daily_rets


def _sharpe(rets, ppy: float = PPY):
    arr = np.array(rets)
    if len(arr) < 2 or arr.std() < 1e-10:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(ppy))


def _mdd(rets):
    arr = np.array(rets)
    wealth = np.ones(len(arr) + 1)
    for i, r in enumerate(arr):
        wealth[i + 1] = max(0.0, wealth[i] * (1.0 + r))
    peak = np.maximum.accumulate(wealth[1:])
    dd = np.where(peak > 1e-12, (wealth[1:] - peak) / peak, 0)
    return float(np.min(dd)) * 100


def _ann_ret(rets, ppy: float = PPY):
    arr = np.array(rets)
    return float(np.prod(1 + arr) ** (ppy / len(arr)) - 1) * 100 if len(arr) > 0 else 0


def _volatility(rets, ppy: float = PPY):
    arr = np.array(rets)
    return float(arr.std() * np.sqrt(ppy)) * 100 if len(arr) > 1 else 0


def _get_real_rate_regime(raw_path: Path, rebal_dates):
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    if "treasury_10y" not in raw.columns or "cpi_all_urban" not in raw.columns:
        return {}
    tc10 = raw["treasury_10y"].ffill()
    cpi = raw["cpi_all_urban"].ffill()
    cpi_yoy = cpi.pct_change(252) * 100
    real_rate = tc10 - cpi_yoy
    rr_change_6m = real_rate - real_rate.shift(126)
    regimes = {}
    for d in rebal_dates:
        dts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
        try:
            idx = raw.index.get_indexer([dts], method="ffill")[0]
        except Exception:
            regimes[d] = "unknown"
            continue
        if idx < 126:
            regimes[d] = "unknown"
            continue
        chg = rr_change_6m.iloc[idx]
        if pd.isna(chg):
            regimes[d] = "unknown"
        elif chg > REAL_RATE_CHANGE_THRESHOLD:
            regimes[d] = "rising"
        elif chg < -REAL_RATE_CHANGE_THRESHOLD:
            regimes[d] = "falling"
        else:
            regimes[d] = "flat"
    return regimes


def main():
    EXP_OUT.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "raw_data_extended_2005.csv"
    proc_path = DATA_DIR / "processed_features_extended_2005.csv"
    if not raw_path.exists():
        exp_raw = EXP_DATA / "raw_data_extended_2005.csv"
        exp_proc = EXP_DATA / "processed_features_extended_2005.csv"
        if exp_raw.exists():
            import shutil
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(exp_raw, raw_path)
            shutil.copy(exp_proc, proc_path)
        else:
            raise FileNotFoundError("Extended data not found. Run block1_extended_validation.py first.")

    print("\n=== Block 2: Sector Momentum (12M-1M) ===")
    print("  Signal: 12M minus 1M log return. Top 3 equal weight. Monthly rebalance.")
    print("  No risk management.\n")

    # Block 2 backtest
    net_rets, rebal_dates, turnover_list, daily_rets_b2 = _run_block2_backtest(raw_path)
    avg_turnover = np.mean(turnover_list) * 100 if turnover_list else 0

    # STEP 1 — Standalone performance
    print("=== STEP 1: Standalone Performance ===")
    sharpe = _sharpe(net_rets)
    mdd = _mdd(net_rets)
    ann_ret = _ann_ret(net_rets)
    vol = _volatility(net_rets)
    roll_sh = pd.Series(net_rets, index=pd.DatetimeIndex(rebal_dates)).rolling(
        ROLLING_12M_PERIODS, min_periods=ROLLING_12M_PERIODS
    ).apply(lambda x: _sharpe(x.tolist()) if len(x) >= 2 and x.std() > 1e-10 else np.nan)
    roll_sh_valid = roll_sh.dropna()
    roll_sh_mean = float(roll_sh_valid.mean()) if len(roll_sh_valid) > 0 else np.nan
    roll_sh_std = float(roll_sh_valid.std()) if len(roll_sh_valid) > 1 else np.nan

    print(f"  Net Sharpe: {sharpe:.4f}")
    print(f"  Net MDD: {mdd:.2f}%")
    print(f"  Annualized Return: {ann_ret:.2f}%")
    print(f"  Volatility: {vol:.2f}%")
    print(f"  Turnover: {avg_turnover:.2f}%")
    print(f"  Rolling 12M Sharpe: mean={roll_sh_mean:.4f}, std={roll_sh_std:.4f}")

    # STEP 2 — Crisis analysis
    print("\n=== STEP 2: Crisis Analysis ===")
    crises = [
        ("2008", "2008-01-01", "2008-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
    ]
    crisis_results = []
    for name, start, end in crises:
        mask = [pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end) for d in rebal_dates]
        sub = [net_rets[i] for i in range(len(net_rets)) if i < len(mask) and mask[i]]
        if len(sub) >= 2:
            crisis_results.append({
                "period": name,
                "sharpe": _sharpe(sub),
                "mdd": _mdd(sub),
                "ret": _ann_ret(sub, ppy=252 / 21) if len(sub) > 0 else np.nan,
                "n": len(sub),
            })
        else:
            crisis_results.append({"period": name, "sharpe": np.nan, "mdd": np.nan, "ret": np.nan, "n": len(sub)})
        print(f"  {name}: Sharpe={crisis_results[-1]['sharpe']:.4f}, MDD={crisis_results[-1]['mdd']:.2f}%, n={crisis_results[-1]['n']}")

    # STEP 3 — Correlation vs Block 1
    print("\n=== STEP 3: Correlation vs Block 1 ===")
    _, _, daily_rets_b1 = _get_block1_returns(raw_path, proc_path)
    # Align daily returns
    common = daily_rets_b1.index.intersection(daily_rets_b2.index)
    b1_aligned = daily_rets_b1.reindex(common).ffill().bfill().fillna(0)
    b2_aligned = daily_rets_b2.reindex(common).ffill().bfill().fillna(0)
    full_corr = float(b1_aligned.corr(b2_aligned))

    # Rolling 36M correlation (in trading days: 36*21 ≈ 756)
    roll_days = 36 * 21
    roll_corr = b1_aligned.rolling(roll_days, min_periods=roll_days).corr(b2_aligned)
    roll_corr_valid = roll_corr.dropna()
    roll_corr_mean = float(roll_corr_valid.mean()) if len(roll_corr_valid) > 0 else np.nan
    roll_corr_max = float(roll_corr_valid.max()) if len(roll_corr_valid) > 0 else np.nan

    # 2020 and 2022 correlation
    mask_2020 = (common >= "2020-01-01") & (common <= "2020-12-31")
    mask_2022 = (common >= "2022-01-01") & (common <= "2022-12-31")
    corr_2020 = float(b1_aligned.loc[mask_2020].corr(b2_aligned.loc[mask_2020])) if mask_2020.sum() > 10 else np.nan
    corr_2022 = float(b1_aligned.loc[mask_2022].corr(b2_aligned.loc[mask_2022])) if mask_2022.sum() > 10 else np.nan

    print(f"  Full-sample correlation: {full_corr:.4f}")
    print(f"  Rolling 36M: mean={roll_corr_mean:.4f}, max={roll_corr_max:.4f}")
    print(f"  2020 correlation: {corr_2020:.4f}")
    print(f"  2022 correlation: {corr_2022:.4f}")

    # STEP 4 — Regime dependency
    print("\n=== STEP 4: Regime Dependency ===")
    rr_regimes = _get_real_rate_regime(raw_path, rebal_dates)
    regime_results = {}
    for regime in ["rising", "falling", "flat"]:
        idxs = [i for i, d in enumerate(rebal_dates) if rr_regimes.get(d) == regime]
        sub = [net_rets[i] for i in idxs if i < len(net_rets)]
        regime_results[regime] = _sharpe(sub) if len(sub) >= 2 else np.nan
        print(f"  {regime}: Sharpe={regime_results[regime]:.4f}, n={len(sub)}")

    # STEP 5 — Save deliverables
    # block2_returns.csv
    ret_df = pd.DataFrame({
        "rebalance_date": rebal_dates,
        "period_return": net_rets,
    })
    ret_df.to_csv(EXP_OUT / "block2_returns.csv", index=False)

    # block2_rolling_sharpe.csv
    roll_sh_df = pd.DataFrame({
        "date": roll_sh.index,
        "rolling_12m_sharpe": roll_sh.values,
    }).dropna(subset=["rolling_12m_sharpe"])
    roll_sh_df.to_csv(EXP_OUT / "block2_rolling_sharpe.csv", index=False)

    # Report
    report = f"""# Block 2 Momentum Extended Report

> Standalone cross-sectional sector momentum (12M-1M). Risk-source diversification analysis.

## Setup

- **Data**: 2005-01-03 ~ 2026-01-28 (extended)
- **Signal**: 12M minus 1M log return (skip last 21 days)
- **Universe**: Same sector ETFs as Block 1
- **Portfolio**: Top 3 equal weight
- **Rebalance**: Monthly (21 trading days)
- **Cost**: 0.1% per side
- **Risk Management**: None

## STEP 1 — Standalone Performance

| Metric | Value |
|--------|-------|
| Net Sharpe | {sharpe:.4f} |
| Net MDD | {mdd:.2f}% |
| Annualized Return | {ann_ret:.2f}% |
| Volatility | {vol:.2f}% |
| Turnover | {avg_turnover:.2f}% |
| Rolling 12M Sharpe (mean) | {roll_sh_mean:.4f} |
| Rolling 12M Sharpe (std) | {roll_sh_std:.4f} |
| Number of rebalances | {len(net_rets)} |

## STEP 2 — Crisis Analysis

| Period | Sharpe | MDD | Return | N |
|--------|--------|-----|--------|---|
"""
    for r in crisis_results:
        sh_s = f"{r['sharpe']:.4f}" if not np.isnan(r['sharpe']) else "—"
        mdd_s = f"{r['mdd']:.2f}%" if not np.isnan(r['mdd']) else "—"
        ret_s = f"{r['ret']:.2f}%" if not np.isnan(r['ret']) else "—"
        report += f"| {r['period']} | {sh_s} | {mdd_s} | {ret_s} | {r['n']} |\n"

    report += f"""
## STEP 3 — Correlation vs Block 1

| Metric | Value |
|--------|-------|
| Full-sample correlation | {full_corr:.4f} |
| Rolling 36M correlation (mean) | {roll_corr_mean:.4f} |
| Rolling 36M correlation (max) | {roll_corr_max:.4f} |
| 2020 correlation | {corr_2020:.4f} |
| 2022 correlation | {corr_2022:.4f} |

*Correlation is an observation, not a design target.*

## STEP 4 — Regime Dependency (Real Rate)

| Regime | Sharpe |
|--------|--------|
| Rising | {regime_results.get('rising', np.nan):.4f} |
| Falling | {regime_results.get('falling', np.nan):.4f} |
| Flat | {regime_results.get('flat', np.nan):.4f} |

---
*Generated by experiments/scripts/block2_momentum_extended.py*
*Block 1 and Block 2 are NOT combined. Goal: validate Block 2 as economically independent return stream.*
"""

    with open(EXP_OUT / "block2_momentum_extended_report.md", "w") as f:
        f.write(report)
    print(f"\nSaved: {EXP_OUT / 'block2_momentum_extended_report.md'}")
    print(f"Saved: {EXP_OUT / 'block2_returns.csv'}")
    print(f"Saved: {EXP_OUT / 'block2_rolling_sharpe.csv'}")


if __name__ == "__main__":
    main()
