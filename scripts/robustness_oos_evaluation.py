#!/usr/bin/env python3
"""
Full robustness and out-of-sample evaluation of the 2-state HMM risk engine.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_trainer import (
    REBALANCE_DAYS,
    _load_data,
    _get_feature_cols,
    _load_sentiment,
    _metrics,
    _walk_forward_backtest,
)
from src.strategy_analyzer import HYSTERESIS_ENTER


def _cagr(returns: list[float]) -> float:
    """CAGR from period returns."""
    arr = np.array(returns)
    if len(arr) < 1:
        return 0.0
    n_per = len(arr)
    periods_per_year = 252 / REBALANCE_DAYS
    total = np.prod(1 + arr)
    return float(total ** (periods_per_year / n_per) - 1) if n_per > 0 else 0.0


def _run_backtest_for_period(
    df: pd.DataFrame,
    feature_cols: list[str],
    raw_path: Path,
    start: str,
    end: str,
    regime_df: pd.DataFrame,
    hmm_X: np.ndarray,
    hmm_dates: pd.DatetimeIndex,
) -> tuple[list, list, list, list, list]:
    """Run backtest for period [start, end]. Returns (gross, net, dates, turnover, p_crisis_log)."""
    from sklearn.preprocessing import StandardScaler
    mask = (df["date"] >= start) & (df["date"] <= end)
    df_sub = df[mask].copy()
    if len(df_sub) < 20:
        return [], [], [], [], []

    scaler = StandardScaler()
    sentiment_series = _load_sentiment(raw_path)
    p_crisis_log = []
    gross_rets, net_rets, rebal_dates, turnover_list = _walk_forward_backtest(
        df_sub, feature_cols, scaler,
        sentiment_series=sentiment_series,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if hmm_X is not None and len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if hmm_dates is not None and len(hmm_dates) > 0 else None,
        use_institutional=True,
        p_crisis_log=p_crisis_log,
        show_progress=False,
    )
    return gross_rets, net_rets, rebal_dates, turnover_list, p_crisis_log


def _get_regime_for_period(start: str, end: str):
    """Get regime_df for period [start, end] (full-sample fit)."""
    from src.strategy_analyzer import get_hmm_regime_model
    _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
    return regime_df


def _get_hmm_input_for_period(start: str, end: str):
    """Get (hmm_X, hmm_dates) for period."""
    from src.strategy_analyzer import get_hmm_input_data
    return get_hmm_input_data(start=start, end=end)


def _get_frozen_oos_regime(train_start: str, train_end: str, apply_start: str, apply_end: str) -> pd.DataFrame:
    """Train HMM on [train_start, train_end], apply to [apply_start, apply_end] without refitting."""
    from src.strategy_analyzer import (
        _load_hmm_features_4d,
        _fit_hmm_2state,
        _apply_hysteresis,
        FEATURE_NAMES_4D,
    )
    from sklearn.preprocessing import StandardScaler

    out_train = _load_hmm_features_4d(start=train_start, end=train_end)
    if len(out_train) < 3:
        return pd.DataFrame()
    df_scaled_train, scaler_train, df_raw_train = out_train[0], out_train[1], out_train[2]
    if df_raw_train.empty or scaler_train is None:
        return pd.DataFrame()

    X_train = scaler_train.fit_transform(df_raw_train[FEATURE_NAMES_4D].values)
    model, p_crisis_train, probs, _, feature_means = _fit_hmm_2state(X_train, scaler=scaler_train)
    if model is None:
        return pd.DataFrame()

    out_apply = _load_hmm_features_4d(start=apply_start, end=apply_end)
    if len(out_apply) < 3:
        return pd.DataFrame()
    df_raw_apply = out_apply[2]
    if df_raw_apply.empty:
        return pd.DataFrame()

    X_apply_raw = df_raw_apply[FEATURE_NAMES_4D].values
    X_apply_scaled = scaler_train.transform(X_apply_raw)
    dates_apply = pd.DatetimeIndex(df_raw_apply.index)

    probs_apply = model.predict_proba(X_apply_scaled)
    states_apply = model.predict(X_apply_scaled)

    stress_scores = []
    for k in range(2):
        mask = states_apply == k
        if mask.any():
            fm = {FEATURE_NAMES_4D[j]: float(X_apply_raw[mask, j].mean()) for j in range(4)}
            stress_scores.append(fm.get("credit_stress", 0) + fm.get("vol_term", 0))
        else:
            stress_scores.append(-1e9)
    crisis_idx = int(np.argmax(stress_scores))
    p_crisis_apply = probs_apply[:, crisis_idx]
    p_crisis_hyst = _apply_hysteresis(p_crisis_apply, HYSTERESIS_ENTER, 0.6)

    return pd.DataFrame({
        "date": dates_apply,
        "state": states_apply,
        "P_Crisis": p_crisis_hyst,
    })


def run() -> None:
    processed_path = ROOT / "data" / "processed_features.csv"
    raw_path = ROOT / "data" / "raw_data.csv"
    selected_path = ROOT / "outputs" / "selected_features.json"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not processed_path.exists() or not raw_path.exists():
        print("[ERROR] Data files not found.")
        return

    df = _load_data(processed_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)
    data_start = raw.index.min().strftime("%Y-%m-%d")
    data_end = raw.index.max().strftime("%Y-%m-%d")

    # ---------------------------------------------------------------------------
    # 1. SUBPERIOD PERFORMANCE
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. SUBPERIOD PERFORMANCE TEST")
    print("=" * 70)

    subperiods = [
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020-2022", "2020-01-01", "2022-12-31"),
        ("2023-present", "2023-01-01", data_end),
    ]

    subperiod_rows = []
    for name, start, end in subperiods:
        regime_df = _get_regime_for_period(start, end)
        hmm_X, hmm_dates = _get_hmm_input_for_period(start, end)
        gross, net, dates, turnover, pc_log = _run_backtest_for_period(
            df, feature_cols, raw_path, start, end, regime_df, hmm_X, hmm_dates
        )
        if len(net) < 2:
            subperiod_rows.append({
                "period": name,
                "cagr": np.nan,
                "sharpe": np.nan,
                "mdd": np.nan,
                "cvar_95": np.nan,
                "pct_crisis_on": np.nan,
            })
            print(f"  {name}: Insufficient data")
            continue

        m = _metrics(net)
        cagr = _cagr(net)
        pct_crisis = 100 * sum(1 for p in pc_log if p["p_crisis"] >= HYSTERESIS_ENTER - 1e-6) / max(1, len(pc_log))

        subperiod_rows.append({
            "period": name,
            "cagr": cagr * 100,
            "sharpe": m["sharpe"],
            "mdd": m["mdd"] * 100,
            "cvar_95": m["cvar"] * 100,
            "pct_crisis_on": pct_crisis,
        })
        print(f"  {name}: CAGR={cagr*100:.2f}%  Sharpe={m['sharpe']:.4f}  MDD={m['mdd']*100:.2f}%  Crisis={pct_crisis:.1f}%")

    pd.DataFrame(subperiod_rows).to_csv(out_dir / "subperiod_performance.csv", index=False)
    print(f"\nSaved: {out_dir / 'subperiod_performance.csv'}")

    # ---------------------------------------------------------------------------
    # 2. TRUE OUT-OF-SAMPLE TEST
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. TRUE OUT-OF-SAMPLE TEST (Train 2010-2018, Apply 2019-present)")
    print("=" * 70)

    oos_regime = _get_frozen_oos_regime("2010-01-01", "2018-12-31", "2019-01-01", data_end)
    if oos_regime.empty:
        print("  OOS regime empty (check data availability).")
        oos_rows = [{"metric": "oos_sharpe", "value": np.nan}]
    else:
        hmm_X_oos, hmm_dates_oos = np.array([]), pd.DatetimeIndex([])
        gross_oos, net_oos, dates_oos, _, pc_oos = _run_backtest_for_period(
            df, feature_cols, raw_path, "2019-01-01", data_end,
            oos_regime, hmm_X_oos, hmm_dates_oos,
        )
        if len(net_oos) >= 2:
            m_oos = _metrics(net_oos)
            oos_rows = [
                {"metric": "oos_cagr", "value": _cagr(net_oos) * 100},
                {"metric": "oos_sharpe", "value": m_oos["sharpe"]},
                {"metric": "oos_mdd", "value": m_oos["mdd"] * 100},
                {"metric": "oos_cvar_95", "value": m_oos["cvar"] * 100},
                {"metric": "oos_pct_crisis_on", "value": 100 * sum(1 for p in pc_oos if p["p_crisis"] >= HYSTERESIS_ENTER - 1e-6) / max(1, len(pc_oos))},
            ]
            print(f"  OOS CAGR: {_cagr(net_oos)*100:.2f}% | Sharpe: {m_oos['sharpe']:.4f} | MDD: {m_oos['mdd']*100:.2f}%")
        else:
            oos_rows = [{"metric": "oos_sharpe", "value": np.nan}]
            print("  OOS: Insufficient data")

    pd.DataFrame(oos_rows).to_csv(out_dir / "out_of_sample_test.csv", index=False)
    print(f"Saved: {out_dir / 'out_of_sample_test.csv'}")

    # ---------------------------------------------------------------------------
    # 3. ROLLING 3-YEAR SHARPE
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. ROLLING 3-YEAR SHARPE")
    print("=" * 70)

    regime_full = _get_regime_for_period(data_start, data_end)
    hmm_X_full, hmm_dates_full = _get_hmm_input_for_period(data_start, data_end)
    gross_full, net_full, dates_full, _, _ = _run_backtest_for_period(
        df, feature_cols, raw_path, data_start, data_end,
        regime_full, hmm_X_full, hmm_dates_full,
    )

    periods_per_year = 252 / REBALANCE_DAYS
    n_per_3y = int(3 * periods_per_year)
    rolling_sharpe = []
    for i in (range(n_per_3y, len(net_full) + 1) if len(net_full) >= n_per_3y else []):
        window = net_full[i - n_per_3y : i]
        if len(window) >= 2 and np.std(window) > 1e-10:
            sr = (np.mean(window) / np.std(window)) * np.sqrt(periods_per_year)
        else:
            sr = np.nan
        rolling_sharpe.append({"date": str(dates_full[i - 1])[:10], "rolling_3y_sharpe": sr})

    rs_df = pd.DataFrame(rolling_sharpe)
    rs_df.to_csv(out_dir / "rolling_sharpe.csv", index=False)
    print(f"  Rolling periods: {len(rs_df)}")
    print(f"Saved: {out_dir / 'rolling_sharpe.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        valid = rs_df["rolling_3y_sharpe"].dropna()
        ax.plot(range(len(valid)), valid.values, color="steelblue")
        ax.set_xlabel("Period index")
        ax.set_ylabel("Rolling 3Y Sharpe")
        ax.set_title("Rolling 3-Year Sharpe Ratio")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "rolling_sharpe_plot.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  Plot skipped: {e}")

    # ---------------------------------------------------------------------------
    # 4. CRISIS SANITY PLOT
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. CRISIS SANITY PLOT")
    print("=" * 70)

    if not regime_full.empty and "P_Crisis" in regime_full.columns:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rdf = regime_full.copy()
            rdf["date"] = pd.to_datetime(rdf["date"])
            rdf = rdf.set_index("date").sort_index()

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(rdf.index, rdf["P_Crisis"], color="darkred", alpha=0.8, linewidth=0.8)

            highlights = [
                ("2018 Q4", "2018-10-01", "2018-12-31", "orange"),
                ("2020 Mar", "2020-03-01", "2020-03-31", "red"),
                ("2022 rate shock", "2022-01-01", "2022-06-30", "darkorange"),
            ]
            for label, s, e, c in highlights:
                ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.2, color=c, label=label)

            ax.set_xlabel("Date")
            ax.set_ylabel("P_Crisis")
            ax.set_title("P_Crisis Over Time (2-State HMM) — Stress Period Highlights")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.savefig(out_dir / "crisis_detection_plot.png", dpi=150)
            plt.close()
            print(f"Saved: {out_dir / 'crisis_detection_plot.png'}")
        except Exception as e:
            print(f"  Crisis plot failed: {e}")
    else:
        print("  No regime data for crisis plot.")

    # ---------------------------------------------------------------------------
    # 5. FINAL ROBUSTNESS SUMMARY
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. ROBUSTNESS SUMMARY")
    print("=" * 70)

    sp_df = pd.DataFrame(subperiod_rows)
    valid_sharpes = sp_df["sharpe"].dropna()
    subperiod_stability = 1.0 - float(valid_sharpes.std()) if len(valid_sharpes) > 1 else 1.0
    subperiod_stability = max(0, min(1, subperiod_stability))

    in_sample_sharpe = float(_metrics(net_full)["sharpe"]) if net_full and len(net_full) >= 2 else np.nan
    oos_df = pd.DataFrame(oos_rows)
    oos_sharpe = float(oos_df[oos_df["metric"] == "oos_sharpe"]["value"].iloc[0]) if "oos_sharpe" in oos_df["metric"].values else np.nan

    sharpe_gap = in_sample_sharpe - oos_sharpe if pd.notna(in_sample_sharpe) and pd.notna(oos_sharpe) else np.nan
    max_sharpe_dd_gap = float(valid_sharpes.max() - valid_sharpes.min()) if len(valid_sharpes) > 1 else np.nan

    robustness = 100
    if pd.notna(sharpe_gap) and sharpe_gap > 0.5:
        robustness -= 20
    if subperiod_stability < 0.5:
        robustness -= 20
    if pd.notna(max_sharpe_dd_gap) and max_sharpe_dd_gap > 2.0:
        robustness -= 15
    robustness = max(0, min(100, robustness))

    print(f"  Subperiod stability score:  {subperiod_stability:.2f}")
    print(f"  OOS Sharpe:                 {oos_sharpe:.4f}" if pd.notna(oos_sharpe) else "  OOS Sharpe:                 N/A")
    print(f"  In-sample Sharpe:           {in_sample_sharpe:.4f}" if pd.notna(in_sample_sharpe) else "  In-sample Sharpe:           N/A")
    print(f"  OOS vs In-sample gap:       {sharpe_gap:+.4f}" if pd.notna(sharpe_gap) else "  OOS vs In-sample gap:       N/A")
    print(f"  Max Sharpe drawdown gap:    {max_sharpe_dd_gap:.4f}" if pd.notna(max_sharpe_dd_gap) else "  Max Sharpe drawdown gap:    N/A")
    print(f"  Final robustness rating:    {robustness}/100")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run()
