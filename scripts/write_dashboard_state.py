#!/usr/bin/env python3
"""
Write dashboard state files: latest_state.json, run_history.csv.
Runs after pipeline. Reads from pipeline outputs only (no strategy modification).
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_suffix = "_refresh" if os.environ.get("PIPELINE_REFRESH_MODE") == "1" else ""
EXP_OUT = ROOT / "experiments" / f"outputs{_suffix}"
OUTPUTS = ROOT / f"outputs{_suffix}"
STATE_DIR = OUTPUTS / "state"
LOG_DIR = OUTPUTS / "logs"
LIVE_START = "2025-12-15"
PRINCIPAL = 3000
PRINCIPAL_CURRENCY = "SGD"
PRINCIPAL_START = "2025-12-15"


def _get_exp_out() -> Path:
    if (ROOT / "experiments" / "outputs_refresh").exists() and (
        ROOT / "experiments" / "outputs_refresh" / "true_daily_block1.csv"
    ).exists():
        return ROOT / "experiments" / "outputs_refresh"
    return ROOT / "experiments" / "outputs"


def _load_portfolio_returns() -> pd.Series | None:
    exp = _get_exp_out()
    try:
        b1 = pd.read_csv(exp / "true_daily_block1.csv", parse_dates=["date"]).set_index("date")["block1"].squeeze()
        b2_path = exp / "block2_hmm_expanding_rebalonly.csv"
        if b2_path.exists():
            b2 = pd.read_csv(b2_path, parse_dates=["date"]).set_index("date").iloc[:, 0].squeeze()
        else:
            b2 = pd.read_csv(exp / "true_daily_block2.csv", parse_dates=["date"]).set_index("date")["block2"].squeeze()
        common = b1.index.intersection(b2.index)
        r_p = 0.3 * b1.reindex(common).ffill().bfill().fillna(0) + 0.7 * b2.reindex(common).ffill().bfill().fillna(0)
        return r_p.dropna()
    except Exception:
        return None


def _get_p_crisis(out: Path) -> float:
    for p in [out / "hmm_regime.csv", ROOT / "outputs" / "hmm_regime.csv"]:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "P_Crisis" in df.columns:
                    return float(df["P_Crisis"].iloc[-1])
                if "p_crisis" in df.columns:
                    return float(df["p_crisis"].iloc[-1])
            except Exception:
                pass
    for p in [out / "p_crisis_log.csv", ROOT / "outputs" / "p_crisis_log.csv"]:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "p_crisis" in df.columns:
                    return float(df["p_crisis"].iloc[-1])
            except Exception:
                pass
    return 0.0


def _get_cvar(exp: Path) -> float | None:
    p = exp / "stress_conditional_stats.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        row = df[df["condition"] == "port_worst_5pct"]
        if len(row) > 0:
            return float(row["cvar_95"].iloc[0])
    except Exception:
        pass
    return None


def _get_beta(exp: Path) -> float | None:
    p = exp / "factor_regression_summary.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        row = df[df["factor"] == "r_mkt"]
        if len(row) > 0:
            return float(row["coef"].iloc[0])
    except Exception:
        pass
    return None


def _load_spy_vix(exp: Path) -> tuple[float | None, float | None]:
    """Load latest SPY daily return and delta_vix from benchmark_factor_data.csv."""
    p = exp / "benchmark_factor_data.csv"
    if not p.exists():
        return None, None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if len(df) < 1:
            return None, None
        row = df.iloc[-1]
        r_spy = float(row["r_spy"]) if "r_spy" in row.index else None
        dv = float(row["delta_vix"]) if "delta_vix" in row.index else None
        return r_spy, dv
    except Exception:
        return None, None


def _build_alerts(
    p_crisis: float,
    current_drawdown: float | None,
    r1d: float | None,
    roll_vol: float | None,
    spy_1d: float | None,
    delta_vix: float | None,
) -> tuple[list[dict], list[dict]]:
    """
    Build Core Alerts (Executive) and Monitoring Alerts.
    Returns (core_alerts, monitoring_alerts). Each alert: {severity, message, rule}.
    """
    core_alerts: list[dict] = []
    monitoring_alerts: list[dict] = []

    # 1. Regime Alert (CRITICAL)
    if p_crisis >= 0.60:
        core_alerts.append({
            "severity": "CRITICAL",
            "message": f"⚠ CRISIS REGIME WARNING\np_crisis = {p_crisis:.2f}\nMarket stress probability elevated",
            "rule": "regime",
        })

    # 2. Drawdown Alert (CRITICAL)
    if current_drawdown is not None and current_drawdown <= -0.10:
        core_alerts.append({
            "severity": "CRITICAL",
            "message": f"⚠ PORTFOLIO DRAWDOWN ALERT\nCurrent drawdown = {current_drawdown*100:.1f}%",
            "rule": "drawdown",
        })

    # 3. Daily Shock Alert (CRITICAL)
    if r1d is not None and r1d <= -0.02:
        core_alerts.append({
            "severity": "CRITICAL",
            "message": f"⚠ DAILY LOSS ALERT\n1d return = {r1d*100:.2f}%",
            "rule": "daily_loss",
        })

    # 4. Volatility Spike (WARNING)
    if roll_vol is not None and roll_vol >= 0.15:
        monitoring_alerts.append({
            "severity": "WARNING",
            "message": f"VOLATILITY WARNING\n63d volatility = {roll_vol*100:.1f}%",
            "rule": "vol_spike",
        })

    # 5. Market Stress (WARNING)
    if spy_1d is not None and spy_1d <= -0.03:
        monitoring_alerts.append({
            "severity": "WARNING",
            "message": f"MARKET STRESS\nSPY drop = {spy_1d*100:.2f}%",
            "rule": "spy_shock",
        })

    # 6. VIX Shock (WARNING)
    if delta_vix is not None and delta_vix >= 0.10:
        monitoring_alerts.append({
            "severity": "WARNING",
            "message": f"VIX SPIKE\nΔVIX = {delta_vix*100:.1f}%",
            "rule": "vix_spike",
        })

    return core_alerts, monitoring_alerts


def main() -> int:
    exp = _get_exp_out()
    out = OUTPUTS if OUTPUTS.exists() else ROOT / "outputs"
    state_dir = out / "state"
    log_dir = out / "logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    r_p = _load_portfolio_returns()
    if r_p is None or len(r_p) < 2:
        print("[write_dashboard_state] No portfolio data. Skipping.")
        return 0

    last_date = r_p.index[-1]
    asof_date = str(last_date.date())

    r1d = float(r_p.iloc[-1]) if len(r_p) >= 1 else None
    r5d = float((1 + r_p.tail(5)).prod() - 1) if len(r_p) >= 5 else None
    r21d = float((1 + r_p.tail(21)).prod() - 1) if len(r_p) >= 21 else None

    roll_vol = float(r_p.tail(63).std() * np.sqrt(252)) if len(r_p) >= 63 else None
    if len(r_p) >= 252:
        ann_ret = r_p.tail(252).mean() * 252
        ann_vol = r_p.tail(252).std() * np.sqrt(252)
        roll_sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else None
    else:
        roll_sharpe = None

    cum = (1 + r_p).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    current_dd = float(dd.iloc[-1]) if len(dd) > 0 else None

    p_crisis = _get_p_crisis(out)
    risk_mult = float(np.clip(1.0 - p_crisis, 0.5, 1.0))

    if p_crisis < 0.4:
        regime = "Core"
    elif p_crisis < 0.6:
        regime = "Elevated"
    else:
        regime = "Crisis"

    cvar = _get_cvar(exp)
    beta = _get_beta(exp)

    # Benchmark metrics (from stress_test output)
    bench_metrics = {}
    bench_path = exp / "benchmark_metrics.json"
    if bench_path.exists():
        try:
            with open(bench_path) as f:
                bench_metrics = json.load(f)
        except Exception:
            pass

    # SPY / VIX for monitoring alerts
    spy_1d, delta_vix = _load_spy_vix(exp)

    # Build Core + Monitoring alerts
    core_alerts, monitoring_alerts = _build_alerts(
        p_crisis, current_dd, r1d, roll_vol, spy_1d, delta_vix
    )

    # Cumulative return from principal start (2025-12-15) and current value
    start_ts = pd.Timestamp(PRINCIPAL_START)
    r_from_start = r_p[r_p.index >= start_ts]
    cum_ret_since_start = float((1 + r_from_start).prod() - 1) if len(r_from_start) > 0 else 0.0
    current_value = PRINCIPAL * (1 + cum_ret_since_start)

    # Current holdings (Block1+Block2 from pipeline)
    current_holdings = []
    last_rebalance_turnover = None
    holdings_path = exp / "current_holdings.json"
    if holdings_path.exists():
        try:
            with open(holdings_path) as f:
                h = json.load(f)
            current_holdings = h.get("holdings", [])
            last_rebalance_turnover = h.get("last_rebalance_turnover")
        except Exception:
            pass

    # Alert history (historical events from benchmark_comparison)
    alert_history = []
    alert_path = exp / "alert_history.csv"
    if alert_path.exists():
        try:
            ah = pd.read_csv(alert_path)
            alert_history = ah.to_dict("records") if "Date" in ah.columns and "Alert" in ah.columns else []
        except Exception:
            pass

    state = {
        "asof_date": asof_date,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "principal": PRINCIPAL,
        "principal_currency": PRINCIPAL_CURRENCY,
        "principal_start": PRINCIPAL_START,
        "cum_ret_since_start": cum_ret_since_start,
        "current_value": round(current_value, 2),
        "current_holdings": current_holdings,
        "1d_return": r1d,
        "1w_return": r5d,
        "1m_return": r21d,
        "rolling_vol_63d": roll_vol,
        "rolling_sharpe_252d": roll_sharpe,
        "current_drawdown": current_dd,
        "p_crisis": p_crisis,
        "risk_mult": risk_mult,
        "regime": regime,
        "core_alerts": core_alerts,
        "monitoring_alerts": monitoring_alerts,
        "beta": beta,
        "cvar_95": cvar,
        "live_start": LIVE_START,
        "source": "pipeline",
        "excess_return_ann": bench_metrics.get("excess_return_ann"),
        "tracking_error": bench_metrics.get("tracking_error"),
        "information_ratio": bench_metrics.get("information_ratio"),
        "sortino_ratio": bench_metrics.get("sortino_ratio"),
        "downside_deviation": bench_metrics.get("downside_deviation"),
        "alert_history": alert_history[:15],
        "last_rebalance_turnover": last_rebalance_turnover,
    }

    state_file = state_dir / "latest_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[write_dashboard_state] Wrote {state_file}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "asof_date": asof_date,
        "1d_return": r1d,
        "1w_return": r5d,
        "1m_return": r21d,
        "rolling_vol_63d": roll_vol,
        "current_drawdown": current_dd,
        "p_crisis": p_crisis,
        "regime": regime,
        "alert_triggered": "; ".join(a["rule"] for a in core_alerts) if core_alerts else "",
    }

    run_file = log_dir / "run_history.csv"
    run_df = pd.DataFrame([run_row])
    if run_file.exists():
        run_df = pd.concat([pd.read_csv(run_file), run_df], ignore_index=True)
    run_df.to_csv(run_file, index=False)
    print(f"[write_dashboard_state] Appended to {run_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
