"""Risk Analytics — Tail risk, Stress scenarios, Factor exposure."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.state_loader import get_exp_out, load_state
from dashboard.utils.styles import SHARED_CSS
from dashboard.utils.metrics import colored_metric
from dashboard.utils.date_filter import get_chart_period, filter_series_by_period, filter_by_period

st.set_page_config(page_title="Risk Analytics", layout="wide", page_icon="📊")
get_chart_period()
st.markdown(SHARED_CSS, unsafe_allow_html=True)
exp = get_exp_out()
state = load_state()


def _fmt(v, pct=True):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if pct and isinstance(v, float) and abs(v) < 10:
        return f"{v*100:.2f}%"
    return f"{v:.2f}"


# 1. Tail risk
st.markdown("**Tail Risk Metrics**")
cond_path = exp / "stress_conditional_stats.csv"
if cond_path.exists():
    df = pd.read_csv(cond_path)
    port_5 = df[df["condition"] == "port_worst_5pct"]
    port_1 = df[df["condition"] == "port_worst_1pct"]
    cols = st.columns(4)
    with cols[0]:
        v = float(port_5["cvar_95"].iloc[0]) if len(port_5) > 0 else None
        colored_metric("CVaR (95%)", _fmt(v) if v else "—", "neg" if v and v < -0.02 else "neutral")
    with cols[1]:
        v = float(port_1["mean_port_return"].iloc[0]) if len(port_1) > 0 else None
        colored_metric("Worst 1% avg loss", _fmt(v), "neg" if v and v < 0 else "neutral")
    with cols[2]:
        v = float(port_5["mean_port_return"].iloc[0]) if len(port_5) > 0 else None
        colored_metric("Worst 5% avg loss", _fmt(v), "neg" if v and v < 0 else "neutral")
    with cols[3]:
        v = state.get("current_drawdown")
        status = "neg" if v and v <= -0.05 else ("warn" if v and v <= -0.02 else "pos")
        colored_metric("Current Drawdown", _fmt(v), status)
else:
    st.info("Run stress test to populate tail risk metrics.")

# Top 5 worst days
st.markdown("**Top 5 Worst Trading Days**")
worst_path = exp / "stress_top5_worst_days.csv"
if worst_path.exists():
    df = pd.read_csv(worst_path)
    col_map = {df.columns[0]: "Date"}
    for c in df.columns:
        if "r_port" in c or c == "r_port": col_map[c] = "Portfolio"
        if "r_spy" in c or c == "r_spy": col_map[c] = "SPY"
    df = df.rename(columns=col_map)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("No data.")

# 2. Stress scenarios
st.markdown("---")
st.markdown("**Stress Scenario Analysis**")
if cond_path.exists():
    df = pd.read_csv(cond_path)
    spy3 = df[df["condition"] == "spy_selloff_3pct"]
    vix10 = df[df["condition"] == "vix_spike_10pct"]
    cols = st.columns(2)
    with cols[0]:
        if len(spy3) > 0:
            v = float(spy3["mean_port_return"].iloc[0])
            colored_metric("SPY ≤ −3% days avg return", f"{_fmt(v)} (N={int(spy3['N'].iloc[0])})", "neg" if v < 0 else "neutral")
    with cols[1]:
        if len(vix10) > 0:
            v = float(vix10["mean_port_return"].iloc[0])
            colored_metric("ΔVIX ≥ +10% days avg return", f"{_fmt(v)} (N={int(vix10['N'].iloc[0])})", "neg" if v < 0 else "neutral")
else:
    st.caption("Run stress test.")

# 3. Benchmark Risk Metrics
st.markdown("---")
st.markdown("**Benchmark Risk Metrics**")
bench_path = exp / "benchmark_metrics.json"
if not bench_path.exists():
    bench_path = ROOT / "experiments" / "outputs" / "benchmark_metrics.json"
if bench_path.exists():
    import json
    with open(bench_path) as f:
        bm = json.load(f)
    cols = st.columns(5)
    with cols[0]:
        colored_metric("Beta vs SPY", _fmt(state.get("beta"), pct=False), "neutral")
    with cols[1]:
        colored_metric("Tracking Error", _fmt(bm.get("tracking_error")), "neutral")
    with cols[2]:
        colored_metric("Information Ratio", _fmt(bm.get("information_ratio"), pct=False), "pos" if bm.get("information_ratio", 0) > 0 else "neutral")
    with cols[3]:
        colored_metric("Downside Deviation", _fmt(bm.get("downside_deviation")), "neutral")
    with cols[4]:
        colored_metric("Sortino Ratio", _fmt(bm.get("sortino_ratio"), pct=False), "pos" if bm.get("sortino_ratio", 0) > 0 else "neutral")
else:
    st.caption("Run pipeline for benchmark risk metrics.")

# 4. Factor exposure
st.markdown("---")
st.markdown("**Factor Exposure**")
factor_path = exp / "factor_regression_summary.csv"
if factor_path.exists():
    df = pd.read_csv(factor_path)
    r2_row = df[df["factor"] == "R2"]
    beta_row = df[df["factor"] == "r_mkt"]
    mom_row = df[df["factor"] == "r_mom"]
    vol_row = df[df["factor"] == "delta_vix"]
    cols = st.columns(4)
    with cols[0]:
        v = float(beta_row["coef"].iloc[0]) if len(beta_row) > 0 else None
        st.metric("Market beta", _fmt(v, pct=False), help="SPY exposure")
    with cols[1]:
        v = float(mom_row["coef"].iloc[0]) if len(mom_row) > 0 else None
        st.metric("Momentum exposure", _fmt(v, pct=False))
    with cols[2]:
        v = float(vol_row["coef"].iloc[0]) if len(vol_row) > 0 else None
        st.metric("Volatility sensitivity", _fmt(v, pct=False), help="ΔVIX exposure")
    with cols[3]:
        v = float(r2_row["coef"].iloc[0]) if len(r2_row) > 0 else None
        st.metric("R²", _fmt(v, pct=False), help="Factor model fit")
else:
    st.info("Run factor regression.")

# 5. Regime Performance Analysis
st.markdown("---")
st.markdown("**Regime Performance Analysis**")
regime_path = exp / "regime_performance.csv"
if not regime_path.exists():
    regime_path = ROOT / "experiments" / "outputs" / "regime_performance.csv"
if regime_path.exists():
    df_reg = pd.read_csv(regime_path)
    st.dataframe(df_reg, use_container_width=True, hide_index=True)
    st.caption("Strategy shows different risk-return behaviour across regimes.")
else:
    st.caption("Run pipeline for regime analysis.")

# 6. Time-Series Risk
st.markdown("---")
st.markdown("**Time-Series Risk**")
port_path = exp / "true_daily_block1.csv"
b2_path = exp / "block2_hmm_expanding_rebalonly.csv"
if not b2_path.exists():
    b2_path = exp / "true_daily_block2.csv"
roll_beta_path = exp / "factor_rolling_betas.csv"
if not roll_beta_path.exists():
    roll_beta_path = ROOT / "experiments" / "outputs" / "factor_rolling_betas.csv"
if port_path.exists() and b2_path.exists():
    b1 = pd.read_csv(port_path, parse_dates=["date"]).set_index("date")["block1"].squeeze()
    b2 = pd.read_csv(b2_path, parse_dates=["date"]).set_index("date").iloc[:, 0].squeeze()
    common = b1.index.intersection(b2.index)
    r_p = 0.3 * b1.reindex(common).ffill().bfill().fillna(0) + 0.7 * b2.reindex(common).ffill().bfill().fillna(0)
    r_p = r_p.dropna()
    post = r_p[r_p.index >= "2013-06-05"]
    post = filter_series_by_period(post)
    if len(post) >= 63:
        roll_vol = post.rolling(63).std() * (252**0.5) * 100
        cum = (1 + post).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, mode="lines", name="Rolling Vol (63d)", line=dict(color="#ffd93d", width=1.5)))
        fig_vol.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white", xaxis_title="", yaxis_title="Vol (Ann. %)")
        st.plotly_chart(fig_vol, use_container_width=True)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown", line=dict(color="#ff6b6b", width=1.5), fill="tozeroy", fillcolor="rgba(255,107,107,0.2)"))
        fig_dd.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white", xaxis_title="", yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.caption("Need ≥63 days for rolling vol.")
if roll_beta_path.exists():
    df_beta = pd.read_csv(roll_beta_path, parse_dates=["date"])
    df_beta = df_beta[df_beta["date"] >= "2013-06-05"]
    df_beta = filter_by_period(df_beta, "date")
    if len(df_beta) > 0:
        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(x=df_beta["date"], y=df_beta["beta_mkt"], mode="lines", name="Rolling Beta (36M)", line=dict(color="#00d4ff", width=1.5)))
        fig_beta.add_hline(y=1.0, line_dash="dash", line_color="#999", annotation_text="β=1")
        fig_beta.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white", xaxis_title="", yaxis_title="Market Beta")
        st.plotly_chart(fig_beta, use_container_width=True)
else:
    st.caption("Run factor regression for rolling beta.")

# 7. Tail correlation
st.markdown("---")
st.markdown("**Tail Correlation (Block1 vs Block2)**")
tail_path = exp / "stress_tail_corr.csv"
if tail_path.exists():
    df = pd.read_csv(tail_path)
    port_row = df[df["condition"] == "port_worst_5pct"]
    if len(port_row) > 0:
        corr = float(port_row["tail_corr_b1_b2"].iloc[0])
        st.metric("Port worst 5% tail correlation", f"{corr:.3f}", help="Correlation in worst days")

# 8. Backtest cumulative return
st.markdown("---")
st.markdown("**Backtest Performance**")

port_path = exp / "true_daily_block1.csv"
b2_path = exp / "block2_hmm_expanding_rebalonly.csv"
if not b2_path.exists():
    b2_path = exp / "true_daily_block2.csv"

if port_path.exists() and b2_path.exists():
    b1 = pd.read_csv(port_path, parse_dates=["date"]).set_index("date")["block1"].squeeze()
    b2 = pd.read_csv(b2_path, parse_dates=["date"]).set_index("date").iloc[:, 0].squeeze()
    common = b1.index.intersection(b2.index)
    r_p = 0.3 * b1.reindex(common).ffill().bfill().fillna(0) + 0.7 * b2.reindex(common).ffill().bfill().fillna(0)
    r_p = r_p.dropna()
    r_p = filter_series_by_period(r_p)
    if len(r_p) > 0:
        cum = (1 + r_p).cumprod()
        cum_pct = ((cum / cum.iloc[0]) - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum_pct.values, mode="lines", name="Portfolio", line=dict(color="#00d4ff", width=2), fill="tozeroy", fillcolor="rgba(0,212,255,0.1)"))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white", xaxis_title="", yaxis_title="Cumulative Return (%)", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No data for selected period.")
else:
    st.info("Run pipeline for backtest data.")
