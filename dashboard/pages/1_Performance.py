"""Performance — Portfolio vs SPY, Excess Return, Rolling Sharpe, Drawdown."""

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
from dashboard.utils.date_filter import get_chart_period, filter_by_period, filter_series_by_period

st.set_page_config(page_title="Performance", layout="wide", page_icon="📈")
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


# Benchmark metrics row
st.markdown("**Benchmark Comparison (vs SPY)**")
bench_path = exp / "benchmark_metrics.json"
if not bench_path.exists():
    bench_path = ROOT / "experiments" / "outputs" / "benchmark_metrics.json"
if bench_path.exists():
    import json
    with open(bench_path) as f:
        bm = json.load(f)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        colored_metric("Excess Return (Ann.)", _fmt(bm.get("excess_return_ann")), "pos" if bm.get("excess_return_ann", 0) > 0 else "neg")
    with c2:
        colored_metric("Information Ratio", _fmt(bm.get("information_ratio"), pct=False), "pos" if bm.get("information_ratio", 0) > 0 else "neutral")
    with c3:
        colored_metric("Tracking Error", _fmt(bm.get("tracking_error")), "neutral")
    with c4:
        colored_metric("Sortino Ratio", _fmt(bm.get("sortino_ratio"), pct=False), "pos" if bm.get("sortino_ratio", 0) > 0 else "neutral")
    with c5:
        colored_metric("Downside Deviation", _fmt(bm.get("downside_deviation")), "neutral")
else:
    st.info("Run pipeline (stress test) to populate benchmark comparison.")

# Portfolio vs SPY cumulative return
st.markdown("---")
st.markdown("**Portfolio vs SPY Cumulative Return**")
bench_daily = exp / "benchmark_daily.csv"
if not bench_daily.exists():
    bench_daily = ROOT / "experiments" / "outputs" / "benchmark_daily.csv"
if bench_daily.exists():
    df = pd.read_csv(bench_daily, parse_dates=["date"])
    post2013 = df[df["date"] >= "2013-06-05"] if len(df) > 0 else df
    if len(post2013) > 0:
        post2013 = filter_by_period(post2013, "date")
        if len(post2013) > 0:
            post2013 = post2013.copy()
            post2013["cum_port_pct"] = ((post2013["cum_port"] / post2013["cum_port"].iloc[0]) - 1) * 100
            post2013["cum_spy_pct"] = ((post2013["cum_spy"] / post2013["cum_spy"].iloc[0]) - 1) * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=post2013["date"], y=post2013["cum_port_pct"], mode="lines", name="Portfolio", line=dict(color="#00d4ff", width=2), fill="tozeroy", fillcolor="rgba(0,212,255,0.1)"))
            fig.add_trace(go.Scatter(x=post2013["date"], y=post2013["cum_spy_pct"], mode="lines", name="SPY", line=dict(color="#ffd93d", width=1.5), fill="tozeroy", fillcolor="rgba(255,217,61,0.05)"))
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white", xaxis_title="", yaxis_title="Cumulative Return (%)", legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No data for selected period.")
    else:
        st.caption("No post-2013 data.")
else:
    st.info("Run pipeline for benchmark chart.")

# Rolling Sharpe, Drawdown, Rolling Vol
st.markdown("---")
st.markdown("**Rolling Metrics**")
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
    post = r_p[r_p.index >= "2013-06-05"]
    post = filter_series_by_period(post)
    if len(post) >= 252:
        roll_sharpe = post.rolling(252).mean() / post.rolling(252).std() * (252**0.5)
        cum = (1 + post).cumprod()
        dd = (cum / cum.cummax()) - 1
        roll_vol = post.rolling(63).std() * (252**0.5) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, mode="lines", name="Rolling Sharpe (12M)", line=dict(color="#00ff9d", width=1.5)))
        fig.add_hline(y=0, line_dash="dash", line_color="#999")
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white", xaxis_title="", yaxis_title="Sharpe")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, mode="lines", name="Drawdown", line=dict(color="#ff6b6b", width=1.5), fill="tozeroy", fillcolor="rgba(255,107,107,0.2)"))
        fig2.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white", xaxis_title="", yaxis_title="Drawdown (%)")
        st.plotly_chart(fig2, use_container_width=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, mode="lines", name="Rolling Vol (63d)", line=dict(color="#ffd93d", width=1.5)))
        fig3.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white", xaxis_title="", yaxis_title="Vol (Ann. %)")
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Run pipeline for rolling metrics.")
