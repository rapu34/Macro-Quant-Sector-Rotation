"""
Quant Risk Dashboard — Overview.
Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.utils.state_loader import load_state, get_data_mode
from dashboard.utils.styles import SHARED_CSS
from dashboard.utils.metrics import colored_metric, status_color, return_color, drawdown_color
from dashboard.utils.intraday import is_market_open, fetch_intraday_spy, compute_today_cumulative_return
from dashboard.utils.holdings import enrich_holdings_with_current

st.set_page_config(page_title="Quant Risk Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.sidebar.title("Quant Risk Dashboard")
st.sidebar.markdown("---")
from dashboard.utils.date_filter import get_chart_period
get_chart_period()  # Period selector for Performance, Risk, Strategy charts

state = load_state()
data_mode = get_data_mode()
mode_label = "LIVE" if data_mode == "refresh" else "BACKTEST"
asof = state.get("asof_date") or state.get("last_updated") or "—"
last_run = state.get("last_updated")

def _fmt(v, pct=True):
    if v is None: return "—"
    if pct and abs(v) < 10: return f"{v*100:.2f}%"
    return f"{v:.2f}"

caption = f"Data through: {asof}  ·  Mode: {mode_label}"
if last_run:
    caption += f"  ·  Pipeline: {last_run}"
st.caption(caption)

# Risk status row — color by status
p_crisis = state.get("p_crisis") or 0
if p_crisis < 0.4:
    status, status_key = "Normal", "pos"
elif p_crisis < 0.6:
    status, status_key = "Elevated", "warn"
else:
    status, status_key = "Crisis", "neg"

dd = state.get("current_drawdown")
dd_str = f"{dd*100:.2f}%" if dd is not None else "—"
regime = state.get("regime") or "—"
risk_mult = state.get("risk_mult")
rm_str = f"{risk_mult:.2f}" if risk_mult is not None else "—"

h1, h2, h3, h4, h5 = st.columns(5)
with h1:
    colored_metric("Risk Status", status, status_key)
with h2:
    colored_metric("Regime", regime, "neutral")
with h3:
    colored_metric("p_crisis", f"{p_crisis:.3f}", status_color(p_crisis))
with h4:
    colored_metric("risk_mult", rm_str, "neutral")
with h5:
    colored_metric("Drawdown", dd_str, drawdown_color(dd))

# Principal / Current Value
principal = state.get("principal")
principal_currency = state.get("principal_currency", "SGD")
principal_start = state.get("principal_start", "2025-12-15")
current_value = state.get("current_value")
cum_ret = state.get("cum_ret_since_start")
if principal is not None and current_value is not None:
    st.markdown("---")
    st.markdown("**Portfolio Value**")
    p1, p2, p3 = st.columns(3)
    with p1:
        colored_metric("Principal", f"{principal:,.0f} {principal_currency}", "neutral")
    with p2:
        colored_metric("Current Value", f"{current_value:,.2f} {principal_currency}", return_color(cum_ret) if cum_ret is not None else "neutral")
    with p3:
        ret_str = f"{cum_ret*100:.2f}%" if cum_ret is not None else "—"
        colored_metric("Return (since start)", ret_str, return_color(cum_ret) if cum_ret is not None else "neutral")
    st.caption(f"Start date: {principal_start}")

# Executive metrics row — Beta, Volatility, Tail Risk, Turnover
st.markdown("---")
st.markdown("**Executive Metrics**")
e1, e2, e3, e4 = st.columns(4)
with e1:
    beta = state.get("beta")
    colored_metric("Portfolio Beta", _fmt(beta, pct=False) if beta is not None else "—", "neutral")
with e2:
    vol = state.get("rolling_vol_63d")
    colored_metric("Portfolio Volatility", _fmt(vol), "neutral")
with e3:
    cvar = state.get("cvar_95")
    colored_metric("Expected Tail Loss (CVaR 95%)", _fmt(cvar) if cvar else "—", "neg" if cvar and cvar < -0.02 else "neutral")
with e4:
    to = state.get("last_rebalance_turnover") or {}
    to_comb = to.get("combined_pct")
    to_str = f"{to_comb}%" if to_comb is not None else "—"
    colored_metric("Last Rebalance Turnover", to_str, "neutral")

# Executive Alerts (Core only)
st.markdown("**Executive Alerts**")
core_alerts = state.get("core_alerts") or []
if core_alerts:
    for a in core_alerts:
        sev = a.get("severity", "CRITICAL")
        msg = a.get("message", "—")
        header = f"**{sev}** — "
        if sev == "CRITICAL":
            st.error(header + msg)
        else:
            st.warning(header + msg)
else:
    st.success("No active alerts")

# Today's P&L (refreshes every 5 min during market hours)
st.markdown("**Today's P&L**")
df_raw = fetch_intraday_spy()
cum_df = compute_today_cumulative_return(df_raw)
if cum_df is not None and len(cum_df) > 1:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_df.iloc[:, 0], y=cum_df["cum_return"] * 100,
        mode="lines+markers", name="SPY (proxy)", line=dict(color="#2ecc71", width=2), marker=dict(size=4)
    ))
    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_white", xaxis_title="", yaxis_title="Cumulative Return (%)",
        yaxis_tickformat=".2f", xaxis_tickformat="%H:%M"
    )
    st.plotly_chart(fig, use_container_width=True)
    caption = "SPY intraday (5m). Portfolio beta ~0.3 — actual P&L may differ."
    if is_market_open():
        caption += " Refreshes every 5 min."
    else:
        caption += " Market closed."
    st.caption(caption)
    if is_market_open():
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=5 * 60 * 1000, key="intraday")  # 5 min
        except ImportError:
            pass
elif is_market_open():
    st.info("Fetching intraday data...")
else:
    st.caption("Market closed. Intraday chart available during US session (9:30–16:00 ET).")

# Current Holdings (Block1+Block2 30:70)
raw_holdings = state.get("current_holdings") or []
holdings = enrich_holdings_with_current(raw_holdings, current_value) if raw_holdings and current_value else []
if holdings:
    st.markdown("**Current Holdings** (Block1+Block2 30:70)")
    import pandas as pd
    h_df = pd.DataFrame(holdings)
    h_df = h_df.rename(columns={
        "ticker": "ETF", "weight": "Weight %", "purchase_price": "Purchase", "current_price": "Current",
        "invested": "Invested", "current_value": "Current Value", "pnl_amt": "P&L", "pnl_pct": "P&L %"
    })
    cols = ["ETF", "Weight %", "Purchase", "Current", "Invested", "Current Value", "P&L", "P&L %"]
    cols = [c for c in cols if c in h_df.columns]
    st.dataframe(
        h_df[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ETF": st.column_config.TextColumn("ETF", width="small"),
            "Weight %": st.column_config.NumberColumn("Weight %", format="%.1f", width="small"),
            "Purchase": st.column_config.NumberColumn("Purchase", format="%.2f", width="small"),
            "Current": st.column_config.NumberColumn("Current", format="%.2f", width="small"),
            "Invested": st.column_config.NumberColumn("Invested", format="%.2f", width="small"),
            "Current Value": st.column_config.NumberColumn("Current Value", format="%.2f", width="small"),
            "P&L": st.column_config.NumberColumn("P&L", format="%.2f", width="small"),
            "P&L %": st.column_config.NumberColumn("P&L %", format="%.2f", width="small"),
        },
    )
    to = state.get("last_rebalance_turnover") or {}
    to_b1 = to.get("block1_pct")
    to_b2 = to.get("block2_pct")
    to_comb = to.get("combined_pct")
    to_str = []
    if to_b1 is not None:
        to_str.append(f"Block1: {to_b1}%")
    if to_b2 is not None:
        to_str.append(f"Block2: {to_b2}%")
    if to_comb is not None:
        to_str.append(f"Combined: {to_comb}%")
    cap = "Block1+Block2 30:70. 21-day rebalance. Purchase = rebalance date close. Current = live/delayed."
    if to_str:
        cap += f" Last rebalance turnover: {' | '.join(to_str)}. Block1: turnover cap (>25%→50%) applied."
    st.caption(cap)

# Monitoring Alerts
monitoring_alerts = state.get("monitoring_alerts") or []
if monitoring_alerts:
    st.markdown("**Monitoring**")
    for a in monitoring_alerts:
        sev = a.get("severity", "WARNING")
        msg = a.get("message", "—")
        header = f"**{sev}** — "
        if sev == "WARNING":
            st.warning(header + msg)
        else:
            st.info(header + msg)

# Alert History
alert_history = state.get("alert_history") or []
if alert_history:
    st.markdown("**Alert History**")
    import pandas as pd
    ah_df = pd.DataFrame(alert_history)
    st.dataframe(ah_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("**Performance & Risk Metrics**")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    v = state.get("1d_return")
    colored_metric("1d Return", _fmt(v), return_color(v))
with c2:
    v = state.get("1w_return")
    colored_metric("1w Return", _fmt(v), return_color(v))
with c3:
    v = state.get("1m_return")
    colored_metric("1m Return", _fmt(v), return_color(v))
with c4:
    colored_metric("Rolling Vol (63d)", _fmt(state.get("rolling_vol_63d")), "neutral")
with c5:
    sh = state.get("rolling_sharpe_252d")
    colored_metric("Rolling Sharpe (252d)", _fmt(sh, pct=False), "pos" if sh and sh > 0 else "neg" if sh and sh < 0 else "neutral")
with c6:
    v = state.get("cvar_95")
    colored_metric("CVaR (95%)", _fmt(v) if v else "—", "neg" if v and v < -0.02 else "neutral")

live_start = state.get("live_start") or "2025-12-15"
st.caption(f"Live portfolio monitoring start: {live_start} · Intraday values reflect mark-to-market portfolio valuation using delayed market data.")
