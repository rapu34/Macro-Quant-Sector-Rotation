"""Strategy Monitoring — Allocation, Regime history."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.state_loader import load_state, get_data_mode, get_exp_out
from dashboard.utils.styles import SHARED_CSS
from dashboard.utils.date_filter import get_chart_period, filter_by_period, filter_series_by_period

st.set_page_config(page_title="Strategy Monitoring", layout="wide", page_icon="📈")
get_chart_period()
st.markdown(SHARED_CSS, unsafe_allow_html=True)
state = load_state()
exp = get_exp_out()

st.markdown("**Current Allocation**")
st.metric("Block1 / Block2", "30% / 70%", help="Ensemble weights")
st.caption("Portfolio allocation combines macro-driven sector selection with momentum-based cross-sectional allocation.")
st.caption("Block1: XGBoost sector rotation (Top 3) · Block2: 12M-1M momentum (Top 3)")

st.markdown("---")
st.markdown("**Block1 vs Block2 Cumulative Return**")
port_path = exp / "true_daily_block1.csv"
b2_path = exp / "block2_hmm_expanding_rebalonly.csv"
if not b2_path.exists():
    b2_path = exp / "true_daily_block2.csv"
if port_path.exists() and b2_path.exists():
    b1 = pd.read_csv(port_path, parse_dates=["date"]).set_index("date")["block1"].squeeze()
    b2 = pd.read_csv(b2_path, parse_dates=["date"]).set_index("date").iloc[:, 0].squeeze()
    common = b1.index.intersection(b2.index)
    r_b1 = b1.reindex(common).ffill().bfill().fillna(0)
    r_b2 = b2.reindex(common).ffill().bfill().fillna(0)
    r_p = 0.3 * r_b1 + 0.7 * r_b2
    post = r_p[r_p.index >= "2013-06-05"]
    post = filter_series_by_period(post)
    if len(post) > 0:
        cum_b1 = (1 + r_b1.reindex(post.index).ffill().bfill().fillna(0)).cumprod()
        cum_b2 = (1 + r_b2.reindex(post.index).ffill().bfill().fillna(0)).cumprod()
        cum_p = (1 + post).cumprod()
        cum_b1_pct = ((cum_b1 / cum_b1.iloc[0]) - 1) * 100
        cum_b2_pct = ((cum_b2 / cum_b2.iloc[0]) - 1) * 100
        cum_p_pct = ((cum_p / cum_p.iloc[0]) - 1) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_p.index, y=cum_p_pct.values, mode="lines", name="Portfolio (30/70)", line=dict(color="#00d4ff", width=2)))
        fig.add_trace(go.Scatter(x=cum_b1.index, y=cum_b1_pct.values, mode="lines", name="Block1 (XGBoost)", line=dict(color="#00ff9d", width=1, dash="dash")))
        fig.add_trace(go.Scatter(x=cum_b2.index, y=cum_b2_pct.values, mode="lines", name="Block2 (HMM Mom)", line=dict(color="#ffd93d", width=1, dash="dash")))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white", xaxis_title="", yaxis_title="Cumulative Return (%)", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run pipeline for block comparison.")

st.markdown("---")
st.markdown("**Regime History (p_crisis)**")

regime_path = exp.parent.parent / "outputs" / "hmm_regime.csv"
if not regime_path.exists():
    regime_path = exp.parent.parent / "outputs_refresh" / "hmm_regime.csv"
if not regime_path.exists():
    regime_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "hmm_regime.csv"

if regime_path.exists():
    df = pd.read_csv(regime_path)
    if "date" in df.columns and "P_Crisis" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = filter_by_period(df, "date")
        if len(df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["P_Crisis"], mode="lines", name="p_crisis", line=dict(color="#00d4ff", width=1.5), fill="tozeroy", fillcolor="rgba(0,212,255,0.1)"))
            fig.add_hline(y=0.4, line_dash="dash", line_color="#ffd93d", annotation_text="Elevated")
            fig.add_hline(y=0.6, line_dash="dash", line_color="#ff6b6b", annotation_text="Crisis")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white", xaxis_title="", yaxis_title="p_crisis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No data for selected period.")
    else:
        st.dataframe(df.tail(50), use_container_width=True, hide_index=True)
else:
    st.info("Run pipeline for regime data.")

st.markdown("---")
st.markdown("**Top Macro Drivers (XGBoost Feature Importance)**")
model_path = exp.parent.parent / "outputs" / "model.pkl"
if not model_path.exists():
    model_path = exp.parent.parent / "outputs_refresh" / "model.pkl"
if model_path.exists():
    try:
        import joblib
        loaded = joblib.load(model_path)
        model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        if hasattr(model, "feature_importances_"):
            import json
            if isinstance(loaded, dict) and "features" in loaded:
                feats = loaded["features"]
            else:
                feat_path = exp.parent.parent / "outputs" / "selected_features.json"
                if not feat_path.exists():
                    feat_path = exp.parent.parent / "outputs_refresh" / "selected_features.json"
                if feat_path.exists():
                    with open(feat_path) as f:
                        j = json.load(f)
                    feats = j.get("selected_features", j) if isinstance(j, dict) else j
                else:
                    feats = [f"f{i}" for i in range(len(model.feature_importances_))]
            FEAT_LABELS = {
                "volatility_20d": "Volatility (20d)",
                "sentiment_dispersion": "Sentiment",
                "cpi_all_urban_zscore_lag20": "CPI",
                "real_rate_zscore_lag20": "Real Rate",
                "treasury_10y_zscore_lag20": "Treasury 10Y",
                "yield_curve_10y2y_zscore_lag20": "Yield Curve",
                "unemployment_rate": "Unemployment",
            }
            imp = pd.DataFrame({"feature": feats[:len(model.feature_importances_)], "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            imp["label"] = imp["feature"].map(lambda x: FEAT_LABELS.get(x, x)).fillna(imp["feature"])
            fig = go.Figure(go.Bar(x=imp["importance"], y=imp["label"], orientation="h", marker_color="#00d4ff"))
            fig.update_layout(height=min(400, 30 * len(imp)), margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption("Could not load feature importance.")
else:
    st.caption("Model not found.")
