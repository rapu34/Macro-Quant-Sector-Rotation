"""Operations — System status, Run history."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.utils.state_loader import load_state, get_data_mode, get_run_history_path
from dashboard.utils.styles import SHARED_CSS

st.set_page_config(page_title="Operations", layout="wide", page_icon="⚙️")
st.markdown(SHARED_CSS, unsafe_allow_html=True)
state = load_state()
data_mode = get_data_mode()

st.markdown("**Pipeline Status**")
c1, c2, c3 = st.columns(3)
with c1:
    mode_label = "Refresh (Live)" if data_mode == "refresh" else "Repro (Backtest)"
    st.metric("Mode", mode_label, help="Refresh = Live outputs; Repro = Frozen backtest outputs")
with c2:
    st.metric("Last updated", state.get("last_updated") or "—")
with c3:
    st.metric("Data as of", state.get("asof_date") or "—")

st.markdown("---")
st.markdown("**Run History**")

run_path = get_run_history_path()
if run_path.exists():
    import pandas as pd
    df = pd.read_csv(run_path)
    df = df.sort_values("timestamp", ascending=False).head(30)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No run history. Run pipeline with write_dashboard_state to populate.")

st.caption("Run: python run_pipeline.py --mode refresh (for Live) or python run_pipeline.py (for Repro)")
