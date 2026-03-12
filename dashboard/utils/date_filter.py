"""Shared date range filter for charts."""

from datetime import timedelta

import pandas as pd
import streamlit as st


def get_chart_period():
    """Return selected period: '1Y'|'3Y'|'5Y'|'All'. Renders sidebar widget."""
    period = st.sidebar.radio(
        "Chart period",
        ["1Y", "3Y", "5Y", "All"],
        key="chart_period",
        index=1,  # 3Y default
        horizontal=True,
    )
    return period


def filter_by_period(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """Filter dataframe to selected period. df must have DatetimeIndex or date_col."""
    period = st.session_state.get("chart_period", "3Y") if "chart_period" in st.session_state else "3Y"
    if period == "All":
        return df
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
    else:
        dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    end = dates.max()
    if period == "1Y":
        start = end - timedelta(days=365)
    elif period == "3Y":
        start = end - timedelta(days=365 * 3)
    elif period == "5Y":
        start = end - timedelta(days=365 * 5)
    else:
        return df
    mask = (dates >= start) & (dates <= end)
    if date_col:
        return df[mask].copy()
    return df.loc[mask].copy()


def filter_series_by_period(s: pd.Series) -> pd.Series:
    """Filter Series with DatetimeIndex by selected period."""
    period = st.session_state.get("chart_period", "3Y") if "chart_period" in st.session_state else "3Y"
    if period == "All":
        return s
    end = s.index.max()
    if period == "1Y":
        start = end - timedelta(days=365)
    elif period == "3Y":
        start = end - timedelta(days=365 * 3)
    elif period == "5Y":
        start = end - timedelta(days=365 * 5)
    else:
        return s
    return s[(s.index >= start) & (s.index <= end)]
