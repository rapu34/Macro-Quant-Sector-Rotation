"""Colored metric helpers — green/yellow/red status indication."""

import streamlit as st

# Status colors
GREEN = "#16a34a"   # positive, normal
YELLOW = "#ca8a04"  # warning, elevated
RED = "#dc2626"     # negative, crisis
NEUTRAL = "#374151"  # default


def colored_metric(label: str, value: str, status: str = "neutral"):
    """Render a metric with color-coded value. status: 'pos'|'neg'|'warn'|'neutral'."""
    color = {"pos": GREEN, "neg": RED, "warn": YELLOW, "neutral": NEUTRAL}.get(status, NEUTRAL)
    st.markdown(
        f'<div style="font-size:0.95rem;color:#6b7280;margin-bottom:4px;font-weight:500">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:600;color:{color}">{value}</div>',
        unsafe_allow_html=True,
    )


def status_color(p_crisis: float) -> str:
    """Return status key from p_crisis."""
    if p_crisis < 0.4:
        return "pos"
    if p_crisis < 0.6:
        return "warn"
    return "neg"


def return_color(v: float | None) -> str:
    """Return status key from return value (positive=green, negative=red)."""
    if v is None:
        return "neutral"
    return "pos" if v >= 0 else "neg"


def drawdown_color(dd: float | None) -> str:
    """Return status key from drawdown (more negative = worse)."""
    if dd is None:
        return "neutral"
    if dd <= -0.05:
        return "neg"
    if dd <= -0.02:
        return "warn"
    return "pos"
