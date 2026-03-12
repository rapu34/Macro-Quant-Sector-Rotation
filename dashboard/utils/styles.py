"""Minimal dashboard styles."""

SHARED_CSS = """
<style>
.main .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

/* Section headers (목차) — larger, bolder */
.main .stMarkdown strong { font-size: 1.2rem !important; font-weight: 600 !important; }
.main .stMarkdown p { font-size: 1.05rem !important; }

/* Sidebar title */
section[data-testid="stSidebar"] h1 { font-size: 1.4rem !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] a { font-size: 1rem !important; }

/* Metric labels — slightly larger */
[data-testid="stMetricLabel"] label, [data-testid="stMetricLabel"] div { font-size: 0.95rem !important; }
[data-testid="stMetricValue"] { font-size: 1.6rem !important; }

/* Caption */
[data-testid="stCaptionContainer"] p { font-size: 0.9rem !important; }
</style>
"""
