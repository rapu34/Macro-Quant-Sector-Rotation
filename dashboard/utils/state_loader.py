"""Load dashboard state from pipeline outputs."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS_REFRESH = ROOT / "outputs_refresh"
EXP_OUT = ROOT / "experiments" / "outputs"
EXP_OUT_REFRESH = ROOT / "experiments" / "outputs_refresh"


def get_data_mode() -> str:
    """Return 'refresh' if refresh outputs exist, else 'repro'."""
    if (EXP_OUT_REFRESH / "true_daily_block1.csv").exists():
        return "refresh"
    return "repro"


def _get_state_dir() -> Path:
    if (OUTPUTS_REFRESH / "state" / "latest_state.json").exists():
        return OUTPUTS_REFRESH / "state"
    return OUTPUTS / "state"


def load_state() -> dict[str, Any]:
    """Load latest_state.json. Returns empty dict if missing."""
    state_file = _get_state_dir() / "latest_state.json"
    if not state_file.exists():
        return {}
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception:
        return {}


def get_exp_out() -> Path:
    """Return experiments/outputs or experiments/outputs_refresh."""
    if get_data_mode() == "refresh":
        return EXP_OUT_REFRESH
    return EXP_OUT


def get_run_history_path() -> Path:
    if (OUTPUTS_REFRESH / "logs" / "run_history.csv").exists():
        return OUTPUTS_REFRESH / "logs" / "run_history.csv"
    return OUTPUTS / "logs" / "run_history.csv"
