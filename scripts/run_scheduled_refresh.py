#!/usr/bin/env python3
"""
Scheduled refresh: run pipeline every 21 trading days (Block2 rebalance cycle).

Use with cron to automate. The script checks if 21 trading days have passed since
last run; if so, runs `python run_pipeline.py --mode refresh`.

Usage:
    python scripts/run_scheduled_refresh.py

Cron example (run daily at 6:00 AM; script skips if not rebalance day):
    0 6 * * * cd /path/to/project && python scripts/run_scheduled_refresh.py >> logs/scheduled.log 2>&1
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAST_RUN_FILE = ROOT / "outputs_refresh" / ".last_scheduled_run"
REBALANCE_DAYS = 21  # Block2 cycle


def _trading_days_since(last: pd.Timestamp, today: pd.Timestamp) -> int:
    """Count trading days (Mon–Fri) from day after last run through today."""
    if last >= today:
        return 0
    bdays = pd.bdate_range(start=last + pd.Timedelta(days=1), end=today)
    return len(bdays)


def should_run() -> bool:
    """True if 21+ trading days since last run, or first run."""
    if not LAST_RUN_FILE.exists():
        return True
    try:
        last_str = LAST_RUN_FILE.read_text().strip()
        last = pd.Timestamp(last_str)
        today = pd.Timestamp(datetime.now().date())
        days = _trading_days_since(last, today)
        return days >= REBALANCE_DAYS
    except Exception:
        return True


def mark_run():
    """Record today as last run date."""
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(datetime.now().strftime("%Y-%m-%d"))


def main() -> int:
    if not should_run():
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] Skip: not yet 21 trading days since last run")
        return 0

    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Running scheduled refresh (21-day cycle)...")
    env = os.environ.copy()
    env["PIPELINE_REFRESH_MODE"] = "1"

    ret = subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py"), "--mode", "refresh"],
        cwd=str(ROOT),
        env=env,
    )

    if ret.returncode == 0:
        mark_run()
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] Done. Next run in ~21 trading days.")
    else:
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] Pipeline failed (exit {ret.returncode})")
    return ret.returncode


if __name__ == "__main__":
    sys.exit(main())
