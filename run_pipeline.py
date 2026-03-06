#!/usr/bin/env python3
"""
Run full pipeline: Block1+Block2 → HMM variant → Factor regression → Validation → Stress test.

Two modes:
  - repro (default): Research/Backtest — uses only fixed files in data/, no API calls.
  - refresh: Live/Operations — fetches new data, runs full pipeline, writes to *_refresh/ dirs.

Usage:
    python run_pipeline.py                    # Repro mode (default)
    python run_pipeline.py --mode repro       # Explicit repro
    python run_pipeline.py --mode refresh     # Refresh mode (fetches data, separate output dirs)
    python run_pipeline.py --yes              # Skip prerequisite prompt (repro only)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REPRO_STEPS = [
    ("True Daily Returns (Block1 + Block2)", "experiments/scripts/true_daily_returns.py"),
    ("Block2 HMM Variant", "experiments/scripts/block2_hmm_expanding_variants.py"),
    ("Factor Regression", "experiments/scripts/factor_regression.py"),
    ("Factor Regression Validation", "experiments/scripts/factor_regression_validation.py"),
    ("Stress Test", "experiments/scripts/stress_test.py"),
]

REFRESH_PREP_STEPS = [
    ("Data Loader (FRED + yfinance)", "src.data_loader"),
    ("Feature Engineer", "src.feature_engineer"),
    ("Strategy Analyzer", "src.strategy_analyzer"),
]


def check_prerequisites_repro() -> bool:
    """Check if required data exists for repro mode."""
    data_raw = ROOT / "data" / "raw_data_extended_2005.csv"
    data_proc = ROOT / "data" / "processed_features_extended_2005.csv"
    exp_raw = ROOT / "experiments" / "data" / "raw_data_extended_2005.csv"
    selected = ROOT / "outputs" / "selected_features.json"

    if data_raw.exists() or exp_raw.exists():
        if data_proc.exists() or (ROOT / "experiments" / "data" / "processed_features_extended_2005.csv").exists():
            if selected.exists():
                return True
            print("[WARN] outputs/selected_features.json not found.")
        else:
            print("[WARN] processed_features not found.")
    else:
        print("[WARN] raw_data not found.")
    return False


def run_repro(args) -> int:
    """Repro mode: use only data/ files, no API. Writes to outputs/, experiments/outputs/."""
    print("=" * 60)
    print("Macro-Quant Sector Rotation — REPRO MODE (Research/Backtest)")
    print("  - Uses only fixed files in data/")
    print("  - No external API calls")
    print("=" * 60)

    env = os.environ.copy()
    env.pop("PIPELINE_REFRESH_MODE", None)  # Ensure repro

    if not check_prerequisites_repro():
        print("\n[!] Prerequisites may be missing. Pipeline may fail.")
        print("    Ensure: data/, outputs/selected_features.json exist.")
        if not args.yes:
            resp = input("    Continue anyway? [y/N]: ").strip().lower()
            if resp != "y":
                return 1

    for i, (name, script) in enumerate(REPRO_STEPS, 1):
        path = ROOT / script
        if not path.exists():
            print(f"\n[ERROR] Script not found: {path}")
            return 1
        print(f"\n--- Step {i}/{len(REPRO_STEPS)}: {name} ---")
        ret = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            env=env,
        )
        if ret.returncode != 0:
            print(f"\n[ERROR] Step {i} failed (exit {ret.returncode})")
            return ret.returncode

    print("\n" + "=" * 60)
    print("Repro pipeline completed successfully.")
    print("=" * 60)
    return 0


def run_refresh(args) -> int:
    """Refresh mode: fetch new data, run pipeline. Writes to *_refresh/ dirs only."""
    print("=" * 60)
    print("Macro-Quant Sector Rotation — REFRESH MODE (Live/Operations)")
    print("  - Fetches new data (FRED, yfinance)")
    print("  - Writes to data_refresh/, outputs_refresh/, experiments/outputs_refresh/")
    print("  - Does NOT overwrite repro results")
    print("=" * 60)

    env = os.environ.copy()
    env["PIPELINE_REFRESH_MODE"] = "1"

    # Prep: data_loader → feature_engineer → strategy_analyzer
    for i, (name, module) in enumerate(REFRESH_PREP_STEPS, 1):
        print(f"\n--- Prep {i}/{len(REFRESH_PREP_STEPS)}: {name} ---")
        ret = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(ROOT),
            env=env,
        )
        if ret.returncode != 0:
            print(f"\n[ERROR] Prep step {i} failed (exit {ret.returncode})")
            return ret.returncode

    # Main pipeline (with REFRESH_MODE so scripts write to *_refresh/)
    for i, (name, script) in enumerate(REPRO_STEPS, 1):
        path = ROOT / script
        if not path.exists():
            print(f"\n[ERROR] Script not found: {path}")
            return 1
        print(f"\n--- Step {i}/{len(REPRO_STEPS)}: {name} ---")
        ret = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            env=env,
        )
        if ret.returncode != 0:
            print(f"\n[ERROR] Step {i} failed (exit {ret.returncode})")
            return ret.returncode

    print("\n" + "=" * 60)
    print("Refresh pipeline completed successfully.")
    print("  Outputs: data_refresh/, outputs_refresh/, experiments/outputs_refresh/")
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument(
        "--mode",
        choices=["repro", "refresh"],
        default="repro",
        help="repro (default): fixed data, no API. refresh: fetch new data, separate output dirs.",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Proceed without prompting (repro only)")
    args = parser.parse_args()

    if args.mode == "refresh":
        return run_refresh(args)
    return run_repro(args)


if __name__ == "__main__":
    sys.exit(main())
