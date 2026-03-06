# Macro-Quant Sector Rotation

Combine macroeconomic indicators with US sector ETF prices to run an ensemble of **Block1 (XGBoost sector rotation)** and **Block2 (12M-1M momentum)** with HMM-based crisis regime detection.

**Main document:** [PROJECT_REPORT.md](PROJECT_REPORT.md) — full pipeline, performance, governance, deliverables.

**File structure:** [FILE_STRUCTURE.md](FILE_STRUCTURE.md) — detailed tree view.

---

## Repo structure

```
.
├── PROJECT_REPORT.md       # Main portfolio report (performance, architecture, governance)
├── run_pipeline.py         # One-command full pipeline
├── FILE_STRUCTURE.md       # Detailed file tree
├── README.md
├── requirements.txt
├── .env                     # FRED_API_KEY (git-ignored)
├── .gitignore
│
├── src/                     # Core modules
│   ├── data_loader.py       # Phase 1: FRED + yfinance
│   ├── feature_engineer.py   # Phase 2: Feature engineering
│   ├── strategy_analyzer.py # Phase 3: IC, feature selection, HMM
│   └── model_trainer.py     # Phase 4: XGBoost walk-forward
│
├── data/                    # Raw and processed data (git-ignored)
│   ├── raw_data_extended_2005.csv
│   └── processed_features_extended_2005.csv
│
├── outputs/                 # Production outputs (git-ignored)
│   ├── model.pkl
│   ├── backtest_report.md
│   ├── hmm_regime.csv
│   ├── selected_features.json
│   ├── final_risk_governance_report.csv
│   └── audit/
│
├── experiments/
│   ├── scripts/             # Pipeline and experiment scripts
│   │   ├── true_daily_returns.py
│   │   ├── block2_hmm_expanding_variants.py
│   │   ├── factor_regression.py
│   │   ├── factor_regression_validation.py
│   │   └── stress_test.py
│   └── outputs/             # Main deliverables
│       ├── true_daily_block1.csv
│       ├── block2_hmm_expanding_rebalonly.csv
│       ├── stress_test_report.md
│       ├── factor_regression_validation_report.md
│       └── archive/          # Intermediate experiment outputs
│
├── scripts/                 # Utility scripts
│   ├── run_scheduled_refresh.py   # 21-day rebalance automation
│   ├── run_governance_audit.py
│   └── ...
│
├── dev_logs/                # Internal logs (git-ignored)
│   ├── dev_log.md
│   └── experimental_details.md
│
├── logs/                    # Scheduled run logs (git-ignored)
│
└── tests/
    └── verify_advanced_logic.py
```

---

## Main pipeline (execution order)

**Repro mode (default)** — Research/Backtest, fixed data, no API:
```bash
python run_pipeline.py
```

**Refresh mode** — Live/Operations, fetches new data, writes to `*_refresh/` dirs:
```bash
python run_pipeline.py --mode refresh
```

Or run steps individually:
1. `python experiments/scripts/true_daily_returns.py` → block1, block2 CSV
2. `python experiments/scripts/block2_hmm_expanding_variants.py` → block2_hmm_expanding_rebalonly.csv
3. `python experiments/scripts/factor_regression.py` → factor exposure (SPY/VIX required)
4. `python experiments/scripts/factor_regression_validation.py` → risk audit
5. `python experiments/scripts/stress_test.py` → stress_test_report.md

---

## Scheduled automation (21-day rebalance cycle)

Run refresh automatically every 21 trading days (Block2 cycle):

```bash
python scripts/run_scheduled_refresh.py
```

**Cron** — run daily at 6:00 AM; script runs pipeline only when 21 days have passed:

```bash
# Edit crontab: crontab -e
0 6 * * * cd /path/to/Automatic\ investing\ program && python scripts/run_scheduled_refresh.py >> logs/scheduled.log 2>&1
```

Create `logs/` first: `mkdir -p logs`. Last run date is stored in `outputs_refresh/.last_scheduled_run`.

---

## Setup

```bash
pip install -r requirements.txt
```

Set `FRED_API_KEY` in a `.env` file at the project root.

---

## Self-Verification

```bash
python tests/verify_advanced_logic.py
```

- HMM Labeling Check, BIC Logic Check, Turnover Control Check, Block Bootstrap Check
