# Project File Structure

```
Automatic investing program/
│
├── PROJECT_REPORT.md          # Main portfolio report
├── README.md
├── FILE_STRUCTURE.md          # This file
├── requirements.txt
├── .env                        # FRED_API_KEY (git-ignored)
├── .gitignore
├── app.py                      # Streamlit dashboard (stub)
│
├── src/                        # Core modules
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py          # Phase 1: FRED + yfinance
│   ├── feature_engineer.py     # Phase 2: Feature engineering
│   ├── strategy_analyzer.py    # Phase 3: IC, HMM, feature selection
│   ├── model_trainer.py       # Phase 4: XGBoost walk-forward
│   └── sentiment_analyzer.py
│
├── data/                       # Raw and processed data (git-ignored)
│   ├── raw_data.csv
│   ├── raw_data_extended_2005.csv
│   ├── processed_features.csv
│   └── processed_features_extended_2005.csv
│
├── outputs/                    # Production outputs (git-ignored)
│   ├── model.pkl               # Trained XGBoost model
│   ├── backtest_report.md
│   ├── hmm_regime.csv
│   ├── selected_features.json
│   ├── final_risk_governance_report.csv
│   ├── PRODUCTION_AUDIT.md
│   └── audit/
│       ├── universe_availability_report.csv
│       ├── target_alignment_audit.md
│       └── ...
│
├── experiments/
│   ├── README.md
│   ├── data/                    # Experiment data copy
│   │   ├── raw_data_extended_2005.csv
│   │   ├── processed_features_extended_2005.csv
│   │   └── ...
│   │
│   ├── scripts/                 # Pipeline and experiment scripts
│   │   ├── true_daily_returns.py           # [1] Block1, Block2 daily returns
│   │   ├── block2_hmm_expanding_variants.py # [2] Block2 HMM variant
│   │   ├── factor_regression.py            # [3] Factor exposure
│   │   ├── factor_regression_validation.py # [4] Risk audit
│   │   ├── stress_test.py                  # [5] Stress test
│   │   ├── ensemble_weight_sweep.py
│   │   └── ... (other validation and experiment scripts)
│   │
│   └── outputs/                 # Main deliverables
│       ├── true_daily_block1.csv            # Block1 daily returns
│       ├── true_daily_block2.csv
│       ├── true_daily_portfolio.csv         # 30/70 ensemble
│       ├── block2_hmm_expanding_rebalonly.csv
│       ├── stress_test_report.md
│       ├── stress_*.csv
│       ├── factor_regression_validation_report.md
│       ├── factor_regression_*.csv
│       ├── ensemble_weight_sweep_report.md
│       ├── true_daily_metrics_report.md
│       │
│       └── archive/              # Intermediate experiment outputs (archive)
│           ├── README.md
│           └── ... (83 files)
│
├── dev_logs/                    # Development logs (git-ignored)
│   ├── README.md
│   ├── dev_log.md               # Development flow and decisions
│   └── experimental_details.md  # Experimental details (content moved from Section 3)
│
├── scripts/                     # Utility scripts
│   ├── run_governance_audit.py
│   ├── validation_audit.py
│   └── robustness_oos_evaluation.py
│
└── tests/
    ├── __init__.py
    └── verify_advanced_logic.py # Phase 5-3 self-verification
```

---

## Main Pipeline Execution Order

| Step | Script | Output |
|------|--------|--------|
| 1 | `true_daily_returns.py` | true_daily_block1.csv, true_daily_block2.csv |
| 2 | `block2_hmm_expanding_variants.py` | block2_hmm_expanding_rebalonly.csv |
| 3 | `factor_regression.py` | factor exposure (SPY/VIX required) |
| 4 | `factor_regression_validation.py` | factor_regression_validation_report.md |
| 5 | `stress_test.py` | stress_test_report.md |
