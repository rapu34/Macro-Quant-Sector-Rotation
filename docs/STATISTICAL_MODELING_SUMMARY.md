# Statistical Modeling Summary

> Summary of statistical modeling used in the final model pipeline (30% Block1 + 70% Block2_HMM_REBAL_ONLY).

---

## 1. Usage in Final Pipeline

| Statistical Model | Used in Final Model | Location | Role |
|------------------|---------------------|----------|------|
| **HMM (Gaussian)** | ✅ | strategy_analyzer, model_trainer, block2_hmm | Regime detection (Core/Crisis), risk_mult |
| **OLS (Factor Regression)** | ✅ | factor_regression.py | r_p ~ r_mkt + r_size + r_mom + r_tlt + ΔVIX (attribution) |
| **OLS (Sector Sensitivity)** | ✅ | strategy_analyzer | relative_ret_20d ~ macro (for feature selection) |
| **Spearman IC** | ✅ | strategy_analyzer | feature vs target correlation, p-value → selected_features |
| **Drift Test** | ✅ | strategy_analyzer | early vs late half IC comparison → feature exclusion |
| **t-test, Logistic Regression** | ❌ | validate_hmm_*.py | HMM validation scripts (not in pipeline) |

---

## 2. Details

### 2.1 HMM (Hidden Markov Model)

- **Location**: `src/strategy_analyzer.py` (GaussianHMM, hmmlearn)
- **Role**: 2-state (Core/Crisis) regime detection
- **Output**: p_crisis, hmm_regime.csv
- **Final model use**: Block1 risk_mult, Block2 risk_mult (clip(1 - p_crisis, 0.5, 1.0))
- **Pipeline**: REFRESH_PREP (strategy_analyzer) → model_trainer, block2_hmm_expanding_variants

### 2.2 OLS — Factor Regression

- **Location**: `experiments/scripts/factor_regression.py`
- **Model**: r_p = α + β_mkt·r_mkt + β_size·r_size + β_mom·r_mom + β_tlt·r_tlt + β_vol·ΔVIX + ε
- **Output**: factor_regression_summary.csv, benchmark_factor_data.csv
- **Final model use**: Portfolio attribution, beta estimation, dashboard display
- **Pipeline**: REPRO_STEPS Step 3

### 2.3 OLS — Sector Sensitivity

- **Location**: `src/strategy_analyzer.py` (run_sector_sensitivity)
- **Model**: relative_ret_20d ~ macro indicators (per sector)
- **Role**: Identify rate-shock strong, inflation-vulnerable sectors (strategy EDA)
- **Final model use**: Indirect contribution to selected_features.json (with IC/drift)
- **Pipeline**: REFRESH_PREP (strategy_analyzer)

### 2.4 Spearman IC (Information Coefficient)

- **Location**: `src/strategy_analyzer.py`
- **Role**: Spearman correlation of feature vs target (top-3 binary), p-value
- **Filter**: |IC| ≥ 0.02, p ≤ 0.05 → selected_features
- **Final model use**: Block1 XGBoost input feature selection
- **Pipeline**: REFRESH_PREP (strategy_analyzer)

### 2.5 Drift Test

- **Location**: `src/strategy_analyzer.py` (run_drift_test)
- **Role**: Early half vs late half IC comparison, feature stability
- **Filter**: IC sign flip, large drift → feature exclusion
- **Final model use**: Remove unstable features from selected_features
- **Pipeline**: REFRESH_PREP (strategy_analyzer)

---

## 3. Pipeline Flow (run_pipeline.py)

```
REFRESH_PREP (or pre-run):
  data_loader → feature_engineer → strategy_analyzer
    └─ IC, sector OLS, drift → selected_features.json
    └─ HMM → hmm_regime.csv, p_crisis

REPRO_STEPS:
  1. true_daily_returns  ← model_trainer (Block1), HMM
  2. block2_hmm_expanding_variants  ← HMM
  3. factor_regression  ← OLS (r_p ~ factors)
  4. factor_regression_validation
  5. stress_test
  6. benchmark_comparison
```

---

## 4. Conclusion

**Statistical modeling used in the final model:**

1. **HMM** — Regime detection, risk_mult
2. **OLS (Factor)** — Portfolio attribution (beta, alpha)
3. **OLS (Sector)** — Strategy EDA, feature selection support
4. **Spearman IC** — Feature selection
5. **Drift Test** — Feature stability, selection support

*t-test and logistic regression in validate_hmm_*.py are validation experiments not included in the pipeline.*
