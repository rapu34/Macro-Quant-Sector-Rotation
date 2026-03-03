# Macro-Quant Sector Rotation: Project Report

> **Portfolio / CV** — End-to-end documentation of an automated investment system from development through production-grade governance.

---

## 1. Project Overview

### 1.1 Objective

Build an automated investment pipeline that combines macroeconomic indicators and US sector ETF prices to run an ensemble of **Block1 (XGBoost sector rotation)** and **Block2 (12M-1M momentum)**, with **HMM-based crisis regime detection** for risk control.

### 1.2 Final Portfolio

| Item | Description |
|------|-------------|
| **Composition** | 30% Block1 + 70% Block2_HMM_REBAL_ONLY |
| **Block1** | XGBoost sector rotation (Top 3), HMM risk_mult |
| **Block2** | 12M-1M momentum (Top 3), HMM_REBAL_ONLY overlay |
| **Returns** | True Daily PnL (no smoothing, no target vol scaling) |
| **Period** | 2005-01-04 ~ 2026-02-26 (n=5,279) |

**Block1 warm-up**: Block1 remains inactive during the initial warm-up period (2005–2013) due to expanding window training requirements (MIN_TRAIN_PCT=40%). Full dual-block deployment begins post-2013.

### 1.3 Performance Summary

| Metric | Full Period (2005–2026) | Post-2013 |
|--------|-------------------------|-----------|
| **Annualized Return** | 5.50% | 8.17% |
| **Annualized Volatility** | 10.19% | 11.74% |
| **Sharpe Ratio** | 0.54 | 0.70 |
| **Max Drawdown** | -23.70% | -23.70% |
| **CVaR(95%)** (unconditional daily) | -1.59% | -1.80% |

*The maximum drawdown in both Full and Post-2013 periods is driven by the COVID 2020 event.*

*The strategy emphasizes regime-aware exposure stability and tail-risk efficiency over pure return maximization.*

### 1.4 Stress Test Highlights

*Note: CVaR in Performance Summary refers to unconditional daily distribution, while Stress Test CVaR represents the average loss of the worst 5% trading days.*

| Metric | Result |
|--------|--------|
| **Worst 1% avg loss** | -2.72% |
| **CVaR(95%)** (worst 5% avg) | -2.85% |
| **SPY ≤-3% days avg loss** | -1.98% (N=77) |
| **ΔVIX ≥+10% days avg loss** | -0.99% (N=437) |
| **Tail correlation** (Block1 vs Block2) | ~0.28 |
| **Market beta** (post-2013) | ~0.31 |
| **Alpha** | Not significant (t=0.15) |

### 1.5 Factor Exposure (OLS, post-2013)

*r_p = α + β_mkt·r_mkt + β_size·r_size + β_mom·r_mom + β_tlt·r_tlt + β_vol·ΔVIX + ε*

| Factor | Coefficient | t-stat | Interpretation |
|--------|-------------|--------|----------------|
| **alpha** | 0.00001 | 0.15 | Not significant |
| **r_mkt** (SPY) | 0.31 | 21.1 | Market beta &lt; 1; lower systematic exposure |
| **r_size** | 0.07 | 8.6 | Size tilt |
| **r_mom** | 0.19 | 19.8 | Momentum tilt |
| **r_tlt** (duration) | -0.01 | -2.4 | Rate sensitivity; falls when bonds rally |
| **ΔVIX** | -0.008 | -8.5 | Vol sensitivity; falls when VIX rises |

- **R²**: 0.85 | **Residual vol (ann.)**: 4.7%
- **Crisis betas**: 2020 β_mkt ≈ 0.48; 2022 β_mkt ≈ 0.23

*Overall, the portfolio exhibits controlled systematic exposure (β ≈ 0.3), mild volatility sensitivity, and no statistically significant standalone alpha, consistent with its risk-stabilization objective.*

### 1.6 Event Summary (2020/2022)

| Event | MDD | Recovery |
|-------|-----|----------|
| COVID 2020 | -23.70% | 109d (~5.2mo) |
| Rate Shock 2022 | -13.33% | 370d (~17.6mo) |

### 1.7 Technology Stack

- **Language**: Python 3.12
- **ML**: XGBoost (Block1), hmmlearn (Gaussian HMM)
- **Data**: FRED API (macro), yfinance (ETF prices)
- **Other**: pandas, numpy, scipy, matplotlib

---

## 2. Pipeline Architecture

```
[Phase 1] Data Loader
    └─ FRED + yfinance (9 sector ETFs)
    └─ raw_data_extended_2005.csv

[Phase 2] Feature Engineer
    └─ 20-day lag (look-ahead prevention), Z-Score, MoM/YoY, RSI, volatility, sentiment_dispersion
    └─ processed_features_extended_2005.csv

[Phase 3] Strategy Analyzer
    └─ IC, sector beta, drift test → selected_features.json
    └─ HMM 2-state (Core-Crisis) + get_p_crisis_expanding

[Phase 4] Block1 — Model Trainer
    └─ Walk-forward XGBoost (expanding window, MIN_TRAIN_PCT=40%)
    └─ HMM risk_mult = clip(1 - p_crisis, 0.5, 1.0)
    └─ model.pkl, true_daily_block1.csv

[Phase 5] Block2 — Momentum + HMM
    └─ 12M-1M momentum Top 3, 21-day rebalance
    └─ HMM_REBAL_ONLY: risk_mult at rebalance only
    └─ block2_hmm_expanding_rebalonly.csv

[Phase 6] Ensemble & Risk
    └─ 30/70 ensemble, Factor Regression, Stress Test
    └─ stress_test_report.md, factor_regression_validation_report.md
```

### 2.1 Design Principles

1. **Look-ahead prevention**: Macro 20-day lag; target = t → t+20d forward return
2. **Expanding HMM**: Regime estimated using only data up to each rebalance date
3. **Tail protection**: risk_mult = max(0.5, 1 - p_crisis); MDD/CVaR controlled
4. **True Daily PnL**: ETF price-based daily returns; no period smoothing

---

## 3. Key Design Trade-offs

This section summarizes the main design decisions and their implications. Detailed experimental logs (hysteresis tuning, turnover threshold tests, 5y vs 15y comparison, scaling leakage checks, statelessness debugging) are archived in `dev_logs/experimental_details.md`.

| Trade-off | Decision | Rationale |
|-----------|----------|-----------|
| **Crisis false positive vs tail protection** | Keep risk_mult_min=0.5 floor | Exposure floor design constrains mid-range elasticity by construction; prioritization placed on drawdown control over responsiveness |
| **Exposure elasticity** | Accept mid-range rigidity | Non-linear response (e.g. p_crisis^1.3) negated by same floor; trade-off accepted |
| **Data horizon (5y vs 15y)** | 2005–2026 for stress test; Block1 warm-up 2005–2013 | Regime-sensitive strategies can suffer from long-horizon noise; MIN_TRAIN_PCT=40% drives warm-up |
| **HMM scaling** | Expanding fit only on data up to asof_date | Verified no look-ahead; PASS |
| **Hysteresis state** | Stateful within expanding window | Verified sequential in_crisis maintenance; PASS |

---

## 4. Governance & Quality Assurance

### 4.1 Production Audit (5 Items)

| Item | Result |
|------|--------|
| A) HMM scaling leakage | PASS |
| B) Hysteresis statefulness | STATEFUL |
| C) Timing correctness (p_crisis → exposure → fwd_ret) | PASS |
| D) Reproducibility (seeds) | PASS |
| E) Tagged runs (--tag) | Implemented |

### 4.2 Governance Audit (3 Items)

| Item | Result |
|------|--------|
| 1) Universe & availability | PASS |
| 2) Target alignment sanity | PASS |
| 3) Transaction cost decomposition | PASS |

### 4.3 Project Structure

- **Production**: `src/`, `outputs/` (model.pkl, backtest_report.md, hmm_regime.csv, etc.)
- **Experiments**: `experiments/` (true_daily_returns, block2_hmm, stress_test, factor_regression, etc.)
- **Audit**: `outputs/audit/` (universe, target, cost validation)

---

## 5. Final Model Structure

| Component | Description |
|-----------|--------------|
| **Block1** | XGBoost sector rotation. Stored in model.pkl. 20-day rebalance |
| **Block2** | 12M-1M momentum (rule-based). 21-day rebalance. No XGBoost |
| **HMM** | Expanding fit. get_p_crisis_expanding. Not stored in model.pkl |
| **Features** | volatility_20d, sentiment_dispersion, cpi_all_urban_zscore_lag20, etc. |
| **Target** | 20-day forward relative return Top 3 = 1, else 0 |
| **Risk** | risk_mult = clip(1 - p_crisis, 0.5, 1.0), Target Vol 15%, Kelly Cap 0.25 |
| **Ensemble** | 30% Block1 + 70% Block2_HMM_REBAL_ONLY |

---

## 6. Lessons & Implications

1. **Look-ahead prevention from design**: Lag, target definition, and scaling scope must be strictly separated from the start.
2. **Tail protection vs return**: risk_mult_min floor is essential for crisis protection but limits mid-range elasticity.
3. **Data horizon choice**: For regime-sensitive strategies, longer data is not always better; 5-year outperformed 15-year in Block1.
4. **Governance first**: Scaling, hysteresis, timing, seeds, and cost validation ensure reproducibility and reliability before production.
5. **No alpha inflation**: Factor regression shows alpha not significant (t=0.15). Value lies in tail-risk efficiency and regime-aware exposure control.

---

## 7. Deliverables Summary

| Category | File / Path |
|----------|-------------|
| **Model** | outputs/model.pkl |
| **Block1 daily** | experiments/outputs/true_daily_block1.csv |
| **Block2 HMM** | experiments/outputs/block2_hmm_expanding_rebalonly.csv |
| **Report** | outputs/backtest_report.md |
| **Stress Test** | experiments/outputs/stress_test_report.md |
| **Factor Validation** | experiments/outputs/factor_regression_validation_report.md |
| **Regime** | outputs/hmm_regime.csv |
| **Governance** | outputs/final_risk_governance_report.csv |
| **Audit** | outputs/PRODUCTION_AUDIT.md, outputs/audit/ |

---

## 8. Main Pipeline Execution Order

1. `true_daily_returns.py` → block1, block2 CSV
2. `block2_hmm_expanding_variants.py` → block2_hmm_rebalonly.csv
3. `factor_regression.py` → factor exposure (SPY/VIX required)
4. `factor_regression_validation.py` → risk audit
5. `stress_test.py` → stress_test_report.md

---

*Report prepared for portfolio/CV use. No strategy logic or production outputs were modified in the creation of this document.*
