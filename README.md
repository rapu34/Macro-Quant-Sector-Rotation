# Macro-Quant Sector Rotation

Combine macroeconomic indicators with US sector ETF prices to analyze and predict promising sectors. Pipeline is modular for extension and portfolio use.

---

## Project phases (outline)

| Phase | Module | Description |
|-------|--------|-------------|
| **1** | `src/data_loader.py` | **Data ingestion** – Fetch macro data (FRED) and sector ETF prices (yfinance). |
| **2** | `src/feature_engineer.py` | **Feature engineering** – Moving averages, RSI, return volatility, macro lags. |
| **3** | `src/strategy_analyzer.py` | **Strategic EDA & feature selection** – IC, sector beta, drift test. Outputs: `outputs/strategy_report.md`, `outputs/selected_features.json`. |
| **4** | `src/model_trainer.py` | **Machine learning** – XGBoost or Random Forest for sector relative return prediction. |
| **5** | `app.py` | **Visualization** – Streamlit dashboard for strategy performance and predictions. |

---

## Pipeline contract (modular I/O)

Each phase consumes output from the previous step. Keeping this contract stable makes it easy to extend or swap implementations.

- **Phase 1 output**
  - `macro_df`: `DatetimeIndex`, columns = macro indicator names (e.g. `fed_funds_rate`, `cpi_all_urban`).
  - `sector_df`: `DatetimeIndex`, columns = sector ETF tickers (e.g. `XLK`, `XLF`), values = adjusted close.

- **Phase 2 input:** `data/raw_data.csv` (or `macro_df`, `sector_df`).  
  **Output:** `data/processed_features.csv` — feature matrix (macro lag20, volatility_20d, relative_ret_20d, RSI, ma_ratio) and target (1 if sector in top 3 by 20d forward return, else 0).

- **Phase 3 input:** `data/processed_features.csv`.  
  **Output:** `outputs/strategy_report.md` (IC, sector beta, drift), `outputs/selected_features.json` (validated feature list).

- **Phase 4 input:** `data/processed_features.csv` + `outputs/selected_features.json`.  
  **Output:** `outputs/model.pkl`, `outputs/backtest_report.md`, `outputs/backtest_chart.png` (walk-forward, costs, Sharpe, MDD).

- **Phase 5 input:** Predictions, backtest results, model metadata.  
  **Output:** Streamlit app (charts and tables).

---

## Setup

```bash
pip install -r requirements.txt
```

Set `FRED_API_KEY` in a `.env` file at the project root (see `.env.example` if present).

---

## Self-Verification (Phase 5-3)

개발 완료 후 다음 명령으로 로직 검증을 수행하라:

```bash
python tests/verify_advanced_logic.py
```

- **HMM Labeling Check**: Crisis 상태가 최저 Sharpe(또는 최저 수익/최고 변동성)를 가지는지 Assert
- **BIC Logic Check**: BIC 수식 `-2*ln(L) + k*ln(n)` 검증
- **Turnover Control Check**: <5% Skip, >25% 50% Cap 동작 Print로 증명
- **Block Bootstrap Check**: CVaR 신뢰구간 `(point, lo, hi)` 반환 확인

모든 체크가 **PASS**되면 Phase 5-3 로직이 정상 작동한 것이다.

---

## Repo structure (current)

```
.
├── .env                 # FRED_API_KEY (git-ignored)
├── .gitignore
├── README.md
├── requirements.txt
├── tests/
│   └── verify_advanced_logic.py  # Phase 5-3 self-verification
├── app.py               # Phase 5: Streamlit dashboard (stub)
├── data/
│   ├── raw_data.csv
│   └── processed_features.csv
├── outputs/
│   ├── strategy_report.md
│   └── selected_features.json
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py       # Phase 1
    ├── feature_engineer.py  # Phase 2
    ├── strategy_analyzer.py # Phase 3
    └── model_trainer.py     # Phase 4 (stub)
```
