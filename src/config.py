"""
Shared configuration for the Macro-Quant Sector Rotation pipeline.

Single source of truth for series IDs, tickers, and constants used across
data_loader, feature_engineer, and model_trainer. Change here when extending.
"""

# FRED API: series_id -> output column name (used in data_loader and downstream)
FRED_SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "CPIAUCSL": "cpi_all_urban",
    "UNRATE": "unemployment_rate",
}

# US sector ETFs (SPDR). Order is consistent for relative return / ranking.
SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLB",   # Materials
    "XLU",   # Utilities
]

# Optional: default date range for pipeline runs (None = full history)
DEFAULT_START = None  # e.g. "2010-01-01"
DEFAULT_END = None    # e.g. "2024-12-31"

# Phase 2: Feature Engineering
MACRO_LAG_DAYS = 20       # Minimum 1 month (20 business days) lag for no look-ahead bias
MACRO_ZSCORE_WINDOW = 252 # 1-year rolling window for Z-score transformation
VOLATILITY_WINDOW = 20    # Rolling volatility window (log return std)
TARGET_HORIZON = 20       # Forward return horizon for target (top 3 classification)
RSI_PERIOD = 14           # RSI lookback period
MA_SHORT = 5              # Short moving average window
MA_LONG = 20              # Long moving average window
TOP_K = 3                 # Target: in top K by forward return = 1

# Risk management & sentiment
EXTREME_FEAR_THRESHOLD = 25   # Market Sentiment Score < 25 = High Risk (panic exit)
CROWDING_LOOKBACK = 5         # Days for recent return (crowding filter)
CROWDING_MULTIPLIER = 2.0     # Sector 5d ret > historical_mean * this = crowding (weight limit)

# Phase 5-2: Institutional risk framework
SENTIMENT_SMA_WINDOW = 5      # Sentiment Acceleration: Score - SMA(Score, 5)
TARGET_VOL = 0.15             # Target volatility (15%) for vol scaling
TURNOVER_THRESHOLD = 0.05     # Skip rebalance if |weight change| sum < 5%
CVAR_ALPHA = 0.95             # 95% CVaR (Conditional Value at Risk) confidence level

# Phase 5-3: Regime-Adaptive Risk Management
KELLY_FRACTION = 0.25         # Fractional Kelly for exposure cap
GROSS_EXPOSURE_CAP = 3.0     # Max leverage (300%) — survival guard
KELLY_ROLLING_DAYS = 60       # Rolling window for mu, sigma
TURNOVER_CAP_THRESHOLD = 0.25 # If turnover > 25%, cap adjustment at 50%
TURNOVER_CAP_RATE = 0.5       # Move 50% toward target when capped
SENTIMENT_SHRINKAGE_MIN_NEWS = 3  # Sectors with < N news shrink toward global mean
BLOCK_BOOTSTRAP_ITER = 1000   # Block bootstrap iterations for CVaR CI
BLOCK_SIZE_MIN = 10           # Block size (days) for bootstrap
BLOCK_SIZE_MAX = 20           # Max block size
VIX_Z_THRESHOLD_MIN = 1.0     # VIX z-threshold range for heatmap
VIX_Z_THRESHOLD_MAX = 2.5
TARGET_VOL_MIN = 0.10         # Target vol range for heatmap
TARGET_VOL_MAX = 0.18
