"""
Phase 4: Model Training & Quant Backtesting.

XGBoost model with walk-forward validation, transaction costs, and quant metrics.
Outputs: model.pkl, backtest_report.md.
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# Suppress XGBoost use_label_encoder deprecation (removed in code)
warnings.filterwarnings("ignore", message=".*use_label_encoder.*", category=UserWarning)
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.config import (
    BLOCK_BOOTSTRAP_ITER,
    BLOCK_SIZE_MAX,
    BLOCK_SIZE_MIN,
    RANDOM_SEED,
    CROWDING_LOOKBACK,
    CROWDING_MULTIPLIER,
    CVAR_ALPHA,
    EXTREME_FEAR_THRESHOLD,
    GROSS_EXPOSURE_CAP,
    KELLY_FRACTION,
    KELLY_ROLLING_DAYS,
    P_CRISIS_GUARD_SCALE,
    P_CRISIS_GUARD_THRESHOLD,
    TURNOVER_CAP_RATE,
    TURNOVER_CAP_THRESHOLD,
    SECTOR_ETFS,
    TARGET_HORIZON,
    TARGET_VOL,
    TARGET_VOL_MAX,
    TARGET_VOL_MIN,
    TOP_K,
    TURNOVER_THRESHOLD,
    VIX_Z_THRESHOLD_MAX,
    VIX_Z_THRESHOLD_MIN,
    WEEKLY_GUARD_DAYS,
)

# ---------------------------------------------------------------------------
# Transaction cost (configurable)
# ---------------------------------------------------------------------------
COST_RATE = 0.001  # 0.1% per side (buy or sell); round-trip = 2 * COST_RATE = 0.2%
ROUND_TRIP_RATE = 2 * COST_RATE

# XGBoost anti-overfitting
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COL_SAMPLE_BY_TREE = 0.8
N_ESTIMATORS = 200
RANDOM_STATE = RANDOM_SEED
MIN_TRAIN_PCT = 0.4   # Minimum 40% of data for first training window
REBALANCE_DAYS = 20   # Hold period; rebalance every 20 trading days


def _load_sentiment(raw_path: Path) -> pd.Series:
    """Load Market Sentiment Score (0~100) from VIX+SPY fallback."""
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    start = raw.index.min().strftime("%Y-%m-%d")
    end = raw.index.max().strftime("%Y-%m-%d")
    from src.data_loader import load_market_sentiment_score
    df = load_market_sentiment_score(start=start, end=end)
    if df.empty:
        return pd.Series(dtype=float)
    s = df["market_sentiment"].squeeze()
    s = s if isinstance(s, pd.Series) else df.iloc[:, 0]
    return s.sort_index()


def _load_data(
    processed_path: Path,
    raw_path: Path,
) -> pd.DataFrame:
    """Load processed features and raw prices; compute fwd_ret_20d per (date, sector)."""
    df = pd.read_csv(processed_path, parse_dates=["date"])
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)

    sector_cols = [c for c in raw.columns if c in SECTOR_ETFS]
    prices = raw[sector_cols]
    log_prices = np.log(prices)
    fwd_ret_20d = log_prices.shift(-TARGET_HORIZON) - log_prices

    # Unstack to long: (date, sector) -> fwd_ret_20d
    fwd_long = fwd_ret_20d.stack().reset_index()
    fwd_long.columns = ["date", "sector", "fwd_ret_20d"]

    df = df.merge(fwd_long, on=["date", "sector"], how="inner")
    return df.dropna(subset=["fwd_ret_20d"])


def _get_feature_cols(df: pd.DataFrame, selected_path: Optional[Path] = None) -> list[str]:
    exclude = {"date", "sector", "target", "fwd_ret_20d"}
    all_features = [c for c in df.columns if c not in exclude]
    if selected_path and selected_path.exists():
        with open(selected_path) as f:
            sel = json.load(f).get("selected_features", [])
        return [c for c in sel if c in all_features] or all_features
    return all_features


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """For imbalanced binary: scale_pos_weight = neg_count / pos_count."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos <= 0:
        return 1.0
    return float(n_neg / n_pos)


def _apply_crowding_filter(
    sector_weights: dict[str, float],
    sector_5d_ret: dict[str, float],
    historical_mean_5d: float,
) -> dict[str, float]:
    """
    If sector 5d ret >> historical avg, limit weight (crowding risk).
    Do NOT re-normalize to sum=1 — preserve Gross Exposure for target_vol scaling.
    """
    if historical_mean_5d <= 0:
        return sector_weights
    out = {}
    for s, w in sector_weights.items():
        r = sector_5d_ret.get(s, 0)
        if r > historical_mean_5d * CROWDING_MULTIPLIER:
            out[s] = w * 0.5
        else:
            out[s] = w
    return out


def _compute_sector_covariance(raw: pd.DataFrame, end_date, lookback: int = 252) -> Optional[pd.DataFrame]:
    """Annualized covariance matrix from daily log returns (lookback days up to end_date)."""
    scols = [c for c in raw.columns if c in SECTOR_ETFS]
    if not scols:
        return None
    ret = np.log(raw[scols] / raw[scols].shift(1)).dropna()
    ret.index = pd.to_datetime(ret.index)
    mask = ret.index <= pd.Timestamp(end_date)
    subset = ret.loc[mask].tail(lookback)
    if len(subset) < 20:
        return None
    return subset.cov() * 252  # annualized


def _compute_vol_scaled_weights(
    test_df: pd.DataFrame,
    holdings: set,
    p_crisis: float,
    target_vol: Optional[float] = None,
    kelly_cap: float = 1.0,
    exposure_diag: Optional[dict] = None,
    cov_matrix: Optional[pd.DataFrame] = None,
    gross_exposure_cap: Optional[float] = None,
    risk_mult_min: float = 0.5,
    risk_mult_step: Optional[tuple] = None,
) -> dict[str, float]:
    """
    Portfolio-level vol scaling: scale_factor = target_vol / portfolio_estimated_vol.
    portfolio_vol = sqrt(w^T Σ w). Base weights = inverse vol, normalized.
    """
    tv = target_vol or TARGET_VOL
    if risk_mult_step is not None:
        thresh, mult_val = risk_mult_step
        risk_mult = mult_val if p_crisis > thresh else 1.0
    else:
        risk_mult = max(risk_mult_min, 1.0 - p_crisis)
    cap = gross_exposure_cap if gross_exposure_cap is not None else GROSS_EXPOSURE_CAP
    cap = min(kelly_cap, cap)

    vols = {}
    for s in holdings:
        row = test_df[test_df["sector"] == s]
        vol = row["volatility_20d"].values[0] if "volatility_20d" in row.columns and len(row) > 0 else 0.02
        vols[s] = max(vol, 1e-8)
    # Base weights: inverse vol (risk parity style), normalized to sum 1
    inv_vol = {s: 1.0 / vols[s] for s in holdings}
    total = sum(inv_vol.values())
    w_base = {s: (inv_vol[s] / total) * risk_mult for s in holdings} if total > 0 else {s: risk_mult / len(holdings) for s in holdings}

    # Portfolio vol = sqrt(w^T Σ w)
    holdings_list = sorted(holdings)
    w_vec = np.array([w_base.get(s, 0) for s in holdings_list])
    if cov_matrix is not None and len(holdings_list) > 0:
        try:
            cov_sub = cov_matrix.reindex(index=holdings_list, columns=holdings_list).fillna(0)
            cov_sub = cov_sub + np.eye(len(holdings_list)) * 1e-8  # ensure PD
            port_vol = float(np.sqrt(np.dot(w_vec, np.dot(cov_sub.values, w_vec))))
        except Exception:
            port_vol = sum(w_base.get(s, 0) * vols.get(s, 0.02) for s in holdings)  # fallback
    else:
        # Fallback: diagonal (uncorrelated)
        port_vol = float(np.sqrt(sum(w_base.get(s, 0) ** 2 * vols.get(s, 0.02) ** 2 for s in holdings)))

    port_vol = max(port_vol, 1e-8)
    scale = tv / port_vol
    out = {s: w_base.get(s, 0) * scale for s in holdings}

    gross_before = sum(abs(w) for w in out.values())
    cap_binding = 1 if (gross_before > 1e-8 and cap < gross_before) else 0
    if gross_before > 1e-8 and cap < gross_before:
        out = {s: w * (cap / gross_before) for s, w in out.items()}
    gross_after = sum(abs(w) for w in out.values())

    if exposure_diag is not None:
        exposure_diag["gross_before"].append(gross_before)
        exposure_diag["gross_after"].append(gross_after)
        exposure_diag["cap_binding"].append(cap_binding)
    return out


def _turnover(prev_weights: dict, new_weights: dict) -> float:
    """Sum of |new - prev| across all sectors."""
    all_sectors = set(prev_weights.keys()) | set(new_weights.keys())
    return sum(abs(new_weights.get(s, 0) - prev_weights.get(s, 0)) for s in all_sectors)


def _apply_turnover_cap(
    prev_weights: dict,
    target_weights: dict,
) -> dict[str, float]:
    """If turnover > 25%, move 50% toward target: w_new = w_prev + 0.5*(w_target - w_prev)."""
    turnover = _turnover(prev_weights, target_weights)
    if turnover <= TURNOVER_CAP_THRESHOLD:
        return target_weights
    all_s = set(prev_weights.keys()) | set(target_weights.keys())
    return {
        s: prev_weights.get(s, 0) + TURNOVER_CAP_RATE * (target_weights.get(s, 0) - prev_weights.get(s, 0))
        for s in all_s
    }


def _get_subperiod_return(
    raw: pd.DataFrame,
    start_date,
    end_date,
    holdings: set,
    weights: dict[str, float],
) -> float:
    """5-day (or shorter) log return for portfolio from start_date to end_date."""
    scols = [c for c in raw.columns if c in SECTOR_ETFS]
    if not scols or not holdings:
        return 0.0
    raw_idx = raw.index
    try:
        start_idx = raw_idx.get_indexer([pd.Timestamp(start_date)], method="ffill")[0]
        end_idx = raw_idx.get_indexer([pd.Timestamp(end_date)], method="ffill")[0]
    except Exception:
        return 0.0
    if start_idx < 0 or end_idx < 0 or end_idx >= len(raw) or start_idx >= len(raw):
        return 0.0
    prices_start = raw.iloc[start_idx]
    prices_end = raw.iloc[end_idx]
    ret = 0.0
    for s in holdings:
        if s in scols and s in weights and weights[s] > 1e-8:
            p0 = float(prices_start.get(s, np.nan))
            p1 = float(prices_end.get(s, np.nan))
            if p0 and p1 and p0 > 1e-12:
                ret += weights[s] * (np.log(p1 / p0))
    return ret


def _compute_kelly_cap(returns: np.ndarray) -> float:
    """0.25 Kelly = 0.25 * (mu / sigma^2), clipped. Use as Gross_Exposure cap."""
    if len(returns) < 10:
        return 1.0
    mu = np.mean(returns)
    sigma = np.std(returns)
    if sigma < 1e-8:
        return 1.0
    kelly = KELLY_FRACTION * (mu / (sigma**2))
    return float(np.clip(kelly, 0.1, 2.0))


def _walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    sentiment_series: Optional[pd.Series] = None,
    use_risk_mgmt: bool = True,
    raw_path: Optional[Path] = None,
    regime_df: Optional[pd.DataFrame] = None,
    hmm_X: Optional[np.ndarray] = None,
    hmm_dates: Optional[pd.DatetimeIndex] = None,
    use_institutional: bool = False,
    fear_threshold: Optional[float] = None,
    target_vol: Optional[float] = None,
    p_crisis_log: Optional[list] = None,
    weekly_guard_log: Optional[list] = None,
    kelly_cap_min: Optional[float] = None,
    gross_exposure_log: Optional[list] = None,
    exposure_diag: Optional[dict] = None,
    gross_exposure_cap_override: Optional[float] = None,
    raw_for_cov: Optional[pd.DataFrame] = None,
    use_weekly_guard: bool = True,
    show_progress: bool = False,
    risk_mult_min: float = 0.5,
    risk_mult_step: Optional[tuple] = None,
    audit_capture: Optional[dict] = None,
    audit_capture_date: Optional[str] = None,
    weights_log: Optional[list] = None,
) -> tuple[list, list, list]:
    """
    Walk-forward: train on expanding window, predict at each rebalance, compute period returns.
    Monthly (REBALANCE_DAYS) only. Weekly Guard removed from trading — monitoring log only.

    Transaction cost: ONLY when sectors change (holdings != prev_holdings), not on turnover skip.

    Returns: (gross_returns, net_returns, rebalance_dates)
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    n_dates = len(dates)
    min_train_idx = max(1, int(n_dates * MIN_TRAIN_PCT))
    step = max(1, REBALANCE_DAYS)

    if raw_for_cov is None and raw_path is not None and use_institutional:
        try:
            raw_for_cov = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        except Exception:
            raw_for_cov = None

    raw_for_guard = raw_for_cov  # for monitoring log only

    gross_rets = []
    net_rets = []
    rebal_dates = []
    turnover_list = []
    prev_holdings = None
    prev_weights = None
    in_cash = False

    iter_range = list(range(min_train_idx, n_dates, step))
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(iter_range, desc="Backtest", unit="period")
        except ImportError:
            pbar = iter_range
    else:
        pbar = iter_range

    for test_start_idx in pbar:
        test_end_idx = min(test_start_idx + 1, n_dates)
        train_end_idx = test_start_idx
        test_date = dates[test_start_idx]
        if show_progress and pbar is not None and hasattr(pbar, "set_postfix"):
            pbar.set_postfix(date=str(test_date)[:10])
        test_date_ts = pd.Timestamp(test_date) if not isinstance(test_date, pd.Timestamp) else test_date

        # Panic trigger: sentiment < 25 -> hold cash
        if use_risk_mgmt and sentiment_series is not None and not sentiment_series.empty:
            try:
                sent_val = sentiment_series.asof(test_date_ts)
            except Exception:
                sent_val = np.nan
            th = fear_threshold if fear_threshold is not None else EXTREME_FEAR_THRESHOLD
            if pd.notna(sent_val) and float(sent_val) < th:
                period_ret = 0.0
                cost = 0.0
                if prev_holdings:
                    cost = len(prev_holdings) * COST_RATE * (1.0 / TOP_K)
                gross_rets.append(period_ret)
                net_rets.append(period_ret - cost)
                rebal_dates.append(test_date)
                turnover_list.append(1.0 if prev_holdings else 0.0)
                if weights_log is not None:
                    weights_log.append((test_date, {s: 0.0 for s in SECTOR_ETFS}, cost, True))
                prev_holdings = set()
                in_cash = True
                continue

        if in_cash and use_risk_mgmt:
            in_cash = False

        train_dates = dates[:train_end_idx]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date]

        if len(test_df) < len(SECTOR_ETFS):
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feature_cols].values

        scaler_fit = StandardScaler()
        X_train_s = scaler_fit.fit_transform(X_train)
        X_test_s = scaler_fit.transform(X_test)

        scale_pos = _compute_scale_pos_weight(y_train)
        model = xgb.XGBClassifier(
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE,
            colsample_bytree=COL_SAMPLE_BY_TREE,
            n_estimators=N_ESTIMATORS,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
        model.fit(X_train_s, y_train)

        proba = model.predict_proba(X_test_s)[:, 1]
        test_df = test_df.copy()
        test_df["pred_rank"] = proba
        top3 = test_df.nlargest(TOP_K, "pred_rank")["sector"].tolist()
        holdings = set(top3)

        sector_5d_ret = {}
        hist_mean_5d = 0.0
        if raw_path and use_risk_mgmt:
            try:
                raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
                scols = [c for c in raw.columns if c in SECTOR_ETFS]
                if scols:
                    ret = np.log(raw[scols] / raw[scols].shift(1))
                    ret5 = ret.rolling(CROWDING_LOOKBACK).sum()
                    hist_mean_5d = ret5.abs().mean().mean()
                    idx = raw.index.get_indexer([test_date_ts], method="nearest")[0]
                    if 0 <= idx < len(ret5):
                        row = ret5.iloc[idx]
                        sector_5d_ret = row.dropna().to_dict()
            except Exception:
                pass

        p_crisis = 0.0
        if use_institutional and (hmm_X is not None and len(hmm_X) > 0 and hmm_dates is not None):
            from src.strategy_analyzer import get_p_crisis_expanding
            p_crisis = get_p_crisis_expanding(hmm_X, hmm_dates, test_date_ts)
        elif use_institutional and regime_df is not None and not regime_df.empty:
            from src.strategy_analyzer import get_p_crisis_asof
            p_crisis = get_p_crisis_asof(test_date_ts, regime_df)

        risk_mult = max(risk_mult_min, 1.0 - p_crisis)
        if p_crisis_log is not None:
            p_crisis_log.append({"date": test_date, "p_crisis": p_crisis, "risk_mult": risk_mult})

        kelly_cap = 1.0
        if use_institutional and raw_path and len(gross_rets) >= KELLY_ROLLING_DAYS // REBALANCE_DAYS:
            try:
                hist_rets = np.array(gross_rets[-(KELLY_ROLLING_DAYS // REBALANCE_DAYS):])
                kelly_cap = _compute_kelly_cap(hist_rets)
            except Exception:
                pass
        if kelly_cap_min is not None:
            kelly_cap = max(kelly_cap, kelly_cap_min)

        tv = target_vol or TARGET_VOL
        cov_matrix = None
        if use_institutional and raw_for_cov is not None:
            try:
                cov_matrix = _compute_sector_covariance(raw_for_cov, test_date_ts)
            except Exception:
                pass
        if use_institutional:
            weights = _compute_vol_scaled_weights(
                test_df, holdings, p_crisis, target_vol=tv, kelly_cap=kelly_cap,
                exposure_diag=exposure_diag, cov_matrix=cov_matrix,
                gross_exposure_cap=gross_exposure_cap_override,
                risk_mult_min=risk_mult_min,
                risk_mult_step=risk_mult_step,
            )
            weights = _apply_crowding_filter(weights, sector_5d_ret, hist_mean_5d)
        else:
            weights = {s: 1.0 / TOP_K for s in holdings}
            weights = _apply_crowding_filter(weights, sector_5d_ret, hist_mean_5d)

        target_weights_full = {s: weights.get(s, 0) for s in SECTOR_ETFS}
        prev_weights_full = prev_weights or {s: 0.0 for s in SECTOR_ETFS}
        turnover_raw = _turnover(prev_weights_full, target_weights_full)
        weights = _apply_turnover_cap(prev_weights_full, target_weights_full)
        new_weights_full = {s: weights.get(s, 0) for s in SECTOR_ETFS}
        turnover = _turnover(prev_weights_full, new_weights_full)
        if gross_exposure_log is not None:
            gross_exposure_log.append(sum(abs(w) for w in weights.values()))

        # --- Transaction Cost 1: Sector replacement (섹터 교체 시에만) ---
        # Occurs ONLY when we actually change sector holdings, NOT when we skip rebalance.
        sector_rebalance_cost = 0.0
        if turnover >= TURNOVER_THRESHOLD or prev_holdings is None:
            if prev_holdings is None:
                sector_rebalance_cost = TOP_K * COST_RATE * (1.0 / TOP_K)
            elif holdings != prev_holdings:
                n_changed = len(holdings.symmetric_difference(prev_holdings))
                sector_rebalance_cost = n_changed * ROUND_TRIP_RATE * (1.0 / TOP_K)

        # --- Period return: Monthly (REBALANCE_DAYS) only — Weekly Guard removed from trading ---
        if turnover < TURNOVER_THRESHOLD and prev_holdings is not None:
            holdings = prev_holdings
            weights_held = prev_weights or {s: 1.0 / TOP_K for s in holdings}
        else:
            weights_held = {s: weights.get(s, 0) for s in SECTOR_ETFS}

        active_holdings = {s for s in SECTOR_ETFS if weights_held.get(s, 0) > 1e-8}
        if not active_holdings:
            active_holdings = holdings
        period_ret = sum(
            test_df[test_df["sector"] == s]["fwd_ret_20d"].values[0] * weights_held.get(s, 0)
            for s in active_holdings if s in test_df["sector"].values
        )

        cost = sector_rebalance_cost

        # --- Audit capture (governance only, no strategy impact) ---
        do_capture = False
        if audit_capture is not None and audit_capture_date is not None:
            if audit_capture_date == "first_2022" and pd.Timestamp(test_date) >= pd.Timestamp("2022-01-01") and "rebalance_date" not in audit_capture:
                do_capture = True
            elif str(test_date)[:10] == audit_capture_date:
                do_capture = True
        if do_capture:
            audit_capture["rebalance_date"] = str(test_date)[:10]
            audit_capture["gross_return"] = period_ret
            audit_capture["cost_deducted"] = cost
            audit_capture["net_return"] = period_ret - cost
            audit_capture["turnover_raw"] = turnover_raw
            audit_capture["turnover_after_caps"] = turnover
            audit_capture["cost_rate_applied"] = ROUND_TRIP_RATE
            audit_capture["n_sectors_changed"] = len(holdings.symmetric_difference(prev_holdings)) if prev_holdings is not None else TOP_K

        # --- Monitoring only: p_crisis_weekly_log (does NOT affect period_ret or cost) ---
        if weekly_guard_log is not None and raw_for_guard is not None and (
            (hmm_X is not None and len(hmm_X) > 0 and hmm_dates is not None)
            or (regime_df is not None and not regime_df.empty)
        ):
            period_end_idx = min(test_start_idx + REBALANCE_DAYS, n_dates)
            prev_scale = 1.0
            use_expanding = hmm_X is not None and len(hmm_X) > 0 and hmm_dates is not None
            if use_expanding:
                from src.strategy_analyzer import get_p_crisis_expanding
            else:
                from src.strategy_analyzer import get_p_crisis_asof
            for sub_start in range(test_start_idx, period_end_idx, WEEKLY_GUARD_DAYS):
                sub_end = min(sub_start + WEEKLY_GUARD_DAYS, period_end_idx)
                if sub_start >= sub_end:
                    continue
                sub_start_date = dates[sub_start]
                sub_end_date = dates[sub_end - 1]
                if use_expanding:
                    p_crisis_sub = get_p_crisis_expanding(hmm_X, hmm_dates, pd.Timestamp(sub_start_date))
                else:
                    p_crisis_sub = get_p_crisis_asof(pd.Timestamp(sub_start_date), regime_df)
                scale = P_CRISIS_GUARD_SCALE if p_crisis_sub > P_CRISIS_GUARD_THRESHOLD else 1.0
                scale_changed = abs(scale - prev_scale) > 1e-6
                prev_scale = scale
                sub_ret = _get_subperiod_return(raw_for_guard, sub_start_date, sub_end_date, holdings, weights_held)
                weekly_guard_log.append({
                    "rebalance_date": test_date,
                    "sub_start_date": sub_start_date,
                    "sub_end_date": sub_end_date,
                    "p_crisis": p_crisis_sub,
                    "scale": scale,
                    "sub_ret": sub_ret,
                    "scale_changed": scale_changed,
                })
        gross_rets.append(period_ret)
        net_rets.append(period_ret - cost)
        rebal_dates.append(test_date)
        turnover_list.append(turnover)
        if weights_log is not None:
            w_full = {s: weights_held.get(s, 0) for s in SECTOR_ETFS}
            weights_log.append((test_date, w_full, cost, False))
        prev_holdings = holdings
        prev_weights = {s: weights.get(s, 0) for s in SECTOR_ETFS} if weights else None

    return gross_rets, net_rets, rebal_dates, turnover_list


def _metrics(returns: list[float]) -> dict:
    """
    Compute Sharpe (annualized), MDD, CVaR (95%), cumulative return.
    Wealth curve: Wealth_t = Wealth_{t-1} * (1 + r_t), clamped to >= 0.
    MDD = min((Wealth_t - peak) / peak), in [0, -100%] (decimal: 0 to -1).
    """
    arr = np.array(returns)
    if len(arr) < 2:
        return {"sharpe": 0.0, "mdd": 0.0, "cvar": 0.0, "cum_return": 0.0}

    periods_per_year = 252 / REBALANCE_DAYS
    sharpe = (arr.mean() / arr.std()) * np.sqrt(periods_per_year) if arr.std() > 1e-10 else 0.0

    # Wealth curve: geometric, clamp to 0 (bankruptcy floor)
    wealth = np.ones(len(arr) + 1)
    for i, r in enumerate(arr):
        wealth[i + 1] = max(0.0, wealth[i] * (1.0 + r))
    wealth = wealth[1:]
    peak = np.maximum.accumulate(wealth)
    # MDD = (Wealth - peak) / peak; when peak=0, use 0; result in [-1, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 1e-12, (wealth - peak) / peak, 0.0)
    mdd = float(np.clip(np.nanmin(dd), -1.0, 0.0))

    n_tail = max(1, int(len(arr) * (1 - CVAR_ALPHA)))
    worst = np.partition(arr, n_tail - 1)[:n_tail]
    cvar = float(np.mean(worst))

    total_ret = wealth[-1] / wealth[0] - 1.0 if wealth[0] > 1e-12 else -1.0
    return {"sharpe": float(sharpe), "mdd": mdd, "cvar": cvar, "cum_return": float(total_ret)}


def _block_bootstrap_cvar(
    returns: list[float],
    alpha: float = 0.95,
    n_iter: int = 1000,
    block_size: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> tuple[float, float, float]:
    """Block bootstrap for CVaR: point estimate and 95% CI."""
    from src.config import RANDOM_SEED
    if random_seed is None:
        random_seed = RANDOM_SEED
    rng = np.random.default_rng(random_seed)
    arr = np.array(returns)
    n_tail = max(1, int(len(arr) * (1 - alpha)))
    if len(arr) < 20:
        k = min(len(arr) - 1, n_tail - 1)
        tail = np.partition(arr, k)[:n_tail]
        cvar = float(np.mean(tail))
        return cvar, cvar, cvar
    cvar_obs = float(np.mean(np.partition(arr, n_tail - 1)[:n_tail]))
    bs = block_size or int(rng.integers(BLOCK_SIZE_MIN, BLOCK_SIZE_MAX + 1))
    cvar_boot = []
    for _ in range(n_iter):
        n_blocks = (len(arr) + bs - 1) // bs
        idx = rng.integers(0, len(arr) - bs + 1, n_blocks)
        resampled = []
        for i in idx:
            resampled.extend(arr[i : i + bs].tolist())
        resampled = np.array(resampled[: len(arr)])
        nt = max(1, int(len(resampled) * (1 - alpha)))
        cvar_boot.append(float(np.mean(np.partition(resampled, nt - 1)[:nt])))
    cvar_boot = np.array(cvar_boot)
    lo, hi = np.percentile(cvar_boot, [2.5, 97.5])
    return cvar_obs, float(lo), float(hi)


def _find_worst_mdd_periods_nonoverlapping(
    returns: list[float],
    dates: list,
    n: int = 5,
) -> list[dict]:
    """Non-overlapping worst 5 MDD: Depth, Duration, Recovery Time. Uses wealth-based DD in [-1, 0]."""
    arr = np.array(returns)
    wealth = np.ones(len(arr) + 1)
    for i, r in enumerate(arr):
        wealth[i + 1] = max(0.0, wealth[i] * (1.0 + r))
    wealth = wealth[1:]
    peak = np.maximum.accumulate(wealth)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 1e-12, (wealth - peak) / peak, 0.0)
    periods = []
    i = 0
    while i < len(dd):
        if dd[i] >= 0:
            i += 1
            continue
        start_idx = i
        while i < len(dd) and dd[i] < 0:
            i += 1
        end_idx = i - 1
        depth = float(dd[start_idx : end_idx + 1].min())
        trough_idx = start_idx + np.argmin(dd[start_idx : end_idx + 1])
        duration = end_idx - start_idx + 1
        recovery = end_idx - trough_idx
        periods.append({
            "start_date": dates[start_idx] if start_idx < len(dates) else dates[0],
            "trough_date": dates[trough_idx] if trough_idx < len(dates) else dates[0],
            "end_date": dates[end_idx] if end_idx < len(dates) else dates[-1],
            "start_idx": start_idx,
            "end_idx": end_idx,
            "depth": depth,
            "duration": duration,
            "recovery": recovery,
        })
    periods.sort(key=lambda x: x["depth"])
    selected = []
    used = set()
    for p in periods:
        overlap = False
        for j in range(p["start_idx"], p["end_idx"] + 1):
            if j in used:
                overlap = True
                break
        if not overlap:
            for j in range(p["start_idx"], p["end_idx"] + 1):
                used.add(j)
            selected.append(p)
            if len(selected) >= n:
                break
    if not selected and periods:
        return [{"start_date": p["start_date"], "trough_date": p["trough_date"], "depth": p["depth"], "duration": p.get("duration", 0), "recovery": p.get("recovery", 0)} for p in periods[:n]]
    return selected


def _find_worst_mdd_periods(returns: list[float], dates: list, n: int = 5) -> list[dict]:
    """Identify n worst drawdown periods (start, trough, depth). Uses wealth-based DD in [-1, 0]."""
    arr = np.array(returns)
    wealth = np.ones(len(arr) + 1)
    for i, r in enumerate(arr):
        wealth[i + 1] = max(0.0, wealth[i] * (1.0 + r))
    wealth = wealth[1:]
    peak = np.maximum.accumulate(wealth)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 1e-12, (wealth - peak) / peak, 0.0)
    periods = []
    i = 0
    while i < len(dd):
        if dd[i] >= 0:
            i += 1
            continue
        start_idx = i
        while i < len(dd) and dd[i] < 0:
            i += 1
        end_idx = i - 1
        depth = float(dd[start_idx : end_idx + 1].min())
        trough_idx = start_idx + np.argmin(dd[start_idx : end_idx + 1])
        periods.append({
            "start_date": dates[start_idx] if start_idx < len(dates) else dates[0],
            "trough_date": dates[trough_idx] if trough_idx < len(dates) else dates[0],
            "depth": depth,
        })
    periods.sort(key=lambda x: x["depth"])
    return periods[:n]


def _assert_vol_scaling(df: pd.DataFrame, feature_cols: list[str], raw_path: Optional[Path] = None) -> None:
    """Verify portfolio-level vol scaling: target_vol changes cause gross (before cap) to change proportionally."""
    sample = df[df["date"] == df["date"].unique()[len(df["date"].unique()) // 2]]
    if len(sample) < 3:
        return
    holdings = set(sample["sector"].head(3).tolist())
    test_df = sample[sample["sector"].isin(holdings)]
    if "volatility_20d" not in test_df.columns:
        return
    cov = None
    if raw_path:
        try:
            raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            sample_date = sample["date"].iloc[0]
            cov = _compute_sector_covariance(raw, sample_date)
        except Exception:
            pass
    tv1, tv2 = 0.10, 0.18
    kelly_high = 25.0
    cap_high = 10.0  # High cap so scaling is not truncated
    w1 = _compute_vol_scaled_weights(test_df, holdings, p_crisis=0.0, target_vol=tv1, kelly_cap=kelly_high, cov_matrix=cov, gross_exposure_cap=cap_high)
    w2 = _compute_vol_scaled_weights(test_df, holdings, p_crisis=0.0, target_vol=tv2, kelly_cap=kelly_high, cov_matrix=cov, gross_exposure_cap=cap_high)
    gross1 = sum(abs(v) for v in w1.values())
    gross2 = sum(abs(v) for v in w2.values())
    if gross1 > 1e-8 and gross2 > 1e-8:
        if gross1 < cap_high * 0.99 or gross2 < cap_high * 0.99:
            ratio = gross2 / gross1
            expected_ratio = tv2 / tv1
            assert abs(ratio - expected_ratio) / (expected_ratio + 1e-8) < 0.25, (
                f"Vol scaling: target_vol {tv1}->{tv2} should change gross by ~{expected_ratio:.2f}x, got {ratio:.2f}x"
            )


def _get_fear_threshold_range(sentiment_series: pd.Series, n_points: int = 7) -> np.ndarray:
    """Threshold Auto-Scaling: 10th to 50th percentile of Sentiment_Score (dynamic, not fixed 15~35)."""
    if sentiment_series is None or sentiment_series.empty:
        return np.linspace(15, 35, n_points)
    valid = sentiment_series.dropna()
    if len(valid) < 20:
        return np.linspace(15, 35, n_points)
    p10 = float(np.percentile(valid, 10))
    p50 = float(np.percentile(valid, 50))
    return np.linspace(p10, p50, n_points)


def _save_param_heatmap(out_dir: Path, raw_path: Optional[Path]) -> None:
    """
    Cap Sensitivity Heatmap: Fear threshold x GROSS_EXPOSURE_CAP -> Sharpe, MDD, cap_binding_ratio.
    target_vol fixed at 15%. Analysis of leverage cap impact on performance.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent
    raw_path = raw_path or root / "data" / "raw_data.csv"
    processed_path = root / "data" / "processed_features.csv"
    selected_path = root / "outputs" / "selected_features.json"
    if not processed_path.exists():
        return
    df = _load_data(processed_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    sentiment_series = _load_sentiment(raw_path)
    fear_thresh = _get_fear_threshold_range(sentiment_series, n_points=7)
    # Y-axis: GROSS_EXPOSURE_CAP (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    cap_values = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    target_vol_fixed = 0.15  # Fixed at 15% for all cells
    X_hmm, d_hmm = np.array([]), pd.DatetimeIndex([])
    raw_for_cov = None
    try:
        raw_for_cov = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        from src.strategy_analyzer import get_hmm_input_data
        X_hmm, d_hmm = get_hmm_input_data(raw_for_cov.index.min().strftime("%Y-%m-%d"), raw_for_cov.index.max().strftime("%Y-%m-%d"))
    except Exception:
        pass
    _assert_vol_scaling(df, feature_cols, raw_path)

    sharpes = np.full((len(cap_values), len(fear_thresh)), np.nan)
    mdds = np.full((len(cap_values), len(fear_thresh)), np.nan)
    gross_exposure_diag = np.full((len(cap_values), len(fear_thresh)), np.nan)
    gross_before_diag = np.full((len(cap_values), len(fear_thresh)), np.nan)
    gross_after_diag = np.full((len(cap_values), len(fear_thresh)), np.nan)
    cap_binding_pct_diag = np.full((len(cap_values), len(fear_thresh)), np.nan)
    try:
        from tqdm import tqdm
        cap_iter = tqdm(enumerate(cap_values), total=len(cap_values), desc="Heatmap", unit="cap")
    except ImportError:
        cap_iter = enumerate(cap_values)
    for i, cap_val in cap_iter:
        for j, ft in enumerate(fear_thresh):
            try:
                gross_log: list = []
                exposure_diag = {"gross_before": [], "gross_after": [], "cap_binding": []}
                _, rets, _, _ = _walk_forward_backtest(
                    df, feature_cols, StandardScaler(),
                    sentiment_series=sentiment_series,
                    use_risk_mgmt=True,
                    raw_path=raw_path,
                    hmm_X=X_hmm,
                    hmm_dates=d_hmm,
                    use_institutional=True,
                    target_vol=target_vol_fixed,
                    fear_threshold=float(ft),
                    kelly_cap_min=10.0,
                    gross_exposure_log=gross_log,
                    exposure_diag=exposure_diag,
                    gross_exposure_cap_override=float(cap_val),
                    raw_for_cov=raw_for_cov,
                )
                m = _metrics(rets)
                sharpes[i, j] = m["sharpe"]
                mdds[i, j] = m["mdd"] * 100
                if gross_log:
                    gross_exposure_diag[i, j] = float(np.mean(gross_log))
                if exposure_diag["gross_before"]:
                    gross_before_diag[i, j] = float(np.mean(exposure_diag["gross_before"]))
                    gross_after_diag[i, j] = float(np.mean(exposure_diag["gross_after"]))
                    cap_binding_pct_diag[i, j] = 100.0 * np.mean(exposure_diag["cap_binding"])
            except Exception:
                pass
    # Per (fear_threshold, gross_exposure_cap): gross before/after, cap_binding_ratio
    print("\n[Heatmap] Cap sensitivity diagnostics (target_vol=15% fixed):")
    for i, cap_val in enumerate(cap_values):
        for j, ft in enumerate(fear_thresh):
            gb = gross_before_diag[i, j]
            ga = gross_after_diag[i, j]
            cb = cap_binding_pct_diag[i, j]
            if not (np.isnan(gb) or np.isnan(ga) or np.isnan(cb)):
                mask = " (cap MASKED)" if cb > 50 else ""
                print(f"  ft={ft:.0f}, cap={cap_val:.1f}: gross_before={gb:.3f}, gross_after={ga:.3f}, cap_binding={cb:.1f}%{mask}")
    for i in range(len(cap_values)):
        row = gross_exposure_diag[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            print(f"[Heatmap] GROSS_EXPOSURE_CAP={cap_values[i]:.1f} => mean Gross Exposure = {np.mean(valid):.3f}")
    mdd_valid = mdds[~np.isnan(mdds)]
    if len(mdd_valid) > 0:
        assert np.all(mdd_valid >= -100.01) and np.all(mdd_valid <= 0.01), (
            f"MDD out of range [0, -100]%: min={np.nanmin(mdds):.1f}, max={np.nanmax(mdds):.1f}"
        )

    # 3-panel heatmap: Sharpe, MDD, cap_binding_ratio
    s_min, s_max = float(np.nanmin(sharpes)), float(np.nanmax(sharpes))
    m_min, m_max = float(np.nanmin(mdds)), float(np.nanmax(mdds))
    cb_min, cb_max = float(np.nanmin(cap_binding_pct_diag)), float(np.nanmax(cap_binding_pct_diag))
    s_pad = max(0.01, (s_max - s_min) * 0.05) if s_max > s_min else 0.05
    m_pad = max(0.1, (m_max - m_min) * 0.05) if m_max > m_min else 0.5
    cb_pad = max(1, (cb_max - cb_min) * 0.05) if cb_max > cb_min else 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax0 = axes[0]
    im0 = ax0.imshow(sharpes, aspect="auto", cmap="RdYlGn", vmin=s_min - s_pad, vmax=s_max + s_pad)
    ax0.set_xticks(range(len(fear_thresh)))
    ax0.set_xticklabels([f"{v:.0f}" for v in fear_thresh])
    ax0.set_yticks(range(len(cap_values)))
    ax0.set_yticklabels([f"{v:.1f}" for v in cap_values])
    ax0.set_xlabel("Fear Threshold (Sentiment <)")
    ax0.set_ylabel("GROSS_EXPOSURE_CAP")
    ax0.set_title("Sharpe Ratio")
    plt.colorbar(im0, ax=ax0)
    ax1 = axes[1]
    im1 = ax1.imshow(mdds, aspect="auto", cmap="RdYlGn_r", vmin=m_min - m_pad, vmax=m_max + m_pad)
    ax1.set_xticks(range(len(fear_thresh)))
    ax1.set_xticklabels([f"{v:.0f}" for v in fear_thresh])
    ax1.set_yticks(range(len(cap_values)))
    ax1.set_yticklabels([f"{v:.1f}" for v in cap_values])
    ax1.set_xlabel("Fear Threshold (Sentiment <)")
    ax1.set_ylabel("GROSS_EXPOSURE_CAP")
    ax1.set_title("Max Drawdown (%)")
    plt.colorbar(im1, ax=ax1)
    ax2 = axes[2]
    im2 = ax2.imshow(cap_binding_pct_diag, aspect="auto", cmap="YlOrRd", vmin=cb_min - cb_pad, vmax=cb_max + cb_pad)
    ax2.set_xticks(range(len(fear_thresh)))
    ax2.set_xticklabels([f"{v:.0f}" for v in fear_thresh])
    ax2.set_yticks(range(len(cap_values)))
    ax2.set_yticklabels([f"{v:.1f}" for v in cap_values])
    ax2.set_xlabel("Fear Threshold (Sentiment <)")
    ax2.set_ylabel("GROSS_EXPOSURE_CAP")
    ax2.set_title("Cap Binding Ratio (%)")
    plt.colorbar(im2, ax=ax2)
    plt.suptitle("Cap Sensitivity Analysis (target_vol=15% fixed)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "param_stability_heatmap.png", dpi=150)
    plt.close()


def _save_chart(
    gross_rets: list[float],
    net_rets: list[float],
    rebal_dates: list,
    out_path: Path,
    net_rets_rm: Optional[list[float]] = None,
) -> None:
    """Plot cumulative Gross, Net, and Net+RiskMgmt and save."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cum_gross = np.cumprod(1 + np.array(gross_rets)) - 1
    cum_net = np.cumprod(1 + np.array(net_rets)) - 1
    plt.figure(figsize=(10, 5))
    plt.plot(rebal_dates, cum_gross, label="Gross", color="steelblue")
    plt.plot(rebal_dates, cum_net, label="Net (costs)", color="darkorange")
    if net_rets_rm is not None and len(net_rets_rm) == len(rebal_dates):
        cum_rm = np.cumprod(1 + np.array(net_rets_rm)) - 1
        plt.plot(rebal_dates, cum_rm, label="Net + Risk Mgmt", color="forestgreen")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Sector Rotation: Gross vs Net vs Risk-Managed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _generate_risk_governance_report(
    regime_df: pd.DataFrame,
    net_rets_rm: list[float],
    rebal_dates: list,
    turnover_rm: list[float],
    net_metrics_rm: dict,
    net_metrics: dict,
    out_dir: Path,
) -> None:
    """
    Generate final_risk_governance_report.csv: Crisis ON ratio, annual count,
    avg duration, Crisis vs Normal return/vol, Turnover, Sharpe.
    """
    from src.strategy_analyzer import HYSTERESIS_ENTER

    regime_df = regime_df.copy()
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    regime_df = regime_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Crisis ON: hysteresis output = 0.8 when in crisis
    crisis_on = regime_df["P_Crisis"] >= HYSTERESIS_ENTER - 1e-6
    total_days = len(regime_df)
    crisis_days = crisis_on.sum()
    crisis_on_pct = 100.0 * crisis_days / total_days if total_days > 0 else 0.0

    # Crisis episodes (consecutive crisis days)
    crisis_episodes = []
    in_ep = False
    ep_start = None
    dates_list = regime_df["date"].tolist()
    for i in range(len(regime_df)):
        if crisis_on.iloc[i]:
            if not in_ep:
                in_ep = True
                ep_start = dates_list[i]
        else:
            if in_ep:
                crisis_episodes.append((ep_start, dates_list[i]))
                in_ep = False
    if in_ep:
        crisis_episodes.append((ep_start, dates_list[-1]))

    n_episodes = len(crisis_episodes)
    durations = []
    for s, e in crisis_episodes:
        d = (pd.Timestamp(e) - pd.Timestamp(s)).days
        durations.append(max(1, d))
    avg_duration = np.mean(durations) if durations else 0.0

    # Annual crisis count
    years = (regime_df["date"].max() - regime_df["date"].min()).days / 365.25 if len(regime_df) > 1 else 1.0
    annual_crisis_count = n_episodes / max(0.01, years)

    # Crisis vs Normal: align rebalance returns with regime
    from src.strategy_analyzer import get_p_crisis_asof
    crisis_rets, normal_rets = [], []
    for i, rd in enumerate(rebal_dates):
        if i >= len(net_rets_rm):
            break
        p = get_p_crisis_asof(pd.Timestamp(rd), regime_df)
        if p >= HYSTERESIS_ENTER - 1e-6:
            crisis_rets.append(net_rets_rm[i])
        else:
            normal_rets.append(net_rets_rm[i])

    ret_crisis = np.mean(crisis_rets) * 100 if crisis_rets else np.nan
    vol_crisis = np.std(crisis_rets) * 100 if len(crisis_rets) > 1 else np.nan
    ret_normal = np.mean(normal_rets) * 100 if normal_rets else np.nan
    vol_normal = np.std(normal_rets) * 100 if len(normal_rets) > 1 else np.nan

    # Turnover (avg per rebalance)
    avg_turnover = np.mean(turnover_rm) * 100 if turnover_rm else np.nan

    # 4-state comparison: N/A (deprecated)
    sharpe_2state = net_metrics_rm["sharpe"]
    turnover_2state = avg_turnover
    sharpe_4state = np.nan
    turnover_4state = np.nan
    sharpe_improvement = np.nan
    turnover_improvement = np.nan

    rows = [
        {"metric": "crisis_on_ratio_pct", "value": crisis_on_pct},
        {"metric": "annual_crisis_count", "value": annual_crisis_count},
        {"metric": "avg_crisis_duration_days", "value": avg_duration},
        {"metric": "ret_crisis_pct", "value": ret_crisis},
        {"metric": "vol_crisis_pct", "value": vol_crisis},
        {"metric": "ret_normal_pct", "value": ret_normal},
        {"metric": "vol_normal_pct", "value": vol_normal},
        {"metric": "sharpe_2state", "value": sharpe_2state},
        {"metric": "turnover_2state_pct", "value": turnover_2state},
        {"metric": "sharpe_4state", "value": sharpe_4state},
        {"metric": "turnover_4state_pct", "value": turnover_4state},
        {"metric": "sharpe_improvement_vs_4state", "value": sharpe_improvement},
        {"metric": "turnover_improvement_vs_4state", "value": turnover_improvement},
    ]
    report_df = pd.DataFrame(rows)
    report_path = out_dir / "final_risk_governance_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"Governance report saved to {report_path}")

    # Console summary
    print("\n=== RISK GOVERNANCE SUMMARY (2-State Core-Crisis) ===")
    print(f"  Crisis ON ratio:        {crisis_on_pct:.2f}%")
    print(f"  Annual crisis count:    {annual_crisis_count:.2f}")
    print(f"  Avg crisis duration:     {avg_duration:.1f} days")
    print(f"  Crisis regime:  ret={ret_crisis:.2f}%  vol={vol_crisis:.2f}%")
    print(f"  Normal regime:  ret={ret_normal:.2f}%  vol={vol_normal:.2f}%")
    print(f"  Sharpe (2-state):       {sharpe_2state:.4f}")
    print(f"  Avg Turnover:           {turnover_2state:.2f}%")
    print("  vs 4-state: N/A (deprecated)")
    print("=" * 50)
    print("Transitioning to Robust 2-state Framework: Prioritizing Operational Efficiency and Governance over Model Complexity.")
    print("=" * 50 + "\n")


def run(
    processed_path: Optional[Path] = None,
    raw_path: Optional[Path] = None,
    selected_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> dict:
    """
    Run Phase 4: train with walk-forward, backtest, save model and report.

    Returns
    -------
    metrics : dict with gross_metrics, net_metrics, etc.
    """
    root = Path(__file__).resolve().parent.parent
    processed_path = processed_path or root / "data" / "processed_features.csv"
    raw_path = raw_path or root / "data" / "raw_data.csv"
    selected_path = selected_path or root / "outputs" / "selected_features.json"
    out_dir = out_dir or root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(processed_path, raw_path)
    feature_cols = _get_feature_cols(df, selected_path)
    scaler = StandardScaler()

    sentiment_series = _load_sentiment(raw_path)

    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)
    start = raw.index.min().strftime("%Y-%m-%d")
    end = raw.index.max().strftime("%Y-%m-%d")
    regime_df = pd.DataFrame()
    hmm_model, bic_info = None, {}
    try:
        from src.strategy_analyzer import get_hmm_regime_model
        hmm_model, regime_df, bic_info = get_hmm_regime_model(start=start, end=end)
        if regime_df.empty:
            print("[HMM] get_hmm_regime_model returned empty regime_df")
        else:
            print(f"[HMM] get_hmm_regime_model OK: {len(regime_df)} rows")
    except Exception as e:
        print(f"[HMM] get_hmm_regime_model failed: {type(e).__name__}: {e}")

    # HMM Model Diagnostics
    if hmm_model is not None and not regime_df.empty and bic_info:
        selected_n = bic_info.get("selected_n")
        log_lik = bic_info.get("log_likelihood")
        occupancy = bic_info.get("occupancy", {})
        feature_means = bic_info.get("feature_means_per_state", {})
        print("\n=== HMM MODEL DIAGNOSTICS ===")
        print("Selected n_components:", selected_n)
        print("Log-likelihood:", log_lik)
        print("Occupancy:", occupancy)

        hidden_states = regime_df["state"].values
        unique, counts = np.unique(hidden_states, return_counts=True)
        total = counts.sum()
        occ = {int(u): float(c / total) for u, c in zip(unique, counts)} if total > 0 else occupancy
        print("\nState Occupancy:")
        for k in sorted(occ.keys()):
            print(f"  State {k}: {occ[k]:.4f}")

        inactive_states = [k for k, v in occ.items() if v < 0.05]
        print("\nInactive States (<5%):", inactive_states)

        print("\nTransition Matrix:")
        print(hmm_model.transmat_)
        print("\nStart Probabilities:")
        print(hmm_model.startprob_)

        # Per-state performance + feature means
        diag_rows = []
        if "forward_ret_20d" in regime_df.columns:
            df_diag = regime_df.dropna(subset=["forward_ret_20d"]).copy()
            if len(df_diag) > 0:
                for s in sorted(df_diag["state"].unique()):
                    subset = df_diag[df_diag["state"] == s]["forward_ret_20d"]
                    if len(subset) > 0:
                        q05 = subset.quantile(0.05)
                        cvar_95 = subset[subset <= q05].mean()
                        row = {
                            "state": int(s),
                            "count": len(subset),
                            "mean_return": float(subset.mean()),
                            "volatility": float(subset.std()) if len(subset) > 1 else 0.0,
                            "cvar_95": float(cvar_95),
                        }
                        for fname, fval in feature_means.get(int(s), {}).items():
                            row[f"feat_{fname}"] = fval
                        diag_rows.append(row)
        else:
            for s, fm in feature_means.items():
                row = {"state": int(s), "count": 0, "mean_return": np.nan, "volatility": np.nan, "cvar_95": np.nan}
                for fname, fval in fm.items():
                    row[f"feat_{fname}"] = fval
                diag_rows.append(row)

        if diag_rows:
            summary_df = pd.DataFrame(diag_rows)
            print("\nPer-State Performance (+ Feature Means):")
            print(summary_df.to_string(index=False))
            summary_df.to_csv(out_dir / "hmm_full_diagnostics.csv", index=False)
            print(f"\nDiagnostics saved to {out_dir / 'hmm_full_diagnostics.csv'}")
        print("=== END DIAGNOSTIC ===\n")

    print("[INFO] Running backtest (Net, no risk mgmt)...")
    gross_rets, net_rets, rebal_dates, _ = _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment_series,
        use_risk_mgmt=False,
        show_progress=True,
    )
    print("[INFO] Calculating risk metrics (Net)...")
    gross_metrics = _metrics(gross_rets)
    net_metrics = _metrics(net_rets)

    hmm_X, hmm_dates = np.array([]), pd.DatetimeIndex([])
    try:
        from src.strategy_analyzer import get_hmm_input_data
        hmm_X, hmm_dates = get_hmm_input_data(start, end)
        if len(hmm_X) == 0 or len(hmm_dates) == 0:
            print("[HMM] get_hmm_input_data returned empty — monitoring log may be limited")
        else:
            print(f"[HMM] get_hmm_input_data OK: {len(hmm_X)} rows, {len(hmm_dates)} dates")
    except Exception as e:
        print(f"[HMM] get_hmm_input_data failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    print("[INFO] Running backtest (Net + Risk Mgmt)...")
    p_crisis_log: list = []
    weekly_guard_log: list = []
    gross_rets_rm, net_rets_rm, rebal_dates_rm, turnover_rm = _walk_forward_backtest(
        df, feature_cols, scaler,
        sentiment_series=sentiment_series,
        use_risk_mgmt=True,
        raw_path=raw_path,
        regime_df=regime_df,
        hmm_X=hmm_X if len(hmm_X) > 0 else None,
        hmm_dates=hmm_dates if len(hmm_dates) > 0 else None,
        use_institutional=True,
        p_crisis_log=p_crisis_log,
        weekly_guard_log=weekly_guard_log,
        show_progress=True,
    )
    if p_crisis_log:
        pd.DataFrame(p_crisis_log).to_csv(out_dir / "p_crisis_log.csv", index=False)
        print(f"P_Crisis log saved to {out_dir / 'p_crisis_log.csv'}")
    if weekly_guard_log:
        pd.DataFrame(weekly_guard_log).to_csv(out_dir / "p_crisis_weekly_log.csv", index=False)
        print(f"Monitoring log saved to {out_dir / 'p_crisis_weekly_log.csv'} ({len(weekly_guard_log)} rows)")
    else:
        pd.DataFrame(columns=["rebalance_date", "sub_start_date", "sub_end_date", "p_crisis", "scale", "sub_ret", "scale_changed"]).to_csv(
            out_dir / "p_crisis_weekly_log.csv", index=False
        )
        print(f"[HMM] Monitoring log empty — created empty p_crisis_weekly_log.csv")
    print("[INFO] Calculating risk metrics (Net + Risk Mgmt)...")
    net_metrics_rm = _metrics(net_rets_rm)

    # Final model on full data
    X_all = df[feature_cols].values
    y_all = df["target"].values
    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X_all)
    scale_pos = _compute_scale_pos_weight(y_all)
    final_model = xgb.XGBClassifier(
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COL_SAMPLE_BY_TREE,
        n_estimators=N_ESTIMATORS,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    final_model.fit(X_all_s, y_all)

    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "scaler": scaler_final, "features": feature_cols}, f)
    print(f"Model saved to {model_path}")

    # Chart (Gross, Net, Net+RiskMgmt)
    print("[INFO] Generating chart...")
    chart_path = out_dir / "backtest_chart.png"
    _save_chart(gross_rets, net_rets, rebal_dates, chart_path, net_rets_rm=net_rets_rm)
    print("[INFO] Generating param stability heatmap...")
    _save_param_heatmap(out_dir, raw_path)

    print("[INFO] Calculating CVaR and MDD periods...")
    cvar_pt, cvar_lo, cvar_hi = _block_bootstrap_cvar(net_rets_rm, alpha=CVAR_ALPHA, n_iter=BLOCK_BOOTSTRAP_ITER)
    worst_mdd = _find_worst_mdd_periods_nonoverlapping(net_rets_rm, rebal_dates, n=5)

    # Risk Governance Report (2-state Core-Crisis)
    if regime_df is not None and not regime_df.empty and len(net_rets_rm) == len(rebal_dates) and len(turnover_rm) == len(rebal_dates):
        _generate_risk_governance_report(
            regime_df=regime_df,
            net_rets_rm=net_rets_rm,
            rebal_dates=rebal_dates,
            turnover_rm=turnover_rm,
            net_metrics_rm=net_metrics_rm,
            net_metrics=net_metrics,
            out_dir=out_dir,
        )
    report = _render_report(
        gross_metrics, net_metrics, net_metrics_rm,
        COST_RATE, gross_rets, net_rets,
        net_rets_rm,
        feature_cols, len(rebal_dates),
        rebal_dates=rebal_dates,
        cvar_pt=cvar_pt,
        cvar_ci=(cvar_lo, cvar_hi),
        worst_mdd=worst_mdd,
    )
    print("[INFO] Rendering report...")
    report_path = out_dir / "backtest_report.md"
    report_path.write_text(report, encoding="utf-8")

    # Final summary
    print("\n" + "=" * 50)
    print("BACKTEST COMPLETE — Summary")
    print("=" * 50)
    print(f"  Net:           Sharpe = {net_metrics['sharpe']:.4f}  |  Cum Return = {net_metrics['cum_return']*100:.2f}%")
    print(f"  Net+RiskMgmt:  Sharpe = {net_metrics_rm['sharpe']:.4f}  |  Cum Return = {net_metrics_rm['cum_return']*100:.2f}%")
    print("=" * 50)
    print(f"Report saved to {report_path}")
    print("=" * 50 + "\n")

    return {"gross": gross_metrics, "net": net_metrics, "net_with_risk_mgmt": net_metrics_rm}


def _render_report(
    gross: dict,
    net: dict,
    net_rm: dict,
    cost_rate: float,
    gross_rets: list,
    net_rets: list,
    net_rets_rm: list,
    features: list[str],
    n_rebalances: int,
    rebal_dates: Optional[list] = None,
    cvar_pt: float = 0.0,
    cvar_ci: tuple[float, float] = (0.0, 0.0),
    worst_mdd: Optional[list] = None,
) -> str:
    worst_mdd = worst_mdd or []
    mdd_improve = (net["mdd"] - net_rm["mdd"]) / (abs(net["mdd"]) + 1e-10) * 100
    mdd_improved = net_rm["mdd"] > net["mdd"]  # less negative = improved
    mdd_str = f"{abs(mdd_improve):.1f}% Reduction (Improved)" if mdd_improved else f"{abs(mdd_improve):.1f}% Increase (Worse)"
    sharpe_change = net_rm["sharpe"] - net["sharpe"]
    lines = [
        "# Regime-Adaptive Risk Management Framework",
        "",
        "> 정교한 국면 감지 및 리스크 예산(Risk Budget) 통제를 통한 자산 보호 프레임워크",
        "",
        "## Transaction Cost",
        f"- One-way cost: {cost_rate*100:.2f}%",
        f"- Round-trip: {ROUND_TRIP_RATE*100:.2f}%",
        f"- Turnover: Skip if < {TURNOVER_THRESHOLD*100}%; Cap at {TURNOVER_CAP_RATE*100}% if > {TURNOVER_CAP_THRESHOLD*100}%.",
        "",
        "## Performance Metrics",
        "",
        "| Metric | Gross | Net | Net + Risk Mgmt |",
        "|--------|-------|-----|-----------------|",
        f"| **Sharpe Ratio** | {gross['sharpe']:.4f} | {net['sharpe']:.4f} | {net_rm['sharpe']:.4f} |",
        f"| **Max Drawdown** | {gross['mdd']*100:.2f}% | {net['mdd']*100:.2f}% | {net_rm['mdd']*100:.2f}% |",
        f"| **CVaR (95%)** | {gross.get('cvar', 0)*100:.2f}% | {net.get('cvar', 0)*100:.2f}% | {net_rm.get('cvar', 0)*100:.2f}% |",
        f"| **Cumulative Return** | {gross['cum_return']*100:.2f}% | {net['cum_return']*100:.2f}% | {net_rm['cum_return']*100:.2f}% |",
        "",
        f"**Block Bootstrap CVaR (95% CI):** {cvar_pt*100:.2f}% [{cvar_ci[0]*100:.2f}%, {cvar_ci[1]*100:.2f}%]",
        "",
        "## Institutional Risk Framework",
        "",
        "- **HMM (BIC selection, diag cov)**: Expanding-window inference. Risk_Multiplier = 1 - P_Crisis.",
        f"- **Vol Scaling**: Target_Vol = {TARGET_VOL*100}%, Weight ∝ Target_Vol / Sector_Vol. Kelly Cap = {KELLY_FRACTION} Kelly.",
        f"- **Turnover**: Skip < {TURNOVER_THRESHOLD*100}%; Cap > {TURNOVER_CAP_THRESHOLD*100}%.",
        "- **Weekly Guard**: Removed from trading. `p_crisis_weekly_log.csv` is monitoring-only (dashboard).",
        "",
        "### Impact",
        f"- **MDD change**: {mdd_str}",
        f"- **Sharpe change**: {sharpe_change:+.4f}",
        "",
        f"**Number of rebalances:** {n_rebalances}",
        "",
        "## Non-Overlapping Worst 5 MDD Periods (Net + Risk Mgmt)",
        "",
    ]
    for i, p in enumerate(worst_mdd, 1):
        d = p.get("depth", 0) * 100
        dur = p.get("duration", 0)
        rec = p.get("recovery", 0)
        lines.append(f"- **#{i}**: {p['start_date']} → trough {p['trough_date']} | Depth: {d:.2f}% | Duration: {dur} | Recovery: {rec}")
    if worst_mdd:
        lines.append("")
    lines.extend([
        "## Model Configuration",
        f"- Features: {', '.join(features)}",
        f"- max_depth: {MAX_DEPTH}, learning_rate: {LEARNING_RATE}",
        "",
        "## Charts & Logs",
        "- `outputs/backtest_chart.png` — Cumulative return (Gross, Net, Risk-Managed)",
        "- `outputs/param_stability_heatmap.png` — Sharpe vs Fear threshold",
        "- `outputs/p_crisis_weekly_log.csv` — 5일 단위 p_crisis, scale, scale_changed (주간 리스크 관리 검증)",
        "",
    ])
    return "\n".join(lines)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: Optional[float] = None,
    **kwargs,
) -> xgb.XGBClassifier:
    """Train XGBoost for sector top-3 classification."""
    if scale_pos_weight is None:
        scale_pos_weight = _compute_scale_pos_weight(y.values)
    model = xgb.XGBClassifier(
        max_depth=kwargs.get("max_depth", MAX_DEPTH),
        learning_rate=kwargs.get("learning_rate", LEARNING_RATE),
        subsample=kwargs.get("subsample", SUBSAMPLE),
        colsample_bytree=kwargs.get("colsample_bytree", COL_SAMPLE_BY_TREE),
        n_estimators=kwargs.get("n_estimators", N_ESTIMATORS),
        scale_pos_weight=scale_pos_weight,
        random_state=kwargs.get("random_state", RANDOM_STATE),
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


def predict(model: xgb.XGBClassifier, X: pd.DataFrame, proba: bool = True):
    """Predict top-3 probabilities or class labels."""
    if proba:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: Model training & backtest")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run ID: write outputs to outputs/<tag>/ instead of outputs/. Default: overwrite outputs/")
    args = parser.parse_args()
    out_dir = None
    if args.tag:
        root = Path(__file__).resolve().parent.parent
        out_dir = root / "outputs" / args.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Tagged run: outputs will be written to {out_dir}")
    run(out_dir=out_dir)
