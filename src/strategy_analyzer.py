"""
Phase 3: Automated Strategic EDA & Feature Selection.

Runs three core quant analyses:
1. Information Coefficient (IC) – Spearman vs target, filter noise
2. Sector-specific sensitivity – macro beta per sector
3. Stationarity & drift – feature stability across time
4. Outputs: strategy_report.md, selected_features.json
"""

import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# HMM robustness: min observations for training; relaxed tol for short sub-periods
HMM_MIN_OBS = 30
HMM_TOL = 1e-3
HMM_N_ITER = 200
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IC_MIN_THRESHOLD = 0.02       # Exclude features with |IC| < this (noise)
P_VALUE_MAX = 0.05            # Exclude features with p > this (not significant)
DRIFT_IC_DIFF_THRESHOLD = 0.05  # Warn if |IC_early - IC_late| > this
DRIFT_VOL_RATIO_THRESHOLD = 2.0  # Warn if std_late / std_early > this
SPLIT_FRACTION = 0.5          # First half vs second half for drift test

# Phase 5-4: Dynamic macro detection from feature columns
MACRO_PREFIXES = ("fed_funds", "treasury", "cpi", "unemployment", "yield_curve")
RATE_FEATURE_PREFIXES = ("fed_funds", "treasury", "yield_curve")
INFLATION_FEATURE_PREFIX = "cpi"
RATE_SHOCK_SUFFIX = "_shock_20d"


def _load_data(path: Path) -> pd.DataFrame:
    """Load processed features from Phase 2."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna()
    return df


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Exclude date, sector, target from feature list."""
    exclude = {"date", "sector", "target"}
    return [c for c in df.columns if c not in exclude]


def _get_macro_features(df: pd.DataFrame) -> list[str]:
    """Macro-related columns (zscore, mom, yoy, yield_curve, shock)."""
    feature_cols = _get_feature_columns(df)
    return [c for c in feature_cols if any(c.startswith(p) or p in c for p in MACRO_PREFIXES)]


# ---------------------------------------------------------------------------
# 1. Information Coefficient Analysis
# ---------------------------------------------------------------------------
def run_ic_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_selected: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute Spearman IC and p-value per feature vs target.
    Exclude features with |IC| < 0.02 or p-value > 0.05.
    Phase 5-4: If selected < min_selected, take top min_selected by |IC|.
    """
    results = []
    y = df["target"].values

    for col in feature_cols:
        x = df[col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 50:
            results.append({"feature": col, "ic": np.nan, "pvalue": np.nan, "keep": False})
            continue

        r, p = stats.spearmanr(x[valid], y[valid], nan_policy="omit")
        ic = r if not np.isnan(r) else 0.0
        pval = p if not np.isnan(p) else 1.0

        keep = abs(ic) >= IC_MIN_THRESHOLD and pval <= P_VALUE_MAX
        results.append({"feature": col, "ic": float(ic), "pvalue": float(pval), "keep": keep})

    ic_df = pd.DataFrame(results)
    ic_df = ic_df.sort_values("ic", key=abs, ascending=False).reset_index(drop=True)
    selected = ic_df[ic_df["keep"]]["feature"].tolist()
    if len(selected) < min_selected:
        selected = ic_df.head(min(min_selected, len(ic_df)))["feature"].tolist()
    return ic_df, selected


# ---------------------------------------------------------------------------
# 2. Sector-Specific Sensitivity (Beta)
# ---------------------------------------------------------------------------
def run_sector_sensitivity(
    df: pd.DataFrame,
    macro_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Per-sector OLS: relative_ret_20d ~ macro indicators.
    Returns beta matrix and summary (rate-hike strong, inflation-vulnerable).
    """
    beta_rows = []
    for sector, grp in df.groupby("sector"):
        y = grp["relative_ret_20d"].values
        X = grp[macro_cols].values
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if valid.sum() < 30:
            continue

        X_valid = X[valid]
        y_valid = y[valid]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        reg = LinearRegression().fit(X_scaled, y_valid)
        betas = dict(zip(macro_cols, reg.coef_))
        betas["sector"] = sector
        beta_rows.append(betas)

    beta_df = pd.DataFrame(beta_rows)

    # Summarize: rate-shock strong (positive beta to rate shock), inflation-vulnerable (negative beta to CPI)
    summary = {}
    if not beta_df.empty:
        rate_cols = [c for c in beta_df.columns if c != "sector" and any(p in c for p in RATE_FEATURE_PREFIXES)]
        if rate_cols:
            beta_df["avg_rate_beta"] = beta_df[rate_cols].mean(axis=1)
            strong_rate = beta_df.nlargest(3, "avg_rate_beta")["sector"].tolist()
            summary["strong_in_rate_shock"] = strong_rate

        if INFLATION_FEATURE_PREFIX:
            inflation_cols = [c for c in beta_df.columns if INFLATION_FEATURE_PREFIX in c]
            if inflation_cols:
                beta_df["avg_inflation_beta"] = beta_df[inflation_cols].mean(axis=1)
                valid = beta_df["avg_inflation_beta"].notna()
                if valid.any():
                    weak_inflation = beta_df.loc[valid].nsmallest(3, "avg_inflation_beta")["sector"].tolist()
                    summary["vulnerable_in_inflation"] = weak_inflation

    return beta_df, summary


# ---------------------------------------------------------------------------
# 3. Stationarity & Drift Test
# ---------------------------------------------------------------------------
def run_drift_test(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Split time into first half / second half. Compare IC and feature volatility.
    Return stability table and list of drift warnings.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    mid = int(n * SPLIT_FRACTION)
    early = df.iloc[:mid]
    late = df.iloc[mid:]

    y_early = early["target"].values
    y_late = late["target"].values
    warnings = []

    rows = []
    for col in feature_cols:
        x_early = early[col].values
        x_late = late[col].values
        valid_early = ~(np.isnan(x_early) | np.isnan(y_early))
        valid_late = ~(np.isnan(x_late) | np.isnan(y_late))

        if valid_early.sum() < 30 or valid_late.sum() < 30:
            rows.append({"feature": col, "ic_early": np.nan, "ic_late": np.nan, "ic_diff": np.nan, "vol_ratio": np.nan})
            continue

        r_early, p_early = stats.spearmanr(x_early[valid_early], y_early[valid_early], nan_policy="omit")
        r_late, p_late = stats.spearmanr(x_late[valid_late], y_late[valid_late], nan_policy="omit")
        r_early = r_early if not np.isnan(r_early) else 0.0
        r_late = r_late if not np.isnan(r_late) else 0.0

        ic_diff = abs(r_early - r_late)
        sign_flip = (r_early * r_late < 0) and (abs(r_early) > 0.01 and abs(r_late) > 0.01)

        std_early = np.nanstd(x_early[valid_early])
        std_late = np.nanstd(x_late[valid_late])
        vol_ratio = std_late / std_early if std_early > 1e-10 else np.nan

        if ic_diff > DRIFT_IC_DIFF_THRESHOLD:
            warnings.append(f"Feature '{col}': IC drifted (early={r_early:.4f}, late={r_late:.4f}, diff={ic_diff:.4f})")
        if sign_flip:
            warnings.append(f"Feature '{col}': IC sign flip detected (early={r_early:.4f}, late={r_late:.4f})")
        if np.isfinite(vol_ratio) and vol_ratio > DRIFT_VOL_RATIO_THRESHOLD:
            warnings.append(f"Feature '{col}': Volatility spike (late/early std ratio={vol_ratio:.2f})")

        rows.append({
            "feature": col,
            "ic_early": r_early,
            "ic_late": r_late,
            "ic_diff": ic_diff,
            "vol_ratio": vol_ratio,
        })

    drift_df = pd.DataFrame(rows)
    return drift_df, warnings


# ---------------------------------------------------------------------------
# 4. Report Generation & Feature Export
# ---------------------------------------------------------------------------
def _render_report(
    ic_df: pd.DataFrame,
    selected_ic: list[str],
    beta_df: pd.DataFrame,
    sector_summary: dict,
    drift_df: pd.DataFrame,
    drift_warnings: list[str],
) -> str:
    """Build markdown report."""
    lines = [
        "# Macro-Quant Sector Rotation — Strategic EDA Report",
        "",
        "## 1. Information Coefficient (IC) Analysis",
        "",
        "| Feature | IC | p-value | Keep |",
        "|---------|-----|---------|------|",
    ]
    for _, row in ic_df.iterrows():
        keep_str = "✓" if row["keep"] else "✗ (noise)"
        lines.append(f"| {row['feature']} | {row['ic']:.4f} | {row['pvalue']:.4e} | {keep_str} |")
    lines.extend([
        "",
        f"**Selected features (|IC| ≥ {IC_MIN_THRESHOLD}, p ≤ {P_VALUE_MAX}):**",
        ", ".join(selected_ic) if selected_ic else "*(none)*",
        "",
        "---",
        "",
        "## 2. Sector-Specific Sensitivity (Macro Beta)",
        "",
    ])

    if not beta_df.empty:
        tbl_cols = [c for c in beta_df.columns if c != "sector" and not c.startswith("avg_")]
        lines.append("| Sector | " + " | ".join(tbl_cols) + " |")
        lines.append("|" + "---|" * (len(tbl_cols) + 1) + "")
        for _, row in beta_df.iterrows():
            vals = [row["sector"]] + [
                str(round(row[c], 4)) if isinstance(row[c], (int, float)) else str(row[c])
                for c in tbl_cols
            ]
            lines.append("| " + " | ".join(vals) + " |")

        lines.extend([
            "",
            "### Summary",
            "",
            "- **Strong in rate shock** (positive beta to rate change): " + ", ".join(sector_summary.get("strong_in_rate_shock", [])) + ".",
            "- **Vulnerable in inflation rise** (negative beta to CPI): " + ", ".join(sector_summary.get("vulnerable_in_inflation", [])) + ".",
            "",
        ])
    else:
        lines.append("*(No sufficient data for sector regression)*")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Feature Stability & Drift",
        "",
        "| Feature | IC (early) | IC (late) | |IC diff| | Vol ratio (late/early) |",
        "|---------|------------|-----------|-----------|------------------------|",
    ])
    for _, row in drift_df.iterrows():
        vol_str = f"{row['vol_ratio']:.2f}" if np.isfinite(row.get("vol_ratio", np.nan)) else "—"
        lines.append(f"| {row['feature']} | {row['ic_early']:.4f} | {row['ic_late']:.4f} | {row['ic_diff']:.4f} | {vol_str} |")

    if drift_warnings:
        lines.extend([
            "",
            "### ⚠️ Drift Warnings",
            "",
        ])
        for w in drift_warnings:
            lines.append(f"- {w}")
    else:
        lines.append("")
        lines.append("No significant drift detected.")

    return "\n".join(lines)


def run(
    processed_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
    features_path: Optional[str | Path] = None,
) -> tuple[list[str], str]:
    """
    Run full Phase 3 analysis and save outputs.

    Returns
    -------
    selected_features : list of validated feature names
    report_text : full markdown report string
    """
    root = Path(__file__).resolve().parent.parent
    processed_path = Path(processed_path or root / "data" / "processed_features.csv")
    report_path = Path(report_path or root / "outputs" / "strategy_report.md")
    features_path = Path(features_path or root / "outputs" / "selected_features.json")

    df = _load_data(processed_path)
    feature_cols = _get_feature_columns(df)

    # 1. IC
    ic_df, selected_ic = run_ic_analysis(df, feature_cols)

    # 2. Sector sensitivity (macro cols: prefer rate_shock, yield_curve for rate-shock interpretation)
    macro_cols = _get_macro_features(df)
    if not macro_cols:
        macro_cols = [c for c in feature_cols if "lag20" in c or any(p in c for p in MACRO_PREFIXES)]
    beta_df, sector_summary = run_sector_sensitivity(df, macro_cols)

    # 3. Drift
    drift_df, drift_warnings = run_drift_test(df, feature_cols)

    # 4. HMM Regime (optional)
    regime_df = pd.DataFrame()
    try:
        processed_raw = pd.read_csv(processed_path, parse_dates=["date"])
        start = processed_raw["date"].min().strftime("%Y-%m-%d")
        end = processed_raw["date"].max().strftime("%Y-%m-%d")
        _, regime_df, _ = get_hmm_regime_model(start=start, end=end)
        if not regime_df.empty:
            regime_df.to_csv(root / "outputs" / "hmm_regime.csv", index=False)
    except Exception:
        pass

    # Combine filters: IC-selected AND no critical drift
    drift_excluded = set()
    for _, row in drift_df.iterrows():
        if row["feature"] in selected_ic:
            if row.get("ic_diff", 0) > 2 * DRIFT_IC_DIFF_THRESHOLD:
                drift_excluded.add(row["feature"])
            if row["feature"] in [w.split("'")[1] for w in drift_warnings if "sign flip" in w]:
                drift_excluded.add(row["feature"])

    selected_final = [f for f in selected_ic if f not in drift_excluded]
    if not selected_final and selected_ic:
        selected_final = selected_ic  # Fallback to IC-only if drift too strict

    # 5. Report
    report_text = _render_report(
        ic_df, selected_ic, beta_df, sector_summary, drift_df, drift_warnings
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Report saved to {report_path}")

    # 6. JSON
    features_path.write_text(json.dumps({"selected_features": selected_final}, indent=2), encoding="utf-8")
    print(f"Selected features saved to {features_path}")

    return selected_final, report_text


# ---------------------------------------------------------------------------
# 5. Hidden Regime Detection (HMM) — 2-State Core-Crisis (Production)
# ---------------------------------------------------------------------------
HMM_N_STATES = 2
HMM_N_INIT = 30
HMM_OCCUPANCY_DEAD = 0.05
HMM_OCCUPANCY_GRADE_A = 0.10
HYSTERESIS_ENTER = 0.8
HYSTERESIS_EXIT = 0.6

FEATURE_NAMES_4D = ["credit_stress", "vol_term", "sector_disp", "market_mom"]


def _load_hmm_features_4d(start: Optional[str] = None, end: Optional[str] = None) -> tuple[pd.DataFrame, Optional[object]]:
    """
    Load 4D features for HMM:
    a) Credit Stress: HYG ret - IEF ret (return difference)
    b) Vol Term Structure: VIX / VXV ratio
    c) Sector Dispersion: cross-sectional std of 11 sector returns
    d) Market Momentum: SPY price vs 20d MA distance (log ratio)
    Returns (DataFrame with date + 4 cols, scaler fit on data).
    """
    try:
        import yfinance as yf
        from src.config import SECTOR_ETFS_11
    except ImportError:
        return pd.DataFrame(), None, pd.DataFrame()

    def _get_close(ticker: str) -> pd.Series:
        d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if d.empty:
            return pd.Series(dtype=float)
        c = d["Close"] if "Close" in d.columns else d.iloc[:, 0]
        return c.squeeze()

    closes = {}
    for t in ["HYG", "IEF", "SPY", "^VIX", "^VIX3M", "^VXV"] + list(SECTOR_ETFS_11):
        s = _get_close(t)
        if not s.empty:
            closes[t] = s

    if len(closes) < 7:
        return pd.DataFrame(), None, pd.DataFrame()

    common_idx = None
    for v in closes.values():
        idx = v.dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    if common_idx is None or len(common_idx) < HMM_MIN_OBS:
        return pd.DataFrame(), None, pd.DataFrame()

    # a) Credit Stress: HYG ret - IEF ret
    hyg = closes.get("HYG", pd.Series(dtype=float)).reindex(common_idx).ffill().bfill()
    ief = closes.get("IEF", pd.Series(dtype=float)).reindex(common_idx).ffill().bfill()
    hyg_ret = np.log(hyg / hyg.shift(1))
    ief_ret = np.log(ief / ief.shift(1))
    credit_stress = (hyg_ret - ief_ret).dropna()

    # b) Vol Term: VIX / VIX3M (VXV may be delisted; ^VIX3M = 3-month VIX)
    vix = closes.get("^VIX", pd.Series(dtype=float)).reindex(common_idx).ffill().bfill().replace(0, np.nan)
    vix3m = closes.get("^VIX3M", closes.get("^VXV", pd.Series(dtype=float))).reindex(common_idx).ffill().bfill().replace(0, np.nan)
    vol_term = (vix / vix3m).replace(np.inf, np.nan)

    # c) Sector Dispersion: cross-sectional std of sector returns
    sector_tickers = [c for c in SECTOR_ETFS_11 if c in closes]
    sector_rets = pd.DataFrame({
        c: np.log(closes[c].reindex(common_idx).ffill().bfill() / closes[c].reindex(common_idx).ffill().bfill().shift(1))
        for c in sector_tickers
    })
    sector_disp = sector_rets.std(axis=1)

    # d) Market Momentum: SPY price / 20d MA (log)
    spy = closes.get("SPY", pd.Series(dtype=float)).reindex(common_idx).ffill().bfill()
    ma20 = spy.rolling(20).mean()
    market_mom = np.log(spy / ma20)

    df = pd.DataFrame({
        "credit_stress": credit_stress,
        "vol_term": vol_term,
        "sector_disp": sector_disp,
        "market_mom": market_mom,
    }, index=common_idx).dropna(how="any")

    if len(df) < HMM_MIN_OBS:
        return pd.DataFrame(), None, pd.DataFrame()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_NAMES_4D])
    df_scaled = pd.DataFrame(X_scaled, index=df.index, columns=FEATURE_NAMES_4D)
    return df_scaled, scaler, df


def _fit_hmm_2state(
    X: np.ndarray,
    scaler: Optional[object] = None,
    n_init: int = HMM_N_INIT,
) -> tuple[Optional[object], np.ndarray, np.ndarray, dict, dict]:
    """
    2-state Core-Crisis HMM: Normal vs Crisis.
    KMeans warm start, init_params="", params="stmc", transmat diag=0.8.
    Crisis = state with higher (credit_stress + vol_term).
    Returns (model, p_crisis, probs, bic_info, feature_means).
    """
    n_states = 2
    empty_result = (None, np.array([]), np.array([]), {"selected_n": 2, "log_likelihood": None, "occupancy": {}}, {})
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.cluster import KMeans
    except ImportError:
        return empty_result

    n, n_feat = X.shape
    if n < HMM_MIN_OBS or n_feat != 4:
        return empty_result

    best_model, best_score = None, -np.inf

    for trial in range(n_init):
        try:
            kmeans = KMeans(n_clusters=n_states, random_state=RANDOM_SEED + trial, n_init=10).fit(X)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            covars = np.zeros((n_states, n_feat))
            for k in range(n_states):
                mask = labels == k
                if mask.sum() > 1:
                    covars[k] = np.var(X[mask], axis=0) + 1e-6
                else:
                    covars[k] = np.var(X, axis=0) + 1e-6

            trans = np.eye(n_states) * 0.8 + (1 - np.eye(n_states)) * 0.2 / (n_states - 1)

            m = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=HMM_N_ITER,
                tol=HMM_TOL,
                random_state=RANDOM_SEED + trial,
                verbose=False,
                init_params="",
                params="stmc",
            )
            m.means_ = centers.copy()
            m.covars_ = covars.copy()
            m.startprob_ = np.full(n_states, 1.0 / n_states)
            m.transmat_ = trans.copy()

            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    m.fit(X)

            score = m.score(X)
            if score > best_score:
                best_score = score
                best_model = m

        except Exception:
            continue

    if best_model is None:
        return empty_result

    model = best_model
    X_used = X
    probs = model.predict_proba(X_used)
    states = model.predict(X_used)
    unique, counts = np.unique(states, return_counts=True)
    total = counts.sum()
    occupancy = {int(u): float(c / total) for u, c in zip(unique, counts)}

    # Crisis = state with higher (credit_stress + vol_term)
    feature_means = {}
    stress_scores = []
    for k in range(n_states):
        mask = states == k
        if mask.any():
            fm = {FEATURE_NAMES_4D[j]: float(X_used[mask, j].mean()) for j in range(4)}
            feature_means[k] = fm
            stress_scores.append(fm.get("credit_stress", 0) + fm.get("vol_term", 0))
        else:
            feature_means[k] = {f: 0.0 for f in FEATURE_NAMES_4D}
            stress_scores.append(-1e9)

    crisis_idx = int(np.argmax(stress_scores))
    p_crisis = probs[:, crisis_idx]

    bic_info = {
        "selected_n": n_states,
        "log_likelihood": float(best_score),
        "occupancy": occupancy,
    }
    return model, p_crisis, probs, bic_info, feature_means


def _apply_hysteresis(p_raw: np.ndarray, enter: float = HYSTERESIS_ENTER, exit_: float = HYSTERESIS_EXIT) -> np.ndarray:
    """Apply hysteresis: enter crisis when p > enter, exit when p < exit."""
    out = np.zeros_like(p_raw, dtype=float)
    in_crisis = False
    for i in range(len(p_raw)):
        if in_crisis:
            if p_raw[i] < exit_:
                in_crisis = False
                out[i] = float(p_raw[i])
            else:
                out[i] = enter
        else:
            if p_raw[i] > enter:
                in_crisis = True
                out[i] = enter
            else:
                out[i] = float(p_raw[i])
    return out


def get_hmm_regime_model(
    start: Optional[str] = None,
    end: Optional[str] = None,
    hysteresis_enter: float = HYSTERESIS_ENTER,
    hysteresis_exit: float = HYSTERESIS_EXIT,
    crisis_selector: str = "default",
) -> tuple[Optional[object], pd.DataFrame, dict]:
    """
    Professional 4-state HMM with 4D features, K-Means warm start, 3-stage selection.
    Returns (model, DataFrame with date, state, P_Crisis, forward_ret_20d, bic_info).
    bic_info: {selected_n, log_likelihood, occupancy, feature_means_per_state}
    """
    empty_bic = {"selected_n": 4, "log_likelihood": None, "occupancy": {}, "feature_means_per_state": {}}
    out = _load_hmm_features_4d(start=start, end=end)
    if len(out) < 2:
        return None, pd.DataFrame(), empty_bic
    df_feat, scaler = out[0], out[1]
    if df_feat.empty or scaler is None:
        return None, pd.DataFrame(), empty_bic

    X = df_feat[FEATURE_NAMES_4D].values
    dates_valid = df_feat.index

    model, p_crisis, probs, bic_info, feature_means = _fit_hmm_2state(X, scaler=scaler, n_init=HMM_N_INIT)
    if model is None or len(p_crisis) == 0:
        bic_info["feature_means_per_state"] = feature_means
        return None, pd.DataFrame(), bic_info

    states = model.predict(X)

    # Forward 20-day SPY return
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if not spy.empty:
            spy_close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
            if isinstance(spy_close, pd.DataFrame):
                spy_close = spy_close.squeeze()
            spy_aligned = spy_close.reindex(dates_valid).ffill().bfill()
            fwd_20d = np.log(spy_aligned.shift(-20) / spy_aligned)
            fwd_vals = fwd_20d.values
        else:
            fwd_vals = np.full(len(dates_valid), np.nan)
    except Exception:
        fwd_vals = np.full(len(dates_valid), np.nan)

    p_crisis_hyst = _apply_hysteresis(p_crisis, hysteresis_enter, hysteresis_exit)

    bic_info["feature_means_per_state"] = feature_means

    result = pd.DataFrame({
        "date": dates_valid,
        "state": states,
        "P_Crisis": p_crisis_hyst,
        "forward_ret_20d": fwd_vals,
    })
    return model, result, bic_info


def get_hmm_input_data(start: Optional[str] = None, end: Optional[str] = None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return (X_raw, dates) for HMM: 4D raw features. Used for expanding-window inference."""
    out = _load_hmm_features_4d(start=start, end=end)
    if len(out) < 3:
        return np.array([]), pd.DatetimeIndex([])
    df_scaled, scaler, df_raw = out[0], out[1], out[2]
    if df_raw.empty:
        return np.array([]), pd.DatetimeIndex([])
    X = df_raw[FEATURE_NAMES_4D].values
    dates = pd.DatetimeIndex(df_raw.index)
    if len(dates) < HMM_MIN_OBS:
        return np.array([]), pd.DatetimeIndex([])
    return X, dates


def get_p_crisis_expanding(
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    asof_date: pd.Timestamp,
    hysteresis_enter: float = HYSTERESIS_ENTER,
    hysteresis_exit: float = HYSTERESIS_EXIT,
) -> float:
    """
    Expanding-window HMM: fit on data up to asof_date only (4D features, scaled).
    Returns P_Crisis for the last observation (no look-ahead), with hysteresis.
    On data shortage or convergence failure: return 0.5 (neutral).
    """
    try:
        idx = dates.get_indexer([asof_date], method="ffill")[0]
        if idx < 0:
            return 0.5
        X_sub = X[: idx + 1]
        if X_sub.shape[0] < HMM_MIN_OBS or X_sub.shape[1] != 4:
            return 0.5
        # No look-ahead scaling: scaler fit ONLY on X_sub (data up to asof_date)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        _, p_crisis, _, _, _ = _fit_hmm_2state(X_scaled, scaler=scaler, n_init=min(10, HMM_N_INIT))
        if len(p_crisis) == 0:
            return 0.5
        p_hyst = _apply_hysteresis(p_crisis, HYSTERESIS_ENTER, HYSTERESIS_EXIT)
        return float(p_hyst[-1])
    except Exception:
        return 0.5


def get_p_crisis_asof(date: pd.Timestamp, regime_df: pd.DataFrame) -> float:
    """Return P_Crisis as of date (latest available before date)."""
    if regime_df.empty or "P_Crisis" not in regime_df.columns:
        return 0.0
    subset = regime_df[regime_df["date"] <= date]
    if subset.empty:
        return 0.0
    return float(subset.iloc[-1]["P_Crisis"])


if __name__ == "__main__":
    run()
