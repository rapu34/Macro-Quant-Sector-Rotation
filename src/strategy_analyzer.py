"""
Phase 3: Automated Strategic EDA & Feature Selection.

Runs three core quant analyses:
1. Information Coefficient (IC) – Spearman vs target, filter noise
2. Sector-specific sensitivity – macro beta per sector
3. Stationarity & drift – feature stability across time
4. Outputs: strategy_report.md, selected_features.json
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
        _, regime_df = get_hmm_regime_model(start=start, end=end)
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
# 5. Hidden Regime Detection (HMM) — Robust Engine
# ---------------------------------------------------------------------------
def _compute_bic(model, X: np.ndarray) -> float:
    """BIC = -2*ln(L) + k*ln(n). Lower is better."""
    n = X.shape[0]
    logL = model.score(X)
    n_states = model.n_components
    n_features = X.shape[1]
    k = n_states**2 + (n_states - 1) + n_states * n_features * 2  # trans + start + means + diag cov
    return -2 * logL + k * np.log(n)


def _fit_hmm_bic(X: np.ndarray) -> tuple[object, np.ndarray, dict]:
    """
    Fit HMM with BIC-based n_components selection (2~4).
    covariance_type='diag'. Returns (model, P_Crisis array, state_mapping).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None, np.array([]), {}

    best_bic, best_model, best_probs = np.inf, None, None
    for n_comp in [2, 3, 4]:
        if X.shape[0] < n_comp * 5:
            continue
        try:
            m = GaussianHMM(n_components=n_comp, covariance_type="diag", n_iter=100, random_state=42)
            m.fit(X)
            bic = _compute_bic(m, X)
            if bic < best_bic:
                best_bic = bic
                best_model = m
                probs = m.predict_proba(X)
                states = m.predict(X)
                scores = []
                for k in range(n_comp):
                    mask = states == k
                    if not mask.any():
                        scores.append(1e9)
                        continue
                    mr = X[mask, 0].mean()
                    vr = X[mask, 0].std() if mask.sum() > 1 else 0.01
                    scores.append(mr - 1.0 * vr)
                crisis_idx = int(np.argmin(scores))
                best_probs = probs[:, crisis_idx]
        except Exception:
            continue
    return best_model, best_probs if best_probs is not None else np.array([]), {}


def get_hmm_regime_model(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[Optional[object], pd.DataFrame]:
    """
    Fit HMM (BIC selection, diag cov, state labeling).
    Returns (model, DataFrame with date, state, P_Crisis).
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, pd.DataFrame()

    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    if spy.empty or vix.empty:
        return None, pd.DataFrame()

    spy_close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
    vix_close = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.squeeze()
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.squeeze()

    spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    vix_chg = (vix_close - vix_close.shift(1)) / vix_close.shift(1)
    common = spy_ret.index.intersection(vix_chg.index)
    spy_a = spy_ret.reindex(common).ffill().bfill()
    vix_a = vix_chg.reindex(common).ffill().bfill()
    valid = ~(spy_a.isna() | vix_a.isna())
    spy_a = spy_a.loc[valid].values.reshape(-1, 1)
    vix_a = vix_a.loc[valid].values.reshape(-1, 1)
    dates_valid = common[valid]
    X = np.column_stack([spy_a.squeeze(), vix_a.squeeze()])

    model, p_crisis, _ = _fit_hmm_bic(X)
    if model is None or len(p_crisis) == 0:
        return None, pd.DataFrame()
    states = model.predict(X)

    result = pd.DataFrame({
        "date": dates_valid,
        "state": states,
        "P_Crisis": p_crisis,
    })
    return model, result


def get_hmm_input_data(start: Optional[str] = None, end: Optional[str] = None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return (X, dates) for HMM: SPY log return + VIX change rate. Used for expanding-window inference."""
    try:
        import yfinance as yf
    except ImportError:
        return np.array([]), pd.DatetimeIndex([])
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    if spy.empty or vix.empty:
        return np.array([]), pd.DatetimeIndex([])
    spy_close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0].squeeze()
    vix_close = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0].squeeze()
    spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    vix_chg = (vix_close - vix_close.shift(1)) / vix_close.shift(1)
    common = spy_ret.index.intersection(vix_chg.index)
    valid = ~(spy_ret.reindex(common).isna() | vix_chg.reindex(common).isna())
    dates = common[valid]
    X = np.column_stack([
        spy_ret.reindex(dates).ffill().bfill().values,
        vix_chg.reindex(dates).ffill().bfill().values,
    ])
    return X, pd.DatetimeIndex(dates)


def get_p_crisis_expanding(
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    asof_date: pd.Timestamp,
) -> float:
    """
    Expanding-window HMM: fit on data up to asof_date only.
    Returns P_Crisis for the last observation (no look-ahead).
    """
    idx = dates.get_indexer([asof_date], method="ffill")[0]
    if idx < 0:
        return 0.0
    X_sub = X[: idx + 1]
    if X_sub.shape[0] < 20:
        return 0.0
    _, p_crisis, _ = _fit_hmm_bic(X_sub)
    return float(p_crisis[-1]) if len(p_crisis) > 0 else 0.0


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
