"""
Daily Forecasting Pipeline
====================================================

Production-oriented pipeline that:
  1. Loads and validates daily disbursement history
  2. Builds calendar / holiday / strike / lag / rolling features
  3. Fits three candidate models (Prophet, SARIMAX, LightGBM)
  4. Backtests all three with an expanding-window scheme
  5. Scores overall + business-relevant segments (Sundays, holidays, etc.)
  6. Builds a performance-weighted ensemble
  7. Produces 12-month and 24-month forecasts with P10/P50/P90 bands
  8. Aggregates to monthly totals
  9. Ships retraining / monitoring / drift-detection helpers

Usage
-----
    python daily_forecaster.py --csv path/to/disbursements.csv \
        --date-col date --value-col disbursement_amount \
        --strike-start 2025-10-29 --strike-end 2025-11-02 \
        --outdir ./forecast_output

Input CSV requirements
-----------------------
Two columns minimum: a date column (parseable) and a numeric disbursement
column. One row per calendar day (gaps are tolerated - see `load_data`,
which reindexes to a full daily range and flags/imputes missing days).

Dependencies
------------
    pip install pandas numpy scikit-learn statsmodels lightgbm prophet holidays

Author: generated for a credit portfolio analytics use case (Tanzania).
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("daily_forecaster")

RNG_SEED = 42
QUANTILES = {"lower": 0.10, "mid": 0.50, "upper": 0.90}


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_data(csv_path: str, date_col: str, value_col: str) -> pd.DataFrame:
    """Load, validate, and regularize the daily series to a continuous
    calendar (no gaps). Missing days are linearly interpolated and flagged
    with `is_imputed=1` so models/backtests can inspect that later if needed.
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{date_col}' and '{value_col}'")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    df = df.groupby("ds", as_index=False)["y"].sum()  # collapse any duplicate dates
    df = df.sort_values("ds").reset_index(drop=True)

    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(full_range)
    df.index.name = "ds"
    df["is_imputed"] = df["y"].isna().astype(int)
    n_missing = int(df["is_imputed"].sum())
    if n_missing:
        log.warning("Found %d missing calendar day(s); linearly interpolating.", n_missing)
        df["y"] = df["y"].interpolate(method="linear").bfill().ffill()

    df = df.reset_index()
    log.info("Loaded %d daily observations from %s to %s.",
              len(df), df["ds"].min().date(), df["ds"].max().date())
    return df


# ---------------------------------------------------------------------------
# 2. HOLIDAY / EVENT CALENDAR
# ---------------------------------------------------------------------------

def load_custom_holidays(csv_path_or_df: str | pd.DataFrame) -> pd.DataFrame:
    """Load a user-supplied holiday CSV (or DataFrame) with columns `date`
    and `holiday`. Returns a DataFrame in the same schema as
    `build_tanzania_holiday_table` so the two can be merged seamlessly.
    Holiday names are inspected for keywords (eid, christmas, easter) so
    the feature-engineering layer can build the right period flags.
    """
    if isinstance(csv_path_or_df, str):
        custom = pd.read_csv(csv_path_or_df)
    else:
        custom = csv_path_or_df.copy()
    required = {"date", "holiday"}
    if not required.issubset(set(custom.columns)):
        raise ValueError(f"Holiday input must contain columns {required}. Found: {list(custom.columns)}")

    custom["date"] = pd.to_datetime(custom["date"]).dt.normalize()
    custom = custom.drop_duplicates(subset=["date"]).sort_values("date")

    rows = []
    for _, row in custom.iterrows():
        name = str(row["holiday"]).strip()
        name_lower = name.lower()
        if "eid" in name_lower:
            htype = "eid"
        elif any(k in name_lower for k in ("christmas", "zawadi", "kuzaliwa", "krismasi")):
            htype = "christmas"
        elif any(k in name_lower for k in ("easter", "pasaka", "ufufuko")):
            htype = "easter"
        else:
            htype = "custom"
        rows.append({
            "date": row["date"],
            "holiday_name": name,
            "holiday_type": htype,
            "is_estimated": 0,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_tanzania_holiday_table(years: list[int], custom_holidays: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uses the `holidays` package (Tanzania) as the base calendar, tags
    Christmas / Eid periods explicitly, and flags which years' Eid dates are
    library *estimates* (Islamic calendar dates shift and are only
    confirmed close to the event by moon sighting) so forecast users know
    to double check them nearer the time.

    If `custom_holidays` is provided, it is merged with the Tanzania calendar.
    Custom entries override built-in entries on the same date.
    """
    import holidays as pyholidays

    tz_holidays = pyholidays.Tanzania(years=years)
    rows = []
    for date, name in tz_holidays.items():
        name_lower = name.lower()
        if "eid" in name_lower:
            holiday_type = "eid"
        elif "kuzaliwa kristo" in name_lower or "zawadi" in name_lower:
            holiday_type = "christmas"
        elif "pasaka" in name_lower or "kuu" in name_lower:
            holiday_type = "easter"
        else:
            holiday_type = "public_holiday"
        rows.append({
            "date": pd.Timestamp(date),
            "holiday_name": name,
            "holiday_type": holiday_type,
            "is_estimated": int("makisio" in name_lower),  # "makisio" = estimated
        })

    tz_table = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    if custom_holidays is not None and not custom_holidays.empty:
        custom_holidays = custom_holidays.copy()
        custom_holidays["date"] = pd.to_datetime(custom_holidays["date"]).dt.normalize()
        custom_dates = set(custom_holidays["date"])
        tz_table = tz_table[~tz_table["date"].dt.normalize().isin(custom_dates)]
        tz_table = pd.concat([tz_table, custom_holidays], ignore_index=True)
        tz_table = tz_table.sort_values("date").reset_index(drop=True)

    return tz_table


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    strike_start: str | None = None     # set to None to disable strike features
    strike_end: str | None = None
    strike_recovery_days: int = 14      # how long "days_from_strike" decays
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 7, 14, 21, 28, 30, 365])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 28, 30, 90, 365])
    holiday_window: int = 10            # max distance (days) for days_to/from_holiday


def build_features(df: pd.DataFrame, cfg: FeatureConfig, holiday_table: pd.DataFrame) -> pd.DataFrame:
    """Adds calendar, holiday/event, lag, and rolling features.
    IMPORTANT: lag/rolling features are shifted so that row t only uses
    information available up to and including t-1 (no leakage). Rolling
    stats are computed on the *lagged* series, not the raw target at t.
    """
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out = out.sort_values("ds").reset_index(drop=True)

    # --- Basic calendar features -------------------------------------------------
    out["day_of_week"] = out["ds"].dt.dayofweek  # 0=Mon ... 6=Sun
    out["day_name"] = out["ds"].dt.day_name()
    out["is_sunday"] = (out["day_of_week"] == 6).astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["day_of_month"] = out["ds"].dt.day
    out["week_of_year"] = out["ds"].dt.isocalendar().week.astype(int)
    out["month"] = out["ds"].dt.month
    out["quarter"] = out["ds"].dt.quarter
    out["year"] = out["ds"].dt.year
    days_in_month = out["ds"].dt.days_in_month
    out["is_month_end"] = (out["day_of_month"] >= days_in_month - 2).astype(int)      # last 3 days
    out["is_month_start"] = (out["day_of_month"] <= 3).astype(int)                    # first 3 days
    # Common salary-period proxy in TZ: paydays cluster around month-end / 25th-1st.
    out["is_salary_period"] = out["day_of_month"].isin(
        [25, 26, 27, 28, 29, 30, 31, 1, 2, 3]
    ).astype(int)

    # --- Holiday features (fully vectorized: safe & fast even inside a
    # day-by-day recursive forecast loop) -------------------------------------------
    hol = holiday_table.copy()
    norm_dates = out["ds"].dt.normalize()
    d64 = norm_dates.values.astype("datetime64[D]")

    hol_dates_sorted = np.array(sorted(set(hol["date"].dt.normalize())), dtype="datetime64[D]")
    out["is_holiday"] = np.isin(d64, hol_dates_sorted).astype(int)

    hol_type_by_date = hol.drop_duplicates("date").set_index(hol.drop_duplicates("date")["date"].dt.normalize())["holiday_type"]
    out["holiday_type"] = pd.Series(d64, index=out.index).map(hol_type_by_date.to_dict()).fillna("none")

    out["is_day_before_holiday"] = np.isin(d64 + np.timedelta64(1, "D"), hol_dates_sorted).astype(int)
    out["is_day_after_holiday"] = np.isin(d64 - np.timedelta64(1, "D"), hol_dates_sorted).astype(int)

    def _vectorized_holiday_distances(dates64: np.ndarray, hol64: np.ndarray, window: int):
        if len(hol64) == 0:
            return (np.full(len(dates64), window, dtype=float),
                    np.full(len(dates64), window, dtype=float))
        idx_right = np.searchsorted(hol64, dates64, side="left")
        days_to = np.full(len(dates64), window, dtype=float)
        has_future = idx_right < len(hol64)
        days_to[has_future] = (hol64[idx_right[has_future]] - dates64[has_future]) / np.timedelta64(1, "D")
        days_to = np.minimum(days_to, window)

        idx_left = np.searchsorted(hol64, dates64, side="right") - 1
        days_from = np.full(len(dates64), window, dtype=float)
        has_past = idx_left >= 0
        days_from[has_past] = (dates64[has_past] - hol64[idx_left[has_past]]) / np.timedelta64(1, "D")
        days_from = np.minimum(days_from, window)
        return days_to, days_from

    days_to, days_from = _vectorized_holiday_distances(d64, hol_dates_sorted, cfg.holiday_window)
    out["days_to_holiday"] = days_to
    out["days_from_holiday"] = days_from

    # Christmas / Eid specific windows (+/- N days counted as "period")
    christmas_dates = np.array(sorted(set(hol.loc[hol["holiday_type"] == "christmas", "date"].dt.normalize())), dtype="datetime64[D]")
    eid_dates = np.array(sorted(set(hol.loc[hol["holiday_type"] == "eid", "date"].dt.normalize())), dtype="datetime64[D]")

    def _near_any_vectorized(dates64: np.ndarray, target64: np.ndarray, window: int) -> np.ndarray:
        if len(target64) == 0:
            return np.zeros(len(dates64), dtype=int)
        d_to, d_from = _vectorized_holiday_distances(dates64, target64, window + 1)
        return ((d_to <= window) | (d_from <= window)).astype(int)

    out["is_christmas_period"] = _near_any_vectorized(d64, christmas_dates, 3)
    out["is_eid_period"] = _near_any_vectorized(d64, eid_dates, 2)
    out["is_day_before_christmas"] = np.isin(d64 + np.timedelta64(1, "D"), christmas_dates).astype(int)

    # --- Strike / abnormal event (vectorized, optional) -------------------------------
    if cfg.strike_start is not None and cfg.strike_end is not None:
        strike_start64 = np.datetime64(pd.Timestamp(cfg.strike_start), "D")
        strike_end64 = np.datetime64(pd.Timestamp(cfg.strike_end), "D")
        in_strike = (d64 >= strike_start64) & (d64 <= strike_end64)
        out["is_strike_period"] = in_strike.astype(int)

        days_after_end = (d64 - strike_end64) / np.timedelta64(1, "D")
        days_from_strike = np.where(
            in_strike, 0,
            np.where(d64 < strike_start64, -1,
                     np.where(days_after_end <= cfg.strike_recovery_days, days_after_end, -1)),
        )
        out["days_from_strike"] = days_from_strike.astype(int)
        out["is_strike_recovery"] = out["days_from_strike"].between(1, cfg.strike_recovery_days).astype(int)
    else:
        out["is_strike_period"] = 0
        out["days_from_strike"] = -1
        out["is_strike_recovery"] = 0

    # --- Long-term trend index (for tree/SARIMAX exogenous use) -------------------
    out["trend_index"] = np.arange(len(out))

    # --- Lag features (no leakage: shift target BEFORE any rolling calc) ----------
    for lag in cfg.lags:
        out[f"lag_{lag}"] = out["y"].shift(lag)

    # --- Rolling features computed on the ALREADY-LAGGED (t-1) series -------------
    shifted = out["y"].shift(1)
    for w in cfg.rolling_windows:
        out[f"rolling_mean_{w}"] = shifted.rolling(w, min_periods=max(3, w // 4)).mean()
        out[f"rolling_std_{w}"] = shifted.rolling(w, min_periods=max(3, w // 4)).std()

    # One-hot the categorical calendar bits used by tree models
    out = pd.get_dummies(out, columns=["holiday_type"], prefix="holtype")

    return out


# ---------------------------------------------------------------------------
# 4. METRICS
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "WAPE": np.nan, "n": 0}

    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    nonzero = y_true != 0
    mape = np.mean(np.abs(err[nonzero]) / np.abs(y_true[nonzero])) * 100 if nonzero.any() else np.nan
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = np.nan
    smape = np.nanmean(2 * np.abs(err) / denom) * 100
    wape = np.sum(np.abs(err)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "WAPE": wape, "n": len(y_true)}


SEGMENT_DEFINITIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "sunday": lambda d: d["is_sunday"] == 1,
    "weekday": lambda d: d["is_weekend"] == 0,
    "public_holiday": lambda d: d["is_holiday"] == 1,
    "day_before_holiday": lambda d: d["is_day_before_holiday"] == 1,
    "christmas_period": lambda d: d["is_christmas_period"] == 1,
    "eid_period": lambda d: d["is_eid_period"] == 1,
    "month_end": lambda d: d["is_month_end"] == 1,
    "high_disbursement": lambda d: d["y"] >= d["y"].quantile(0.9),
    "low_disbursement": lambda d: d["y"] <= d["y"].quantile(0.1),
}


def segment_scores(eval_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """eval_df must contain 'y', pred_col, and the calendar flag columns."""
    rows = []
    for name, mask_fn in SEGMENT_DEFINITIONS.items():
        try:
            mask = mask_fn(eval_df)
        except KeyError:
            continue
        sub = eval_df[mask]
        if len(sub) == 0:
            continue
        m = compute_metrics(sub["y"].values, sub[pred_col].values)
        m["segment"] = name
        rows.append(m)
    return pd.DataFrame(rows)[["segment", "n", "MAE", "RMSE", "MAPE", "sMAPE", "WAPE"]]


# ---------------------------------------------------------------------------
# 5. MODEL WRAPPERS
# Each fit_* function trains on `train` (a features dataframe) and returns a
# `predict` closure that takes a features dataframe for the forecast period
# and returns a DataFrame with columns ds, yhat, yhat_lower, yhat_upper.
# ---------------------------------------------------------------------------

def fit_prophet(train: pd.DataFrame, holiday_table: pd.DataFrame, strike_windows: list[tuple]):
    from prophet import Prophet

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.80,  # gives ~P10/P90 with yhat_lower/upper
    )

    hol_df = holiday_table[["date", "holiday_name"]].rename(
        columns={"date": "ds", "holiday_name": "holiday"}
    )
    hol_df["lower_window"] = -1
    hol_df["upper_window"] = 1

    strike_rows = []
    for start, end in strike_windows:
        strike_rows.append({
            "holiday": "strike_event", "ds": pd.Timestamp(start),
            "lower_window": 0, "upper_window": (pd.Timestamp(end) - pd.Timestamp(start)).days,
        })
    strike_df = pd.DataFrame(strike_rows)
    all_events = pd.concat([hol_df, strike_df], ignore_index=True) if strike_rows else hol_df
    m.holidays = all_events

    fit_df = train[["ds", "y"]].copy()
    m.fit(fit_df)

    def predict(future_dates: pd.DataFrame) -> pd.DataFrame:
        fut = future_dates[["ds"]].copy()
        fcst = m.predict(fut)
        return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return predict


def fit_sarimax(train: pd.DataFrame, exog_cols: list[str]):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = train["y"].values
    exog = train[exog_cols].fillna(0).values

    # Order chosen via a light grid search over a small, sane candidate set
    # (full auto-ARIMA grid search is expensive at 2 years of daily data;
    # this keeps runtime reasonable while still being data-driven).
    best_aic, best_order, best_res = np.inf, None, None
    candidates = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (3, 1, 1)]
    seasonal_order = (1, 1, 1, 7)  # weekly seasonality
    for order in candidates:
        try:
            mod = SARIMAX(
                y, exog=exog, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            res = mod.fit(disp=False)
            if res.aic < best_aic:
                best_aic, best_order, best_res = res.aic, order, res
        except Exception as e:  # noqa: BLE001
            log.debug("SARIMAX order %s failed: %s", order, e)
            continue

    if best_res is None:
        raise RuntimeError("SARIMAX failed to converge for any candidate order.")
    log.info("SARIMAX selected order=%s seasonal_order=%s (AIC=%.1f)", best_order, seasonal_order, best_aic)

    def predict(future_features: pd.DataFrame) -> pd.DataFrame:
        exog_future = future_features[exog_cols].fillna(0).values
        fc = best_res.get_forecast(steps=len(future_features), exog=exog_future)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.20)  # 80% interval -> matches Prophet's P10/P90
        return pd.DataFrame({
            "ds": future_features["ds"].values,
            "yhat": mean,
            "yhat_lower": ci[:, 0] if hasattr(ci, "shape") else ci.iloc[:, 0].values,
            "yhat_upper": ci[:, 1] if hasattr(ci, "shape") else ci.iloc[:, 1].values,
        })

    return predict


def fit_lightgbm(train: pd.DataFrame, feature_cols: list[str]):
    import lightgbm as lgb

    X = train[feature_cols].fillna(0)
    y = train["y"].values

    params = dict(
        objective="quantile", n_estimators=400, learning_rate=0.03,
        num_leaves=31, min_child_samples=15, subsample=0.8,
        colsample_bytree=0.8, random_state=RNG_SEED, verbosity=-1,
    )
    models = {}
    for name, q in QUANTILES.items():
        mdl = lgb.LGBMRegressor(alpha=q, **params)
        mdl.fit(X, y)
        models[name] = mdl

    def predict(future_features: pd.DataFrame) -> pd.DataFrame:
        Xf = future_features[feature_cols].fillna(0)
        preds = {name: mdl.predict(Xf) for name, mdl in models.items()}
        # enforce monotonicity lower <= mid <= upper
        lower = np.minimum(preds["lower"], preds["mid"])
        upper = np.maximum(preds["upper"], preds["mid"])
        return pd.DataFrame({
            "ds": future_features["ds"].values,
            "yhat": preds["mid"],
            "yhat_lower": lower,
            "yhat_upper": upper,
        })

    return predict, models


LGB_FEATURE_COLS_BASE = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_sunday", "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
    "days_to_holiday", "days_from_holiday", "is_christmas_period", "is_eid_period",
    "is_month_end", "is_month_start", "is_salary_period",
    "is_strike_period", "is_strike_recovery", "days_from_strike", "trend_index",
]


def _lgb_feature_cols(df: pd.DataFrame) -> list[str]:
    lag_roll = [c for c in df.columns if c.startswith(("lag_", "rolling_"))]
    holtype = [c for c in df.columns if c.startswith("holtype_")]
    return [c for c in LGB_FEATURE_COLS_BASE + lag_roll + holtype if c in df.columns]


SARIMAX_EXOG_COLS_BASE = [
    "is_weekend", "is_sunday", "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
    "is_christmas_period", "is_eid_period", "is_month_end", "is_month_start",
    "is_salary_period", "is_strike_period", "is_strike_recovery",
]


def _sarimax_exog_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in SARIMAX_EXOG_COLS_BASE if c in df.columns]


# ---------------------------------------------------------------------------
# 6. BACKTESTING (expanding window)
# ---------------------------------------------------------------------------

def expanding_window_splits(n_days: int, min_train_days: int, horizon_days: int, step_days: int):
    splits = []
    train_end = min_train_days
    while train_end + horizon_days <= n_days:
        splits.append((0, train_end, train_end, train_end + horizon_days))
        train_end += step_days
    return splits


def run_backtests(feat_df: pd.DataFrame, holiday_table: pd.DataFrame, cfg: FeatureConfig,
                   horizon_days: int = 90, step_months: int = 3, min_train_months: int = 12) -> dict:
    """Returns {'Prophet': {...}, 'SARIMAX': {...}, 'LightGBM': {...}} each with
    'overall' metrics (averaged across folds) and 'segments' (averaged across
    folds) and 'fold_details' for transparency.
    """
    n = len(feat_df)
    min_train_days = min_train_months * 30
    step_days = step_months * 30
    splits = expanding_window_splits(n, min_train_days, horizon_days, step_days)
    if not splits:
        raise ValueError("Not enough history for the requested backtest configuration.")

    log.info("Running %d backtest fold(s), horizon=%d days each.", len(splits), horizon_days)

    results = {name: {"fold_metrics": [], "fold_segments": [], "preds": []}
               for name in ["Prophet", "SARIMAX", "LightGBM"]}

    for i, (s0, s1, v0, v1) in enumerate(splits, 1):
        train = feat_df.iloc[s0:s1].copy()
        valid = feat_df.iloc[v0:v1].copy()
        log.info("Fold %d: train %s->%s (%d obs), valid %s->%s (%d obs)",
                  i, train["ds"].min().date(), train["ds"].max().date(), len(train),
                  valid["ds"].min().date(), valid["ds"].max().date(), len(valid))

        strike_windows = [(cfg.strike_start, cfg.strike_end)] if cfg.strike_start and cfg.strike_end else []

        # Prophet
        try:
            pred_fn = fit_prophet(train, holiday_table, strike_windows)
            fc = pred_fn(valid)
            eval_df = valid[["ds", "y"] + list(SEGMENT_DEFINITIONS_COLS(valid))].merge(fc, on="ds")
            results["Prophet"]["fold_metrics"].append(compute_metrics(eval_df["y"], eval_df["yhat"]))
            results["Prophet"]["fold_segments"].append(segment_scores(eval_df.rename(columns={"yhat": "pred"}), "pred"))
        except Exception as e:  # noqa: BLE001
            log.warning("Prophet failed on fold %d: %s", i, e)

        # SARIMAX
        try:
            exog_cols = _sarimax_exog_cols(train)
            pred_fn = fit_sarimax(train, exog_cols)
            fc = pred_fn(valid)
            eval_df = valid[["ds", "y"] + list(SEGMENT_DEFINITIONS_COLS(valid))].merge(fc, on="ds")
            results["SARIMAX"]["fold_metrics"].append(compute_metrics(eval_df["y"], eval_df["yhat"]))
            results["SARIMAX"]["fold_segments"].append(segment_scores(eval_df.rename(columns={"yhat": "pred"}), "pred"))
        except Exception as e:  # noqa: BLE001
            log.warning("SARIMAX failed on fold %d: %s", i, e)

        # LightGBM
        try:
            feat_cols = _lgb_feature_cols(train)
            pred_fn, _ = fit_lightgbm(train, feat_cols)
            fc = pred_fn(valid)
            eval_df = valid[["ds", "y"] + list(SEGMENT_DEFINITIONS_COLS(valid))].merge(fc, on="ds")
            results["LightGBM"]["fold_metrics"].append(compute_metrics(eval_df["y"], eval_df["yhat"]))
            results["LightGBM"]["fold_segments"].append(segment_scores(eval_df.rename(columns={"yhat": "pred"}), "pred"))
        except Exception as e:  # noqa: BLE001
            log.warning("LightGBM failed on fold %d: %s", i, e)

    return results


def SEGMENT_DEFINITIONS_COLS(df: pd.DataFrame) -> list[str]:
    needed = {"is_sunday", "is_weekend", "is_holiday", "is_day_before_holiday",
              "is_christmas_period", "is_eid_period", "is_month_end"}
    return [c for c in needed if c in df.columns]


def summarize_backtest(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, r in results.items():
        if not r["fold_metrics"]:
            continue
        dfm = pd.DataFrame(r["fold_metrics"])
        rows.append({
            "Model": model_name,
            "MAE": dfm["MAE"].mean(),
            "RMSE": dfm["RMSE"].mean(),
            "MAPE": dfm["MAPE"].mean(),
            "sMAPE": dfm["sMAPE"].mean(),
            "WAPE": dfm["WAPE"].mean(),
            "n_folds": len(dfm),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. ENSEMBLE WEIGHTING
# ---------------------------------------------------------------------------

def compute_ensemble_weights(summary: pd.DataFrame, metric: str = "WAPE") -> dict:
    """Inverse-error weighting: models with lower WAPE get proportionally
    higher weight. Falls back to equal weights if a model has no valid score.
    """
    valid = summary.dropna(subset=[metric])
    if valid.empty:
        return {}
    inv = 1.0 / valid[metric].clip(lower=1e-6)
    weights = (inv / inv.sum()).to_dict()
    return {row["Model"]: w for row, w in zip(valid.to_dict("records"), weights.values())}


# ---------------------------------------------------------------------------
# 8. FINAL FORECAST (full history -> future horizon)
# ---------------------------------------------------------------------------

def make_future_feature_frame(full_hist: pd.DataFrame, horizon_days: int, cfg: FeatureConfig,
                               holiday_table_future: pd.DataFrame) -> pd.DataFrame:
    """Builds a features dataframe for the forecast horizon. Lag/rolling
    features for future dates are filled iteratively is out of scope for
    Prophet/SARIMAX (they don't need lag features), but LightGBM DOES need
    them — so this performs a recursive one-step-ahead simulation using each
    model's own prior predictions as pseudo-history for the lag features.
    This function returns the calendar/holiday/strike/trend skeleton only;
    lag/rolling values are filled in `recursive_lgb_forecast`.
    """
    last_date = full_hist["ds"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    skeleton = pd.DataFrame({"ds": future_dates, "y": np.nan, "is_imputed": 0})
    combined = pd.concat([full_hist[["ds", "y", "is_imputed"]], skeleton], ignore_index=True)
    feat = build_features(combined, cfg, holiday_table_future)
    return feat.iloc[-horizon_days:].reset_index(drop=True), feat


def recursive_lgb_forecast(full_hist_feat: pd.DataFrame, horizon_days: int, cfg: FeatureConfig,
                            holiday_table_future: pd.DataFrame, feature_cols: list[str],
                            models: dict) -> pd.DataFrame:
    """Recursively forecasts day-by-day with LightGBM, feeding each day's
    P50 prediction back in as history so lag/rolling features stay valid.
    """
    history = full_hist_feat[["ds", "y", "is_imputed"]].copy()
    preds = []
    for step in range(horizon_days):
        future_skel, feat_all = make_future_feature_frame(history, 1, cfg, holiday_table_future)
        row = future_skel.iloc[[0]]
        Xf = row[feature_cols].fillna(0)
        p_mid = models["mid"].predict(Xf)[0]
        p_low = min(models["lower"].predict(Xf)[0], p_mid)
        p_high = max(models["upper"].predict(Xf)[0], p_mid)
        preds.append({"ds": row["ds"].values[0], "yhat": p_mid, "yhat_lower": p_low, "yhat_upper": p_high})
        history = pd.concat(
            [history, pd.DataFrame({"ds": [row["ds"].values[0]], "y": [p_mid], "is_imputed": [0]})],
            ignore_index=True,
        )
    return pd.DataFrame(preds)


def generate_final_forecast(
    full_hist: pd.DataFrame,
    cfg: FeatureConfig,
    horizon_days: int,
    chosen_model: str,
    ensemble_weights: dict | None = None,
    custom_holidays: pd.DataFrame | None = None,
    forecast_start_date: str | None = None,
) -> pd.DataFrame:
    """Trains the chosen model(s) on the FULL history and forecasts
    `horizon_days` ahead. If `forecast_start_date` is provided, the output
    begins on that date instead of the day after the last historical date
    (useful when you need forecasts aligned to a planning horizon such as
    the start of a future month).
    """
    years_needed = sorted(set(full_hist["ds"].dt.year) | {full_hist["ds"].max().year + 1, full_hist["ds"].max().year + 2})
    holiday_table = build_tanzania_holiday_table(years_needed, custom_holidays)
    feat_full = build_features(full_hist, cfg, holiday_table)

    last_date = full_hist["ds"].max()
    actual_start = last_date + pd.Timedelta(days=1)

    if forecast_start_date is not None:
        forecast_start = pd.Timestamp(forecast_start_date)
        if forecast_start < actual_start:
            raise ValueError(
                f"forecast_start_date ({forecast_start.date()}) must be on or after "
                f"the day after the last history date ({actual_start.date()})."
            )
        required_end = forecast_start + pd.Timedelta(days=horizon_days - 1)
        required_horizon = (required_end - actual_start).days + 1
    else:
        forecast_start = actual_start
        required_horizon = horizon_days

    strike_windows = [(cfg.strike_start, cfg.strike_end)] if cfg.strike_start and cfg.strike_end else []
    future_skel, _ = make_future_feature_frame(full_hist, required_horizon, cfg, holiday_table)

    forecasts = {}

    if chosen_model in ("Prophet", "Ensemble"):
        pred_fn = fit_prophet(feat_full, holiday_table, strike_windows)
        forecasts["Prophet"] = pred_fn(future_skel)

    if chosen_model in ("SARIMAX", "Ensemble"):
        exog_cols = _sarimax_exog_cols(feat_full)
        pred_fn = fit_sarimax(feat_full, exog_cols)
        forecasts["SARIMAX"] = pred_fn(future_skel)

    if chosen_model in ("LightGBM", "Ensemble"):
        feat_cols = _lgb_feature_cols(feat_full)
        _, models = fit_lightgbm(feat_full, feat_cols)
        forecasts["LightGBM"] = recursive_lgb_forecast(
            feat_full, required_horizon, cfg, holiday_table, feat_cols, models
        )

    # Filter to the requested forecast start date (if different from actual_start)
    for name in list(forecasts.keys()):
        forecasts[name] = forecasts[name][forecasts[name]["ds"] >= forecast_start].copy()

    if chosen_model != "Ensemble":
        return forecasts[chosen_model]

    # Weighted blend
    weights = ensemble_weights or {k: 1 / len(forecasts) for k in forecasts}
    blended = None
    for name, fc in forecasts.items():
        w = weights.get(name, 0)
        fc = fc.set_index("ds")
        contrib = fc[["yhat", "yhat_lower", "yhat_upper"]] * w
        blended = contrib if blended is None else blended.add(contrib, fill_value=0)
    blended = blended.reset_index()
    return blended


# ---------------------------------------------------------------------------
# 9. MONTHLY AGGREGATION
# ---------------------------------------------------------------------------

def aggregate_monthly(forecast_df: pd.DataFrame) -> pd.DataFrame:
    df = forecast_df.copy()
    df["month"] = pd.to_datetime(df["ds"]).dt.to_period("M")
    agg = df.groupby("month").agg(
        forecast_disbursement=("yhat", "sum"),
        lower=("yhat_lower", "sum"),
        upper=("yhat_upper", "sum"),
    ).reset_index()
    agg["month"] = agg["month"].astype(str)
    return agg


# ---------------------------------------------------------------------------
# 10. MONITORING / RETRAINING HELPERS
# ---------------------------------------------------------------------------

def score_new_actuals(actuals: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Compare realized actuals to a previously issued forecast. Run this
    monthly to track live forecast accuracy (see monitoring notes below).
    """
    merged = actuals.merge(forecast, on="ds", how="inner")
    return compute_metrics(merged["y"], merged["yhat"])


def check_retrain_trigger(rolling_wape_history: list[float], threshold_pct_increase: float = 25.0) -> bool:
    """Simple drift check: if the most recent WAPE is more than
    `threshold_pct_increase`% worse than the trailing 3-period average,
    flag for rebuild/retrain. Feed this a list of monthly WAPE scores.
    """
    if len(rolling_wape_history) < 4:
        return False
    baseline = np.mean(rolling_wape_history[-4:-1])
    latest = rolling_wape_history[-1]
    if baseline <= 0:
        return False
    pct_increase = (latest - baseline) / baseline * 100
    return pct_increase > threshold_pct_increase


RETRAIN_NOTES = """
Retraining cadence
------------------
- Retrain monthly (or after any known abnormal event, e.g. another strike,
  a policy change, a major product change) using ALL available history up
  to that point (expanding window), not a fixed rolling window — trend and
  yearly seasonality need long history.
- After each retrain, re-run the backtest suite on the last 3-4 folds only
  (cheap sanity check) before promoting the new model to production.
- Re-select the champion model (Prophet / SARIMAX / LightGBM / Ensemble)
  every retrain cycle -- do not hardcode it. Relative performance can shift
  as more data arrives.

Monitoring accuracy over time
------------------------------
- Each month, once actuals for the prior month are final, call
  `score_new_actuals(actual_df, forecast_df)` and log MAE/WAPE/sMAPE to a
  tracking table (date, model_version, horizon, MAE, WAPE, sMAPE).
- Track WAPE separately for the full population and for the business
  segments in SEGMENT_DEFINITIONS (Sundays, holidays, month-end, etc.) --
  a model can look fine in aggregate while quietly degrading on a segment
  that matters (e.g. month-end, which drives liquidity planning).
- Plot a rolling 3-month WAPE trend; a sustained upward trend is more
  informative than any single bad month (which can just be one event).

When to rebuild vs. simply retrain
------------------------------------
Retrain (cheap, routine): refit the same model family/features on newer
data on the regular monthly cadence above.

Rebuild (feature/architecture review) when any of the following hold:
  1. `check_retrain_trigger` fires for 2+ consecutive months.
  2. A new recurring business pattern emerges that isn't in the feature set
     (e.g. a new disbursement product, a new partner channel, a policy
     change to loan limits).
  3. The champion model changes in 3 consecutive monthly backtests (a sign
     the underlying data-generating process may be shifting).
  4. A structural break is suspected (e.g. sustained regime change post
     another strike/crisis) -- consider adding a new event regressor the
     same way the 2025 strike was handled, rather than letting the model
     absorb it as noise.
"""


# ---------------------------------------------------------------------------
# 11. MAIN ORCHESTRATION
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame | None = None,
    csv_path: str | None = None,
    date_col: str = "date",
    value_col: str = "disbursement_amount",
    strike_start: str | None = None,
    strike_end: str | None = None,
    backtest_horizon_days: int = 90,
    backtest_step_months: int = 3,
    backtest_min_train_months: int = 12,
    horizons: dict[str, int] | None = None,
    force_model: str | None = None,
    holiday_csv_path: str | None = None,
    custom_holidays_df: pd.DataFrame | None = None,
    forecast_start_date: str | None = None,
    outdir: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Runs the full pipeline (load -> features -> backtest -> select ->
    forecast -> aggregate) and returns everything as in-memory objects, so
    it can be called directly from a notebook or a UI (e.g. Streamlit)
    without touching disk. Pass `outdir` to ALSO write the usual CSV/JSON
    artifacts (same files `main()`/the CLI produce).

    Provide either `df` (a DataFrame with `date_col`/`value_col`) or
    `csv_path` (a path `load_data` will read). `progress_cb`, if given, is
    called with short human-readable status strings at each pipeline stage
    (handy for a Streamlit spinner/status widget).

    Optional holiday inputs:
        - `holiday_csv_path`: path to a CSV with columns `date` and `holiday`
        - `custom_holidays_df`: an already-loaded DataFrame with the same columns
      Custom holidays are merged with the built-in Tanzania calendar and are
      passed to Prophet/SARIMAX/LightGBM feature engineering.

    Optional forecast alignment:
        - `forecast_start_date`: e.g. '2026-09-01'. If provided, the output
          forecast begins on this date instead of the day after the last
          historical date. The horizon length stays the same.

    Returns a dict with keys:
        history, holiday_table, feature_df, backtest_summary,
        backtest_results, ensemble_weights, champion_model,
        forecasts (dict of horizon_label -> daily forecast DataFrame),
        monthly_forecasts (dict of horizon_label -> monthly DataFrame),
        segment_scores (champion's segment scores on the last backtest fold)
    """
    def _report(msg: str):
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    cfg = FeatureConfig(
        strike_start=strike_start,
        strike_end=strike_end,
        strike_recovery_days=14,
    )
    horizons = horizons or {"12m": 365, "24m": 730}

    _report("Loading data...")
    if df is None:
        if csv_path is None:
            raise ValueError("Provide either df or csv_path.")
        df = load_data(csv_path, date_col, value_col)
    else:
        df = df.rename(columns={date_col: "ds", value_col: "y"})[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)
        df["is_imputed"] = 0

    _report("Building Tanzania holiday calendar...")
    years = sorted(set(df["ds"].dt.year) | {df["ds"].max().year + 1, df["ds"].max().year + 2})
    if custom_holidays_df is not None:
        custom_holidays = load_custom_holidays(custom_holidays_df)
    elif holiday_csv_path:
        custom_holidays = load_custom_holidays(holiday_csv_path)
    else:
        custom_holidays = None
    holiday_table = build_tanzania_holiday_table(years, custom_holidays)

    _report("Engineering features...")
    feat_df = build_features(df, cfg, holiday_table)

    _report(f"Backtesting Prophet, SARIMAX, LightGBM "
            f"({backtest_min_train_months}mo initial train, "
            f"{backtest_step_months}mo step, {backtest_horizon_days}d horizon)...")
    bt_results = run_backtests(
        feat_df, holiday_table, cfg,
        horizon_days=backtest_horizon_days,
        step_months=backtest_step_months,
        min_train_months=backtest_min_train_months,
    )
    summary = summarize_backtest(bt_results)

    weights = compute_ensemble_weights(summary, metric="WAPE")
    champion = summary.sort_values("WAPE").iloc[0]["Model"] if not summary.empty else "LightGBM"
    final_choice = force_model or champion
    _report(f"Champion model: {final_choice} (weights: {weights})")

    forecasts, monthly_forecasts = {}, {}
    for label, horizon_days in horizons.items():
        _report(f"Generating {label} forecast ({horizon_days} days) with {final_choice}...")
        fc = generate_final_forecast(
            df, cfg, horizon_days, final_choice, weights, custom_holidays, forecast_start_date
        ).sort_values("ds").reset_index(drop=True)
        forecasts[label] = fc
        monthly_forecasts[label] = aggregate_monthly(fc)

    segment_df = pd.DataFrame()
    if bt_results.get(final_choice, {}).get("fold_segments"):
        segment_df = bt_results[final_choice]["fold_segments"][-1]

    result = {
        "history": df,
        "holiday_table": holiday_table,
        "feature_df": feat_df,
        "backtest_summary": summary,
        "backtest_results": bt_results,
        "ensemble_weights": weights,
        "champion_model": final_choice,
        "forecasts": forecasts,
        "monthly_forecasts": monthly_forecasts,
        "segment_scores": segment_df,
    }

    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        holiday_table.to_csv(outdir / "holiday_table.csv", index=False)
        summary.to_csv(outdir / "model_comparison.csv", index=False)
        for label, fc in forecasts.items():
            fc.to_csv(outdir / f"daily_forecast_{label}.csv", index=False)
            monthly_forecasts[label].to_csv(outdir / f"monthly_forecast_{label}.csv", index=False)
        if not segment_df.empty:
            segment_df.to_csv(outdir / "segment_scores_last_fold.csv", index=False)
        with open(outdir / "retrain_monitoring_notes.txt", "w") as f:
            f.write(RETRAIN_NOTES)
        with open(outdir / "run_summary.json", "w") as f:
            json.dump({
                "champion_model": final_choice,
                "ensemble_weights": weights,
                "n_history_days": len(df),
                "history_start": str(df["ds"].min().date()),
                "history_end": str(df["ds"].max().date()),
            }, f, indent=2)
        _report(f"Artifacts written to {outdir.resolve()}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Daily forecasting pipeline")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--value-col", default="disbursement_amount")
    parser.add_argument("--strike-start", default=None,
                         help="Optional strike/abnormal event start date. Leave blank to disable strike features.")
    parser.add_argument("--strike-end", default=None,
                         help="Optional strike/abnormal event end date. Leave blank to disable strike features.")
    parser.add_argument("--outdir", default="./forecast_output")
    parser.add_argument("--backtest-horizon-days", type=int, default=90)
    parser.add_argument("--backtest-step-months", type=int, default=3)
    parser.add_argument("--backtest-min-train-months", type=int, default=12)
    parser.add_argument("--force-model", default=None,
                         choices=["Prophet", "SARIMAX", "LightGBM", "Ensemble"],
                         help="Skip auto-selection and force this model for the final forecast.")
    parser.add_argument("--holiday-csv", default=None,
                         help="Optional CSV with columns 'date' and 'holiday' to merge with the Tanzania calendar.")
    parser.add_argument("--forecast-start-date", default=None,
                         help="Optional forecast start date, e.g. '2026-09-01'. Defaults to day after last history date.")
    args = parser.parse_args()

    result = run_pipeline(
        csv_path=args.csv,
        date_col=args.date_col,
        value_col=args.value_col,
        strike_start=args.strike_start,
        strike_end=args.strike_end,
        backtest_horizon_days=args.backtest_horizon_days,
        backtest_step_months=args.backtest_step_months,
        backtest_min_train_months=args.backtest_min_train_months,
        force_model=args.force_model,
        holiday_csv_path=args.holiday_csv,
        forecast_start_date=args.forecast_start_date,
        outdir=args.outdir,
    )
    log.info("\n%s", result["backtest_summary"].to_string(index=False))
    log.info("Done.")


if __name__ == "__main__":
    main()
