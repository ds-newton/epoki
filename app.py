"""
Daily Forecasting Streamlit App
======================================

Upload a daily disbursement CSV and the pipeline will:
  1. Build calendar / holiday / strike features
  2. Backtest Prophet, SARIMAX and LightGBM
  3. Pick a champion model (or use an ensemble)
  4. Produce 12-month and 24-month daily/monthly forecasts
  5. Show model comparison, segment scores and downloadable outputs

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from daily_forecaster import run_pipeline

st.set_page_config(
    page_title="Epoki - The Daily Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def _download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    csv = _to_csv(df)
    b64 = base64.b64encode(csv.encode()).decode()
    return (
        f'<a href="data:file/csv;base64,{b64}" download="{filename}" '
        f'style="text-decoration:none;">📥 {label}</a>'
    )


def _status_callback(msg: str):
    """Updates the global status container used during pipeline execution."""
    if "_status_area" in st.session_state:
        st.session_state["_last_status"] = msg
        st.session_state["_status_area"].info(msg)


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")

    st.markdown("### Column mapping")
    date_col = st.text_input("Date column name", value="date")
    value_col = st.text_input("Disbursement column name", value="amount")

    st.markdown("### Strike / abnormal event")
    include_strike = st.checkbox("Include strike event", value=False)
    strike_start, strike_end = None, None
    if include_strike:
        strike_start = st.date_input("Strike start", value=datetime(2025, 10, 29))
        strike_end = st.date_input("Strike end", value=datetime(2025, 11, 2))
        strike_start = str(strike_start)
        strike_end = str(strike_end)

    st.markdown("### Backtest settings")
    min_train_months = st.slider("Minimum training months", 6, 36, 12)
    step_months = st.slider("Step months", 1, 12, 3)
    horizon_days = st.slider("Backtest horizon (days)", 30, 180, 90)

    st.markdown("### Model selection")
    force_model = st.selectbox(
        "Force a specific model (optional)",
        ["Auto-select champion", "Prophet", "SARIMAX", "LightGBM", "Ensemble"],
    )

    st.markdown("### Forecast horizons")
    run_12m = st.checkbox("12-month forecast", value=True)
    run_24m = st.checkbox("24-month forecast", value=True)

    st.markdown("### Forecast alignment")
    use_forecast_start = st.checkbox("Start forecast from a specific date", value=False)
    forecast_start_date = None
    if use_forecast_start:
        forecast_start_date = st.date_input(
            "Forecast start date",
            value=datetime(2026, 9, 1),
            help="If set, the output forecast begins on this date instead of the day after the last history date.",
        )
        forecast_start_date = str(forecast_start_date)

    st.markdown("---")
    st.caption(
        "Upload a CSV with one row per calendar day. The app will interpolate "
        "missing days and run three candidate models in the background."
    )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("📈 Epoki - The Daily Forecaster")
st.markdown(
    "Upload your `disb_data.csv` and the pipeline will train Prophet, SARIMAX "
    "and LightGBM, compare them on expanding-window backtests, and produce "
    "12-month / 24-month forecasts."
)

uploaded_file = st.file_uploader(
    "Upload disbursement CSV",
    type=["csv"],
    help="Must contain a date column and a numeric disbursement column.",
)

holiday_file = st.file_uploader(
    "Upload optional holiday CSV (columns: date, holiday)",
    type=["csv"],
    help="Optional. Merged with the built-in Tanzania calendar. Prophet will use these as regressors.",
)

if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()

# Read uploaded files once and cache them in session state
df_raw = pd.read_csv(uploaded_file)
custom_holidays_df = None
if holiday_file is not None:
    custom_holidays_df = pd.read_csv(holiday_file)

st.markdown("---")
st.subheader("🔍 Uploaded data preview")

if date_col not in df_raw.columns or value_col not in df_raw.columns:
    st.error(
        f"Could not find columns **{date_col}** and/or **{value_col}** in the uploaded file. "
        f"Available columns: {', '.join(df_raw.columns)}"
    )
    st.stop()

preview = df_raw.copy()
preview[date_col] = pd.to_datetime(preview[date_col], errors="coerce")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(preview))
col2.metric("Date range", f"{preview[date_col].min().date()} → {preview[date_col].max().date()}")
col3.metric("Total disbursement", f"{preview[value_col].sum():,.0f}")

st.dataframe(preview.head(10), use_container_width=True)

# Build the selected horizon map
horizons: dict[str, int] = {}
if run_12m:
    horizons["12m"] = 365
if run_24m:
    horizons["24m"] = 730

if not horizons:
    st.warning("Select at least one forecast horizon in the sidebar.")
    st.stop()

force_model_value = None if force_model == "Auto-select champion" else force_model

run_button = st.button("🚀 Run forecasting pipeline", type="primary", use_container_width=True)

if run_button:
    # Prepare a status area
    st.session_state["_status_area"] = st.empty()
    st.session_state["_last_status"] = ""

    with st.spinner("Running models in the background... this may take a minute or two."):
        try:
            result = run_pipeline(
                df=df_raw,
                date_col=date_col,
                value_col=value_col,
                strike_start=strike_start,
                strike_end=strike_end,
                backtest_horizon_days=horizon_days,
                backtest_step_months=step_months,
                backtest_min_train_months=min_train_months,
                horizons=horizons,
                force_model=force_model_value,
                custom_holidays_df=custom_holidays_df,
                forecast_start_date=forecast_start_date,
                progress_cb=_status_callback,
            )
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    # Store results for this run
    st.session_state["forecast_result"] = result
    st.success("Pipeline complete!")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if "forecast_result" not in st.session_state:
    st.info("Click **Run forecasting pipeline** to see model results.")
    st.stop()

result = st.session_state["forecast_result"]

st.markdown("---")
st.subheader("🏆 Champion model")

champion = result["champion_model"]
weights = result["ensemble_weights"]

champ_col, weight_col = st.columns([1, 2])
champ_col.metric("Selected model", champion)
if weights:
    weight_col.json(weights)
else:
    weight_col.write("No ensemble weights computed (auto-selection used a single model).")

st.markdown("---")
st.subheader("📊 Model comparison (expanding-window backtest)")

summary = result["backtest_summary"]
if summary.empty:
    st.warning("No backtest summary available.")
else:
    # Highlight champion row
    def _highlight_champion(row):
        return ["background-color: #d4edda" if row["Model"] == champion else "" for _ in row]

    st.dataframe(
        summary.style.apply(_highlight_champion, axis=1).format(
            {c: "{:.2f}" for c in ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE"]}
        ),
        use_container_width=True,
    )

    # Bar chart of WAPE by model
    st.bar_chart(summary.set_index("Model")[["WAPE", "MAE", "RMSE"]])

st.markdown("---")
st.subheader("🔎 Champion segment scores (last backtest fold)")

segment_df = result["segment_scores"]
if segment_df is None or segment_df.empty:
    st.info("No segment scores available for the champion model.")
else:
    st.dataframe(
        segment_df.style.format(
            {c: "{:.2f}" for c in ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE"]}
        ),
        use_container_width=True,
    )
    st.bar_chart(segment_df.set_index("segment")[["WAPE", "MAE"]])

st.markdown("---")
st.subheader("📈 Forecasts")

for label, fc in result["forecasts"].items():
    monthly = result["monthly_forecasts"][label]

    with st.expander(f"{label} forecast", expanded=(label == "12m")):
        tab_daily, tab_monthly = st.tabs(["Daily", "Monthly"])

        with tab_daily:
            chart_df = fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
            st.line_chart(chart_df)
            st.dataframe(
                fc.style.format({
                    "yhat": "{:.2f}",
                    "yhat_lower": "{:.2f}",
                    "yhat_upper": "{:.2f}",
                }),
                use_container_width=True,
            )
            st.markdown(
                _download_link(fc, f"daily_forecast_{label}.csv", f"Download daily {label} forecast"),
                unsafe_allow_html=True,
            )

        with tab_monthly:
            st.bar_chart(
                monthly.set_index("month")[["forecast_disbursement", "lower", "upper"]]
            )
            st.dataframe(
                monthly.style.format({
                    "forecast_disbursement": "{:.2f}",
                    "lower": "{:.2f}",
                    "upper": "{:.2f}",
                }),
                use_container_width=True,
            )
            st.markdown(
                _download_link(monthly, f"monthly_forecast_{label}.csv", f"Download monthly {label} forecast"),
                unsafe_allow_html=True,
            )

st.markdown("---")
st.subheader("💾 Download all results")

# Build a zip of all artifacts in memory
import zipfile

zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("model_comparison.csv", _to_csv(result["backtest_summary"]))
    if not segment_df.empty:
        zf.writestr("segment_scores_last_fold.csv", _to_csv(segment_df))
    for label, fc in result["forecasts"].items():
        zf.writestr(f"daily_forecast_{label}.csv", _to_csv(fc))
        zf.writestr(
            f"monthly_forecast_{label}.csv",
            _to_csv(result["monthly_forecasts"][label]),
        )
    run_summary = {
        "champion_model": champion,
        "ensemble_weights": weights,
        "n_history_days": len(result["history"]),
        "history_start": str(result["history"]["ds"].min().date()),
        "history_end": str(result["history"]["ds"].max().date()),
    }
    zf.writestr("run_summary.json", json.dumps(run_summary, indent=2))

zip_buffer.seek(0)
st.download_button(
    label="Download all results (ZIP)",
    data=zip_buffer,
    file_name="daily_forecaster_results.zip",
    mime="application/zip",
    use_container_width=True,
)

st.caption(
    "Tip: Use the notebook `review_models.ipynb` for deeper model diagnostics, "
    "custom plots and reproducible research."
)
