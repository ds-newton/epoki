# 📈 Epoki — The Daily Forecaster

> Turn historical patterns into actionable future forecasts.

**Epoki** is a Streamlit app for forecasting **daily business volumes**. It takes historical daily data, engineers calendar/event features, trains multiple models, backtests them, selects a champion (or builds an ensemble), and produces daily + monthly forecasts up to 24 months out.

Built originally for disbursement forecasting, Epoki works as a general-purpose tool for any daily time series (sales, transactions, collections, service demand, etc.).

---

## How It Works

```
Historical Data → Feature Engineering (calendar, holidays, events)
                → Train Prophet / SARIMAX / LightGBM
                → Expanding-Window Backtesting
                → Champion Selection or Ensemble
                → Daily & Monthly Forecasts (12mo / 24mo)
```

---

## Key Features

- **Flexible CSV upload** — default columns `date` + `disbursement_amount`, but column names are configurable. Data is auto-inspected (row count, date range, totals, preview).
- **Calendar & holiday features** — day-of-week, month, built-in Tanzania holidays, or custom holiday CSVs.
- **Abnormal event handling** — mark strike/outage/disruption date ranges so they don't get learned as normal seasonality.
- **Three forecasting models**:
  - *Prophet* — trend, seasonality, holiday effects
  - *SARIMAX* — statistical AR/MA/seasonal modeling
  - *LightGBM* — gradient boosting on engineered features (lags, rolling stats, calendar/holiday/event indicators)
- **Ensemble mode** — combines all three models using calculated weights instead of picking one.
- **Auto champion selection** — compares models via backtesting (MAE, RMSE, MAPE, sMAPE, WAPE) and picks the best, or you can override manually.
- **Expanding-window backtesting** — repeatedly trains on growing historical windows and tests on rolling future periods, so accuracy reflects real forecasting conditions, not just curve-fitting.
  - Minimum training months: 12 (range 6–36)
  - Backtest horizon: 90 days (range 30–180)
  - Step size between folds: 3 months (range 1–12)
- **Segment-level diagnostics** — checks whether the champion model performs consistently across different parts of the last backtest fold.
- **Configurable forecast horizons** — 12-month and/or 24-month, independently enabled, with an optional custom forecast start date.
- **Downloadable outputs** — daily/monthly CSVs per horizon, or one ZIP bundle containing model comparison, segment scores, all forecasts, and a `run_summary.json`.

---

## Forecast Metrics

| Metric | What it measures |
|--------|-------------------|
| MAE    | Average absolute error |
| RMSE   | Absolute error, penalizing large misses more |
| MAPE   | Percentage error (careful near zero actuals) |
| sMAPE  | Symmetric percentage error |
| WAPE   | Volume-weighted percentage error — best for total-volume accuracy |

---

## Project Structure

```
epoki/
├── app.py                    # Streamlit UI
├── nivusheplus_forecast.py   # Forecasting pipeline & model logic
├── review_models.ipynb       # Model diagnostics / research notebook
├── requirements.txt
├── README.md
└── data/
    └── sample_data.csv
```

---

## Installation & Usage

```bash
git clone https://github.com/ds-newton/epoki.git
cd epoki

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## Input Data

**Required CSV:**
```csv
date,disbursement_amount
2025-01-01,125000000
2025-01-02,138000000
```

**Optional holiday CSV:**
```csv
date,holiday
2026-01-01,New Year
2026-04-03,Good Friday
```

**Data quality tips:** chronological, numeric target, consistent daily frequency, duplicates/missing dates understood, abnormal events flagged, and enough history to cover seasonal cycles. Epoki can interpolate missing calendar days automatically.

---

## Tech Stack

- **App:** Python, Streamlit
- **Data:** Pandas, NumPy
- **Forecasting:** Prophet, Statsmodels (SARIMAX), LightGBM
- **Output:** CSV, JSON, ZIP

---

## Use Cases

Financial services (disbursements, collections, deposits, transactions), sales/demand forecasting, operations (service requests, workload), and planning (budgets, targets, capacity).

---

## Roadmap

More models (XGBoost, Random Forest, LSTM, TFT), automated feature engineering & anomaly detection, dynamic ensemble weighting, forecast explainability, drift monitoring, multi-series/hierarchical forecasting, database connectors, API access, scheduled forecasts, auth, cloud deployment, Excel/PDF export.

---

## Disclaimer

Forecasts are estimates based on historical patterns and available features. Actual outcomes can diverge due to unexpected events, structural or policy changes, market conditions, or data quality issues. Use forecasts as decision support alongside business judgment.

---

## Author

**Newton Mwalongo** — Data Scientist | Business Intelligence | FinTech Analytics
> *Born to stand in the gap between data and business insights.*

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.
