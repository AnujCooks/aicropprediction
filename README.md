# 🌾 KrishiMitra AI — Crop Yield & Price Analytics (SIH Project)

An ML-powered analytics platform that predicts **crop yield** and forecasts
**6-month mandi prices** to give farmers a simple, actionable advisory
("sell now" vs "hold stock", expected best-price month).

## Problem it solves
Small/marginal farmers often can't predict yield in advance or time their
sale to the market — leading to distress sales and avoidable income loss.
This tool gives them a data-backed advisory in plain language.

## What's inside
| File | Purpose |
|---|---|
| `generate_data.py` | Builds the synthetic (but statistically realistic) yield + mandi-price dataset. Swap this for real Agmarknet/data.gov.in CSVs with no other code changes. |
| `train_yield_model.py` | Trains a **Random Forest Regressor** to predict yield (quintal/hectare) from weather, soil & input-use data. |
| `train_price_model.py` | Trains per-(state, crop) **gradient-boosted tree** models (XGBoost) on lag/rolling-window features for price forecasting. |
| `app.py` | The **Streamlit dashboard** — farmer inputs → yield prediction + 6-month price forecast chart + advisory text, plus model-insight and price-explorer pages. |
| `yield_data.csv`, `price_data.csv` | Generated datasets. |
| `yield_model.pkl`, `price_model.pkl` | Trained model artifacts (pipelines ready to load). |

## Model performance (on synthetic data)
- Yield model: R² ≈ 0.997, MAE ≈ 3.5 quintal/ha
- Price forecast models: average holdout MAPE ≈ 7%

## How to run locally

```bash
pip install -r requirements.txt

# (Optional — artifacts are already generated, only needed if you change the data)
python generate_data.py
python train_yield_model.py
python train_price_model.py

streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Gemini API key

Set `GEMINI_API_KEY` as an environment variable, or create
`.streamlit/secrets.toml` with:

```toml
GEMINI_API_KEY = "your-api-key"
```

Keep this file local; it is excluded from Git.

## App pages
1. **🏠 Advisory Dashboard** — enter farm details (state, crop, season, soil,
   rainfall, temperature, fertilizer/pesticide use, irrigation %, area) →
   get predicted yield, total produce estimate, a 6-month price forecast
   chart, and a plain-language advisory.
2. **📊 Yield Model Insights** — feature importance, yield heatmap by
   crop/state, rainfall-vs-yield scatter with trendline.
3. **💰 Price Trends Explorer** — historical mandi price charts, multi-crop
   comparison, per-model accuracy (MAPE).
4. **ℹ️ About / Model Info** — project write-up for judges.

## Using real government data instead of synthetic data
Replace the data-loading step in `generate_data.py` (or write a new loader)
with real sources:
- **Prices**: [Agmarknet](https://agmarknet.gov.in) daily mandi price bulletins
- **Yield/production**: [data.gov.in](https://data.gov.in) — Ministry of
  Agriculture district-wise yield statistics
- **Rainfall**: IMD gridded rainfall data

Keep the same column names used in `yield_data.csv` / `price_data.csv` and
the training scripts will work unchanged.

## Suggested next steps for the SIH finale round
- Regional-language UI (Hindi/Punjabi/Marathi etc.)
- Live weather API integration instead of manual sliders
- WhatsApp/SMS delivery of the advisory for farmers without smartphones
- A crop-recommendation classifier for next-season planning
- District-level granularity instead of state-level

## Pitch structure (suggested)
1. **Problem** — distress sales, no yield visibility, income volatility for
   small farmers (cite MoA/NITI Aayog stats on farmer income & price crashes).
2. **Solution** — this dashboard: predict → forecast → advise, in one flow.
3. **Tech** — ML pipeline (RF + gradient boosting), explainable feature
   importance, deployable as a lightweight web/mobile app.
4. **Impact** — quantify: X% better sell-timing → ₹Y average gain per
   hectare (use your own back-of-envelope estimate based on price volatility
   shown in the Price Explorer page).
5. **Scalability** — swap-in real government data sources, extendable to
   more states/crops, SMS delivery for last-mile reach.
