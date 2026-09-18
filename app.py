import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="KrishiMitra AI — Crop Yield & Price Analytics", page_icon="🌾", layout="wide")

BASE = Path(__file__).parent

# ----------------------------- Load artifacts -----------------------------
@st.cache_resource
def load_yield_model():
    with open(BASE / "yield_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_price_model():
    with open(BASE / "price_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_yield_df():
    return pd.read_csv(BASE / "yield_data.csv")

@st.cache_data
def load_price_df():
    return pd.read_csv(BASE / "price_data.csv", parse_dates=["date"])

yield_artifact = load_yield_model()
price_artifact = load_price_model()
yield_df = load_yield_df()
price_df = load_price_df()

STATE_CROP_MAP = {}
for s in yield_artifact["options"]["state"]:
    STATE_CROP_MAP[s] = sorted(yield_df.loc[yield_df["state"] == s, "crop"].unique().tolist())

# ----------------------------- Sidebar -----------------------------------
st.sidebar.title("🌾 KrishiMitra AI")
st.sidebar.caption("Data-driven crop yield & price advisory for farmers")
page = st.sidebar.radio("Navigate", ["🏠 Advisory Dashboard", "📊 Yield Model Insights", "💰 Price Trends Explorer", "ℹ️ About / Model Info"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built for Smart India Hackathon**\n\n"
    "Theme: Agriculture, FoodTech & Rural Development\n\n"
    "ML models: RandomForest (yield) + Gradient-boosted trees (price forecast)"
)

# ----------------------------- Helper functions ---------------------------

def predict_yield(state, crop, season, soil_type, rainfall, temp, fert, pesticide, irrigation, area):
    row = pd.DataFrame([{
        "state": state, "crop": crop, "season": season, "soil_type": soil_type,
        "year": 2026, "rainfall_mm": rainfall, "avg_temp_c": temp,
        "fertilizer_kg_per_ha": fert, "pesticide_l_per_ha": pesticide,
        "irrigation_coverage_pct": irrigation, "area_hectare": area,
    }])
    pred = yield_artifact["pipeline"].predict(row)[0]
    return max(pred, 0)


def forecast_price(state, crop, months_ahead=6):
    key = (state, crop)
    if key not in price_artifact["models"]:
        return None, None
    model = price_artifact["models"][key]
    hist = price_artifact["histories"][key].copy()

    series = list(hist["modal_price_rs_per_quintal"].values)
    last_date = hist["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=months_ahead, freq="MS")

    preds = []
    for fd in future_dates:
        feat = {
            "lag_1": series[-1], "lag_2": series[-2], "lag_3": series[-3],
            "lag_6": series[-6], "lag_12": series[-12],
            "rolling_mean_3": np.mean(series[-3:]),
            "rolling_std_3": np.std(series[-3:]),
            "month": fd.month,
        }
        X = pd.DataFrame([feat])
        p = float(model.predict(X)[0])
        preds.append(p)
        series.append(p)

    forecast_df = pd.DataFrame({"date": future_dates, "modal_price_rs_per_quintal": preds, "type": "Forecast"})
    hist_tagged = hist.rename(columns={"modal_price_rs_per_quintal": "modal_price_rs_per_quintal"}).copy()
    hist_tagged["type"] = "Historical"
    return hist_tagged, forecast_df


def advisory_text(state, crop, predicted_yield, hist_df, forecast_df):
    avg_hist_price = hist_df["modal_price_rs_per_quintal"].tail(12).mean()
    forecast_peak_idx = forecast_df["modal_price_rs_per_quintal"].idxmax()
    forecast_peak_month = forecast_df.loc[forecast_peak_idx, "date"].strftime("%B %Y")
    forecast_peak_price = forecast_df.loc[forecast_peak_idx, "modal_price_rs_per_quintal"]
    price_trend_pct = (forecast_df["modal_price_rs_per_quintal"].iloc[-1] - avg_hist_price) / avg_hist_price * 100

    lines = []
    lines.append(f"**Expected yield** for {crop} in {state}: **{predicted_yield:.1f} quintal/hectare**.")
    if price_trend_pct > 5:
        lines.append(f"Prices are **trending upward** ({price_trend_pct:+.1f}% vs last 12-month average) — model expects the best mandi price around **{forecast_peak_month}** (₹{forecast_peak_price:,.0f}/quintal). Consider holding stock (if storage is available) instead of selling immediately after harvest.")
    elif price_trend_pct < -5:
        lines.append(f"Prices are **trending downward** ({price_trend_pct:+.1f}% vs last 12-month average). Selling soon after harvest, or exploring government MSP/procurement centers, may reduce loss.")
    else:
        lines.append(f"Prices are expected to stay **relatively stable** (~{price_trend_pct:+.1f}% vs last 12-month average). Normal selling schedule should be fine.")
    return "\n\n".join(lines)


# ----------------------------- Page: Advisory Dashboard --------------------
if page == "🏠 Advisory Dashboard":
    st.title("🌾 Crop Yield & Price Advisory")
    st.caption("Enter your farm details to get an AI-based yield estimate and a 6-month mandi price forecast.")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("Farm Inputs")
        state = st.selectbox("State", yield_artifact["options"]["state"])
        crop = st.selectbox("Crop", STATE_CROP_MAP[state])
        season = st.selectbox("Season", yield_artifact["options"]["season"])
        soil_type = st.selectbox("Soil Type", yield_artifact["options"]["soil_type"],
                                  index=yield_artifact["options"]["soil_type"].index(
                                      yield_df.loc[yield_df["state"] == state, "soil_type"].iloc[0]))

        rainfall = st.slider("Expected Rainfall (mm)", 100, 2200, 900)
        temp = st.slider("Avg Temperature (°C)", 10, 40, 27)
        fert = st.slider("Fertilizer Use (kg/hectare)", 20, 300, 140)
        pesticide = st.slider("Pesticide Use (L/hectare)", 0.0, 5.0, 1.5)
        irrigation = st.slider("Irrigation Coverage (%)", 0, 100, 55)
        area = st.number_input("Farm Area (hectare)", min_value=0.1, value=2.0, step=0.1)

        run = st.button("🔮 Get Advisory", type="primary", use_container_width=True)

    with col2:
        if run:
            pred_yield = predict_yield(state, crop, season, soil_type, rainfall, temp, fert, pesticide, irrigation, area)
            hist, fcst = forecast_price(state, crop, months_ahead=6)

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Yield", f"{pred_yield:.1f} qtl/ha")
            m2.metric("Estimated Total Produce", f"{pred_yield * area:.0f} quintal")
            if fcst is not None:
                m3.metric("6-mo Forecast Peak Price", f"₹{fcst['modal_price_rs_per_quintal'].max():,.0f}/qtl")

            if hist is not None:
                combined = pd.concat([hist.tail(24), fcst], ignore_index=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=combined.loc[combined["type"] == "Historical", "date"],
                    y=combined.loc[combined["type"] == "Historical", "modal_price_rs_per_quintal"],
                    mode="lines", name="Historical Price", line=dict(color="#2E7D32")))
                fig.add_trace(go.Scatter(
                    x=combined.loc[combined["type"] == "Forecast", "date"],
                    y=combined.loc[combined["type"] == "Forecast", "modal_price_rs_per_quintal"],
                    mode="lines+markers", name="Forecast (next 6 months)", line=dict(color="#F57C00", dash="dash")))
                fig.update_layout(title=f"{crop} Mandi Price — {state}", xaxis_title="Month",
                                   yaxis_title="Modal Price (₹/quintal)", height=380,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Advisory")
                st.info(advisory_text(state, crop, pred_yield, hist, fcst))
            else:
                st.warning("No price history available for this state-crop combination in the demo dataset.")
        else:
            st.info("Fill in the farm details and click **Get Advisory** to see your yield prediction and price forecast.")

# ----------------------------- Page: Yield Model Insights -------------------
elif page == "📊 Yield Model Insights":
    st.title("📊 Yield Prediction Model — Insights")

    metrics = yield_artifact["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['r2']:.3f}")
    c2.metric("MAE", f"{metrics['mae']:.2f} qtl/ha")
    c3.metric("RMSE", f"{metrics['rmse']:.2f} qtl/ha")

    st.markdown("#### Feature Importance")
    pipe = yield_artifact["pipeline"]
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocess"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(yield_artifact["categorical_features"])
    all_names = list(cat_names) + yield_artifact["numeric_features"]
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": all_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h", color="importance",
                 color_continuous_scale="Greens", title="Top 15 Features Driving Yield Predictions")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Yield by Crop & State")
    pivot = yield_df.groupby(["crop", "state"])["yield_quintal_per_hectare"].mean().reset_index()
    fig2 = px.density_heatmap(pivot, x="state", y="crop", z="yield_quintal_per_hectare",
                               color_continuous_scale="YlGn", title="Avg Yield (qtl/ha) Heatmap")
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Rainfall vs Yield")
    fig3 = px.scatter(yield_df, x="rainfall_mm", y="yield_quintal_per_hectare", color="crop",
                       trendline="ols", opacity=0.6, title="Rainfall vs Yield by Crop")
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------- Page: Price Trends Explorer ------------------
elif page == "💰 Price Trends Explorer":
    st.title("💰 Mandi Price Trends Explorer")

    col1, col2 = st.columns(2)
    with col1:
        state_sel = st.selectbox("State", sorted(price_df["state"].unique()))
    with col2:
        crop_options = sorted(price_df.loc[price_df["state"] == state_sel, "crop"].unique())
        crop_sel = st.selectbox("Crop", crop_options)

    filtered = price_df[(price_df["state"] == state_sel) & (price_df["crop"] == crop_sel)].sort_values("date")
    fig = px.line(filtered, x="date", y="modal_price_rs_per_quintal",
                   title=f"{crop_sel} Price History — {state_sel} (2019–2024)", markers=False)
    fig.update_traces(line_color="#2E7D32")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    key = (state_sel, crop_sel)
    if key in price_artifact["metrics"]:
        st.caption(f"Model holdout MAPE for this crop-state pair: **{price_artifact['metrics'][key]:.2%}**")

    st.markdown("#### Compare Multiple Crops (same state)")
    multi_crops = st.multiselect("Select crops to compare", crop_options, default=crop_options[:2])
    if multi_crops:
        multi_df = price_df[(price_df["state"] == state_sel) & (price_df["crop"].isin(multi_crops))]
        fig2 = px.line(multi_df, x="date", y="modal_price_rs_per_quintal", color="crop",
                        title=f"Price Comparison — {state_sel}")
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------- Page: About ----------------------------------
else:
    st.title("ℹ️ About This Project")
    st.markdown(f"""
### KrishiMitra AI — Crop Yield & Price Advisory Platform

**Problem it solves:** Small and marginal farmers often lack timely, data-driven
guidance on expected yield and the best time/place to sell their produce,
leading to distress sales and income loss.

**What this prototype does:**
- Predicts crop yield (quintal/hectare) from weather, soil, and input-use data using a **Random Forest Regressor**
  (R² = {yield_artifact['metrics']['r2']:.3f} on held-out data).
- Forecasts mandi prices 6 months ahead per state-crop combination using **gradient-boosted trees**
  ({'XGBoost' if price_artifact['has_xgb'] else 'GradientBoosting'}) trained on lag/rolling-window features
  (avg holdout MAPE ≈ {price_artifact['avg_mape']:.2%}).
- Converts predictions into a **plain-language advisory** (sell now vs. hold, expected peak price month).

**Data note:** This demo uses a *synthetic dataset* generated to statistically resemble real
Agmarknet (mandi price), IMD (rainfall), and state agriculture department (yield) records,
since live government data portals aren't reachable in this environment. Swapping in real
CSVs from [data.gov.in](https://data.gov.in) / [Agmarknet](https://agmarknet.gov.in) requires
no code changes to the model pipeline — only `generate_data.py` → replace with real data loading.

**Tech stack:** Python, scikit-learn, {'XGBoost' if price_artifact['has_xgb'] else 'GradientBoosting'}, Streamlit, Plotly, Pandas.

**Extending for SIH finale:**
- Add regional-language (Hindi/Punjabi/etc.) UI via a translation layer.
- Integrate live weather API (IMD/OpenWeather) instead of manual sliders.
- Add SMS/WhatsApp advisory delivery for farmers without smartphones.
- Crop recommendation module (classification model) for next-season planning.
""")
